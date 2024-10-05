import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.backward_compatibility import worker_init_fn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics


from datamodules import HEDataset, HEDataPipe, HEDataModule, HEFoundationDataset, HEFoundationDataPipe, HEFoundationDataModule
from classifier_model import compute_binary_accuracy_metrics, get_class_weights

import os
os.environ["STAGE"]="prod"
import sys
import dill

import emdatabase as emdb
from emimagepatch import EMCellPatchDataset

import argparse
from typing import List
import wandb

class CNNEncoder(nn.Module):
    def __init__(self, 
            input_channels=3, 
            input_patch_size=64,
            num_layers=4, 
            num_filters=64,
            output_size=128,
            batch_norm=False,
            track_running_stats=True
            ):
        super(CNNEncoder, self).__init__()
        layers = []
        in_channels = input_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU())
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_filters, track_running_stats=track_running_stats))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters
            num_filters *= 2
        self.cnn = nn.Sequential(*layers)
        self.enc_size = self._get_enc_output((input_channels, input_patch_size, input_patch_size))
        self.fc = nn.Linear(in_features=self.enc_size, out_features=output_size)  

    def _get_enc_output(self, shape):
        """Run a forward pass of encoder to get the size of the encoder output"""
        batch_size = 1
        input_var = torch.autograd.Variable(torch.rand(batch_size, *shape))
        
        output_feat = self.cnn(input_var)
        n_size = output_feat.data.view(batch_size, -1).size(1)

        return n_size

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FCEncoder(nn.Module):
    def __init__(self, 
            input_size, 
            hidden_size=512,
            output_size=128,
            batch_norm=False,
            dropout=None,
            ):
        super(FCEncoder, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(2):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
            hidden_size = output_size
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x

class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    Adapted from https://theaisummer.com/simclr/
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=proj_1.get_device())).float()
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

class CustomFlipRotationTransform:
    """Flip and rotate images according to input parameters"""

    def __call__(self, x, vflip=False, hflip=False, rotate=0):
        if vflip:
            x = TF.vflip(x)
        if hflip:
            x = TF.hflip(x)
        if rotate:
            x = TF.rotate(x, rotate)
        return x

class SimCLRModel(nn.Module):
    def __init__(self, 
            he_encoder_type,
            input_dim=[3], # for cnn encoders, input_dim has a single item, the number of channels. For fc encoder, input_dim is a list of two: [CODEX input_channels, H&E input_dim ]`
            input_patch_size=64,
            num_layers=4, 
            embedding_size=128,
            projection_dim=64, 
            num_filters=64,
            batch_norm=False,
            track_running_stats=True,
            dropout=None
            ):
        super(SimCLRModel, self).__init__()
        if he_encoder_type not in ['cnn', 'fc']:
            raise ValueError('encoder_type must be either "cnn" or "fc"')
        if he_encoder_type == 'cnn':
            assert len(input_dim) == 1, 'For CNN encoders, input_dim should be a list of a single item, the number of channels'
        else:
            assert len(input_dim) == 2, 'For FC encoders, input_dim should be a list of two items, the number of channels for CODEX and H&E images'
        self.encoder = CNNEncoder(input_dim[0], input_patch_size, num_layers, num_filters, embedding_size, batch_norm,) #track_running_stats)
        if he_encoder_type == 'fc': # H&E encoder is an FC encoder, but CODEX encoder is a CNN encoder
            self.he_encoder = FCEncoder(input_dim[1], embedding_size, embedding_size, batch_norm, dropout)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, projection_dim)
        )

    def forward(self, x, encoder_type='cnn'):
        if encoder_type not in ['cnn', 'fc']:
            raise ValueError('encoder_type must be either "cnn" or "fc"')
        z = self.encoder(x) if encoder_type == 'cnn' else self.he_encoder(x)
        z = F.normalize(z, dim=1)  # Normalize the embeddings
        projection = self.projection_head(z)
        return z, projection

class SimCLRModule(pl.LightningModule):
    def __init__(self, 
            he_encoder_type, # 'cnn' or 'fc'
            he_channels,
            codex_channels, 
            input_patch_size,
            num_layers,
            embedding_size=128,
            encoder_filter_dim=64,
            batch_norm=False,
            track_running_stats=True,
            projection_dim=128, 
            temperature=0.5, 
            lr=1e-3,
            zero_one_normalize=True,
            diff_transform=False,
            rescale_to_64=False,
            dropout=None
            ):
        super(SimCLRModule, self).__init__()
        if he_encoder_type not in ['cnn', 'fc']:
            raise ValueError('he_encoder_type must be either "cnn" or "fc"')
        self.save_hyperparameters()
        self.he_encoder_type = he_encoder_type
        input_dim_args = [encoder_filter_dim] if he_encoder_type == 'cnn' else [encoder_filter_dim, he_channels]
        self.model = SimCLRModel(he_encoder_type, input_dim_args, input_patch_size, num_layers, embedding_size, projection_dim, num_filters=encoder_filter_dim, batch_norm=batch_norm, track_running_stats=track_running_stats, dropout=dropout)
        self.embedding_size = embedding_size
        self.temperature = temperature
        self.lr = lr
        self.zero_one_normalize = zero_one_normalize
        self.diff_transform = diff_transform
        self.rescale_to_64 = rescale_to_64
        self.dropout = dropout
        
        self.loss = ContrastiveLoss(temperature=temperature)

        # Separate codex and H&E layers before main CNN encoder
        self.codex_model = nn.Sequential(
                nn.Conv2d(codex_channels, encoder_filter_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                )
        self.he_model = nn.Sequential(
                nn.Conv2d(he_channels, encoder_filter_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                ) if he_encoder_type == 'cnn' else nn.Identity()

        self.custom_transformer = CustomFlipRotationTransform()

        self.apply(self._init_weights)

    def _init_weights(self, m, init_method=nn.init.kaiming_normal_):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init_method(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        return

    def _get_data_for_step(self, batch, train=True, embed=False):
        # Dataloader is for a batch of CODEX images, need to fetch the H&E images
        x = batch[0] # batch is tuple (codex image, h&E dataset output, cell_id)
        he_imgs, labels = batch[1]
        cell_ids = batch[2] # cell id is like CITN10Co-88_c001_v001_r001_reg073_3179
        this_batch_size = x.shape[0]

        # convert all to float
        x = x.float()
        he_imgs = he_imgs.float().to(self.device)

        if train and not embed:
            # Perform random flips and 90 degree rotations
            vflip = torch.rand(1) > 0.5
            hflip = torch.rand(1) > 0.5
            rotate = torch.randint(0, 4, (1,)).item() * 90
            x = self.custom_transformer(x, vflip=vflip, hflip=hflip, rotate=rotate)
            if self.he_encoder_type == 'cnn':
                if self.diff_transform:
                    vflip = torch.rand(1) > 0.5
                    hflip = torch.rand(1) > 0.5
                    rotate = torch.randint(0, 4, (1,)).item() * 90
                he_imgs = self.custom_transformer(he_imgs, vflip=vflip, hflip=hflip, rotate=rotate)

        if self.zero_one_normalize:
            x = x / 255.
            if self.he_encoder_type == 'cnn':
                he_imgs = he_imgs / 255.

        if self.rescale_to_64:
            x = TF.resize(x, (64, 64))
            if self.he_encoder_type == 'cnn':
                he_imgs = TF.resize(he_imgs, (64, 64))

        return x, he_imgs, labels, this_batch_size

    def _log_metrics(self, batch_size, loss, train=True):
        prefix = 'train' if train else 'val'
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        return

    def forward(self, x, encoder_type='cnn'):
        _, projection = self.model(x, encoder_type=encoder_type)
        return projection

    def training_step(self, batch, batch_idx):
        codex_imgs, he_imgs, labels, this_batch_size = self._get_data_for_step(batch, train=True)
        codex_imgs = self.codex_model(codex_imgs)
        codex_z, codex_proj = self.model(codex_imgs, encoder_type='cnn')
        he_imgs = self.he_model(he_imgs)
        he_z, he_proj = self.model(he_imgs, encoder_type=self.he_encoder_type)
        
        loss = self.loss(codex_proj, he_proj)
        self._log_metrics(this_batch_size, loss, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        codex_imgs, he_imgs, labels, this_batch_size = self._get_data_for_step(batch, train=False)
        codex_imgs = self.codex_model(codex_imgs)
        codex_z, codex_proj = self.model(codex_imgs, encoder_type='cnn')
        he_imgs = self.he_model(he_imgs)
        he_z, he_proj = self.model(he_imgs, encoder_type=self.he_encoder_type)
        
        loss = self.loss(codex_proj, he_proj)
        self._log_metrics(this_batch_size, loss, train=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def embed(self, x, he=False):
        """ Returns the embedding of x """
        if he:
            x = self.he_model(x)
            z, _ = self.model(x, encoder_type=self.he_encoder_type)
        else:
            x = self.codex_model(x)
            z, _ = self.model(x, encoder_type='cnn')
        return z

# Create a Lightning module for supervised fine tuning of the SimCLR model
class SupervisedModule(pl.LightningModule):
    def __init__(self,
            simclr_module: SimCLRModule,
            train_acq_ids: list,
            celltype_csv: str ='data/renal-transplant/renal-transplant_celltypes_hierarchy.csv',
            hierarchy: int = None,
            lr: float = 1e-3,
            freeze_encoder: bool = False,
            include_contrastive_loss: bool = True,
            temperature: float = 0.5,
            enforce_same_prediction: bool = False,
            loss_term_weight: float = .5,
            weight_loss: bool = True,
            dropout: float = None,
            ):
        super(SupervisedModule, self).__init__()
        self.save_hyperparameters(ignore=['simclr_module','simclr_model', 'simclr_he_model', 'simclr_codex_model'])

        self.dropout = dropout

        # Get celltype label information
        celltype_df = pd.read_csv(celltype_csv)
        self.hierarchy = hierarchy
        if self.hierarchy is None or self.hierarchy == -1:
            celltype_label_col = 'celltype_label'
        else:
            celltype_label_col = f'celltype_hierarchy_{self.hierarchy}'
        # Assigning unique integer ids to each celltype label
        celltype_df['celltype_id'] = pd.factorize(celltype_df[celltype_label_col])[0]
        self.num_classes = len(np.unique(celltype_df.celltype_id))
        self.class_name_dict = pd.Series(celltype_df[celltype_label_col].values, index=celltype_df.celltype_id).to_dict()
        class_counts = torch.from_numpy(celltype_df[celltype_df.acquisition_id.isin(train_acq_ids)].value_counts('celltype_id', sort=False).values)

        self.class_weights = get_class_weights(self.num_classes, class_counts)

        # Get the SimCLR model + any relevant hyperparameters
        self.simclr_model = simclr_module.model
        self.simclr_he_model = simclr_module.he_model
        self.simclr_codex_model = simclr_module.codex_model
        self.diff_transform = simclr_module.diff_transform
        self.rescale_to_64 = simclr_module.rescale_to_64
        self.he_encoder_type = simclr_module.he_encoder_type

        if freeze_encoder:
            # Freeze the encoder
            for param in self.simclr_model.parameters():
                param.requires_grad = False
            for param in self.simclr_he_model.parameters():
                param.requires_grad = False
            for param in self.simclr_codex_model.parameters():
                param.requires_grad = False

        self.zero_one_normalize = simclr_module.zero_one_normalize
        self.lr = lr

        # Add a linear layer to map the embeddings to the number of classes
        self.fc = nn.Linear(simclr_module.embedding_size, self.num_classes)
        if self.dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

        self.loss = nn.NLLLoss(weight=self.class_weights) if weight_loss else nn.NLLLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature) if include_contrastive_loss else None
        self.same_prediction_loss = nn.CrossEntropyLoss() if enforce_same_prediction else None
        self.loss_term_weight = loss_term_weight

        self.custom_transformer = CustomFlipRotationTransform()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1, average=None)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, top_k=1, average=None)

    def _get_data_for_step(self, batch, train=True, embed=False):
        # Dataloader is for a batch of CODEX images, need to fetch the H&E images
        x = batch[0] # batch is tuple (codex image, h&E dataset output, cell_id)
        he_imgs, labels = batch[1]
        cell_ids = batch[2] # cell id is like CITN10Co-88_c001_v001_r001_reg073_3179
        this_batch_size = x.shape[0]

        # convert all to float
        x = x.float()
        he_imgs = he_imgs.float().to(self.device)

        if train and not embed:
            # Perform random flips and 90 degree rotations
            vflip = torch.rand(1) > 0.5
            hflip = torch.rand(1) > 0.5
            rotate = torch.randint(0, 4, (1,)).item() * 90
            x = self.custom_transformer(x, vflip=vflip, hflip=hflip, rotate=rotate)
            if self.he_encoder_type == 'cnn':
                if self.diff_transform:
                    vflip = torch.rand(1) > 0.5
                    hflip = torch.rand(1) > 0.5
                    rotate = torch.randint(0, 4, (1,)).item() * 90
                he_imgs = self.custom_transformer(he_imgs, vflip=vflip, hflip=hflip, rotate=rotate)

        if self.zero_one_normalize:
            x = x / 255.
            if self.he_encoder_type == 'cnn':
                he_imgs = he_imgs / 255.

        if self.rescale_to_64:
            x = TF.resize(x, (64, 64))
            if self.he_encoder_type == 'cnn':
                he_imgs = TF.resize(he_imgs, (64, 64))

        return x, he_imgs, labels, this_batch_size

    def _log_metrics(self, batch_size, loss, preds, labels, train=True, he_or_codex='codex'):
        """Log loss and classification metrics (accuracy, precision, recall, f1 score)"""
        prefix = f'{he_or_codex}_train' if train else f'{he_or_codex}_val'
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        class_weights = torch.tensor([torch.sum(labels == i) for i in range(self.num_classes)], device=self.device) / batch_size

        acc = self.accuracy(preds, labels) # accuracy for each class
        self.log(f'{prefix}_acc', torch.mean(acc), on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        self.log(f'{prefix}_acc_weighted', torch.sum(acc * class_weights) / torch.sum(class_weights), on_step=True, on_epoch=True, logger=True, batch_size=batch_size)

        f1score = self.f1(preds, labels) # f1 score for each class
        self.log(f'{prefix}_f1', torch.mean(f1score), on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        self.log(f'{prefix}_f1_weighted', torch.sum(f1score * class_weights) / torch.sum(class_weights), on_step=True, on_epoch=True, logger=True, batch_size=batch_size)

        for i in range(self.num_classes):
            class_name = self.class_name_dict[i].replace('/', '_').replace(' ', '_')
            prop = torch.sum(preds == i) / batch_size
            true_prop = torch.sum(labels == i) / batch_size
            class_acc, prec, recall = compute_binary_accuracy_metrics(preds, labels, i)
            self.log(f'{class_name}_{prefix}_prop', prop, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
            self.log(f'{class_name}_{prefix}_true_prop', true_prop, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
            self.log(f'{class_name}_{prefix}_acc', class_acc, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
            self.log(f'{class_name}_{prefix}_prec', prec, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
            self.log(f'{class_name}_{prefix}_recall', recall, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)

            self.log(f'{class_name}_{prefix}_torchmetric_acc', acc[i], on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
            self.log(f'{class_name}_{prefix}_f1', f1score[i], on_step=True, on_epoch=True, logger=True, batch_size=batch_size)

        return

    def training_step(self, batch, batch_idx):
        codex_imgs, he_imgs, labels, this_batch_size = self._get_data_for_step(batch, train=True)
        codex_imgs = self.simclr_codex_model(codex_imgs)
        he_imgs = self.simclr_he_model(he_imgs)

        codex_z, codex_proj = self.simclr_model(codex_imgs, encoder_type='cnn')
        he_z, he_proj = self.simclr_model(he_imgs, encoder_type=self.he_encoder_type)

        if self.dropout is not None:
            codex_z = self.dropout(codex_z)
            he_z = self.dropout(he_z)
        codex_logits = self.softmax(self.fc(codex_z))
        he_logits = self.softmax(self.fc(he_z))

        he_loss = self.loss(he_logits, labels)
        codex_loss = self.loss(codex_logits, labels)

        he_preds = torch.argmax(he_logits, dim=1)
        codex_preds = torch.argmax(codex_logits, dim=1)

        self._log_metrics(this_batch_size, he_loss, he_preds, labels, train=True, he_or_codex='he')
        self._log_metrics(this_batch_size, codex_loss, codex_preds, labels, train=True, he_or_codex='codex')

        if self.contrastive_loss is not None:
            contrastive_loss = self.contrastive_loss(codex_proj, he_proj)
            self.log('train_contrastive_loss', contrastive_loss, on_step=True, on_epoch=True, logger=True, batch_size=this_batch_size)
            loss = (1 - self.loss_term_weight) * (he_loss + codex_loss) + self.loss_term_weight * contrastive_loss
        elif self.same_prediction_loss is not None:
            same_prediction_loss_he = self.same_prediction_loss(F.one_hot(he_preds, num_classes=self.num_classes).float(), codex_preds)
            same_prediction_loss_codex = self.same_prediction_loss(F.one_hot(codex_preds, num_classes=self.num_classes).float(), he_preds)
            same_prediction_loss = (same_prediction_loss_he + same_prediction_loss_codex) / 2
            self.log('train_same_prediction_loss', same_prediction_loss, on_step=True, on_epoch=True, logger=True, batch_size=this_batch_size)
            loss = (1 - self.loss_term_weight) * (he_loss + codex_loss) + self.loss_term_weight * same_prediction_loss
        else:
            loss = he_loss + codex_loss

        return loss

    def validation_step(self, batch, batch_idx):
        codex_imgs, he_imgs, labels, this_batch_size = self._get_data_for_step(batch, train=False)
        codex_imgs = self.simclr_codex_model(codex_imgs)
        he_imgs = self.simclr_he_model(he_imgs)

        codex_z, codex_proj = self.simclr_model(codex_imgs, encoder_type='cnn')
        he_z, he_proj = self.simclr_model(he_imgs, encoder_type=self.he_encoder_type)

        codex_logits = self.softmax(self.fc(codex_z))
        he_logits = self.softmax(self.fc(he_z))

        he_loss = self.loss(he_logits, labels)
        codex_loss = self.loss(codex_logits, labels)

        he_preds = torch.argmax(he_logits, dim=1)
        codex_preds = torch.argmax(codex_logits, dim=1)

        self._log_metrics(this_batch_size, he_loss, he_preds, labels, train=False, he_or_codex='he')
        self._log_metrics(this_batch_size, codex_loss, codex_preds, labels, train=False, he_or_codex='codex')

        if self.contrastive_loss is not None:
            contrastive_loss = self.contrastive_loss(codex_proj, he_proj)
            self.log('val_contrastive_loss', contrastive_loss, on_step=True, on_epoch=True, logger=True, batch_size=this_batch_size)
            loss = (1 - self.loss_term_weight) * (he_loss + codex_loss) + self.loss_term_weight * contrastive_loss
        elif self.same_prediction_loss is not None:
            same_prediction_loss_he = self.same_prediction_loss(F.one_hot(he_preds, num_classes=self.num_classes).float(), codex_preds)
            same_prediction_loss_codex = self.same_prediction_loss(F.one_hot(codex_preds, num_classes=self.num_classes).float(), he_preds)
            same_prediction_loss = (same_prediction_loss_he + same_prediction_loss_codex) / 2
            self.log('val_same_prediction_loss', same_prediction_loss, on_step=True, on_epoch=True, logger=True, batch_size=this_batch_size)
            loss = (1 - self.loss_term_weight) * (he_loss + codex_loss) + self.loss_term_weight * same_prediction_loss
        else:
            loss = he_loss + codex_loss

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        
    def embed(self, x, he=False):
        """ Returns the embedding of x """
        if he:
            x = self.simclr_he_model(x)
            z, _ = self.simclr_model(x, encoder_type=self.he_encoder_type)
        else:
            x = self.simclr_codex_model(x)
            z, _ = self.simclr_model(x, encoder_type='cnn')
        return z

    def predict(self, x, labels, he=False):
        """ Returns the prediction of x """
        if he:
            x = self.simclr_he_model(x)
            z, _ = self.simclr_model(x, encoder_type=self.he_encoder_type)
        else:
            x = self.simclr_codex_model(x)
            z, _ = self.simclr_model(x, encoder_type='cnn')
        logits = self.softmax(self.fc(z))
        preds = torch.argmax(logits, dim=1)
        return preds

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Train SimCLR model on CODEX and H&E images')

    # Dataset arguments
    parser.add_argument('--train-acq-ids-path', type=str, help='List of acqusition ids for training', default='data/renal-transplant/renal-transplant_acq_ids_train.csv')
    parser.add_argument('--val-acq-ids-path', type=str, help='List of acqusition ids for validation', default='data/renal-transplant/renal-transplant_acq_ids_val.csv')
    parser.add_argument('--he-patches-dir', type=str, help='Directory of H&E patches', default='data/renal-transplant/he_patches_64')
    parser.add_argument('--codex-train-uri', type=str, help='path to TileDB CODEX training dataset')
    parser.add_argument('--codex-val-uri', type=str, help='path to TileDB CODEX validation dataset')
    parser.add_argument('--codex-all-uri', type=str, help='path to TileDB CODEX all dataset')
    parser.add_argument('--celltype-csv', type=str, help='CSV listing celltype labels of each cell (with hierarchy)', default='data/renal-transplant/renal-transplant_celltypes_hierarchy.csv')
    parser.add_argument('--celltype-hierarchy', type=int, help='Hierarchy level to train on', default=0)
    parser.add_argument('--patch-size', type=int, help='Size of patches', default=64)
    parser.add_argument('--rescale-to-64', help='set if want to rescale images to 64x64', action='store_true')

    # Foundation embedding arguments
    parser.add_argument('--use-foundation-embeds', help='set if want to use foundation embeddings. If set, must provide path to foundation embeddings', action='store_true')
    parser.add_argument('--foundation-embeds-path', type=str, help='Path to foundation embeddings')
    parser.add_argument('--foundation-embeds-ids-path', type=str, help='Path to foundation embedding ids')

    # Model architecture arguments
    parser.add_argument('--num-layers', type=int, help='Number of layers in CNN encoder', default=4)
    parser.add_argument('--encoder-filter-dim', type=int, help='Number of filters in first layer of CNN encoder', default=64)
    parser.add_argument('--embedding-size', type=int, help='Dimension of embedding', default=128)
    parser.add_argument('--projection-dim', type=int, help='Dimension of projection head', default=128)
    parser.add_argument('--batch-norm', help='set if want to use batch norm in CNN encoder', action='store_true')
    parser.add_argument('--track-running-stats', help='set if want to track running stats in batch norm', action='store_true')
    parser.add_argument('--dropout', type=float, help='Dropout rate for classifier and fc encoder', default=None)

    # Training arguments
    parser.add_argument('--batch-size', type=int, help='Batch size', default=16)
    parser.add_argument('--zero-one-normalize', help='set if want to normalize images to [0, 1]', action='store_true')
    parser.add_argument('--diff-transform', help='set if want to use different transformations for H&E and CODEX pairs', action='store_true')
    parser.add_argument('--randaugment', help='set if using randaugment on H&E', action='store_true')
    parser.add_argument('--randaugment-params', nargs='+', help='Parameters for randaugment. 2 values: (N, M)', default=[3, 5])
    parser.add_argument('--num-workers', type=int, help='Number of workers for dataloader', default=0)
    parser.add_argument('--shuffle-buffer-size', type=int, help='Buffer size for shuffling', default=1000)
    parser.add_argument('--num-epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--temperature', type=float, help='Temperature for contrastive loss', default=0.5)
    parser.add_argument('--no-logger', help='set if not logging any metrics', action='store_true')
    parser.add_argument('--max-hours', type=int, help='Maximum number of hours to train (must be an integer)')
    parser.add_argument('--which-gpu', type=int, help='Which GPU to use for training', default=0)
    parser.add_argument('--checkpoint', help='path to training checkpoint. only set if you want to resume training', type=str)

    # finetuning
    parser.add_argument('--finetune', help='set if want to finetune on downstream task', action='store_true')
    parser.add_argument('--freeze-encoder', help='set if want to freeze encoder weights', action='store_true')
    parser.add_argument('--finetune-checkpoint', help='path to finetuning checkpoint. only set if you want to resume finetuning', type=str)
    parser.add_argument('--include-contrastive-loss', help='set if want to include contrastive loss in finetuning', action='store_true')
    parser.add_argument('--enforce-same-prediction', help='set if want to enforce same prediction for CODEX and H&E images', action='store_true')
    parser.add_argument('--loss-term-weight', type=float, help='Multiplier for contrastive loss or enforce-same-prediction weight. Must be a value between 0 and 1.', default=0.5)
    parser.add_argument('--weight-loss', help='set if want to weight loss by class frequency', action='store_true')

    parser.add_argument('--verbose', help='set if want to print out dataset creation info', action='store_true')
    parser.add_argument('--save-checkpoint-path', help='path to save training checkpoint. Optional, the script will save last training checkpoint anyway.', type=str)
    parser.add_argument('--save-checkpoint-every-n-epochs', help='set if you want to save a checkpoint every n epochs', type=int)

    args = parser.parse_args()

    if args.use_foundation_embeds and args.foundation_embeds_path is None:
        raise ValueError('Must provide path to foundation embeddings if using foundation embeddings')
    if args.use_foundation_embeds and args.foundation_embeds_ids_path is None:
        raise ValueError('Must provide path to foundation embedding ids if using foundation embeddings')

    if args.codex_all_uri is not None and (args.codex_train_uri is not None or args.codex_val_uri is not None):
        raise ValueError('Cannot have codex-all-uri and codex-train-uri/val-uri set at the same time')

    if args.include_contrastive_loss and args.enforce_same_prediction:
        raise ValueError('Cannot have both include-contrastive-loss and enforce-same-prediction set to True')

    if args.loss_term_weight < 0 or args.loss_term_weight > 1:
        raise ValueError('loss-term-weight must be between 0 and 1')

    if not args.no_logger:
        wandb_logger = WandbLogger(entity="mdayao", project="joint-he-codex")

    train_acq_ids = list(pd.read_csv(args.train_acq_ids_path, header=None).iloc[:,0].values)
    val_acq_ids = list(pd.read_csv(args.val_acq_ids_path, header=None).iloc[:,0].values)
    
    # read in biomarkers
    cur = emdb.connect()
    biomarkers = emdb.get_all_biomarkers_for_acquisition_id(train_acq_ids[0], cur)
    biomarkers.remove('DAPI')
    biomarkers = ['DAPI'] + biomarkers

    # Create H&E datasets
    if args.use_foundation_embeds:
        if args.verbose:
            print(f'Using foundation embeddings for H&E from {args.foundation_embeds_path}')
        train_he_dataset = HEFoundationDataset(
                embeds_path=args.foundation_embeds_path,
                ids_path=args.foundation_embeds_ids_path,
                acq_ids=train_acq_ids,
                label_csv=args.celltype_csv,
                verbose=args.verbose,
                )
        val_he_dataset = HEFoundationDataset(
                embeds_path=args.foundation_embeds_path,
                ids_path=args.foundation_embeds_ids_path,
                acq_ids=val_acq_ids,
                label_csv=args.celltype_csv,
                verbose=args.verbose,
                )
        foundation_embedding_size = train_he_dataset[0][0].shape[0]
    else:
        train_he_dataset = HEDataset(
                data_dir=args.he_patches_dir,
                acq_ids=train_acq_ids,
                label_csv=args.celltype_csv,
                hierarchy=args.celltype_hierarchy,
                transform=None, # SimCLRModule already has transform
                randaugment=args.randaugment,
                randaugment_params=args.randaugment_params,
                verbose=args.verbose,
                )
        val_he_dataset = HEDataset(
                data_dir=args.he_patches_dir,
                acq_ids=val_acq_ids,
                label_csv=args.celltype_csv,
                hierarchy=args.celltype_hierarchy,
                transform=None,
                randaugment=False,
                verbose=args.verbose,
                )

    # filter out CODEX cells that don't have H&E images
    # loop through cell keys, check if they are in train/val_he_dataset.acq_id_cell_id_to_idx
    if args.codex_all_uri is not None:
        if args.verbose:
            print('loading all patches')
        all_patches = EMCellPatchDataset(
                path = args.codex_all_uri,
                patch_size=args.patch_size, 
                biomarker_subset=biomarkers,
                nucleus_mask=False
                )
        if args.verbose:
            print('before filtering all', len(all_patches.cell_key_datapipe))
        train_subset_inds = []
        val_subset_inds = []
        for i, cell_key in enumerate(all_patches.cell_key_datapipe):
            if cell_key in train_he_dataset.acq_id_cell_id_to_idx:
                train_subset_inds.append(i)
            if cell_key in val_he_dataset.acq_id_cell_id_to_idx:
                val_subset_inds.append(i)
        # Recreate dataset with filtered cells only
        train_patches = EMCellPatchDataset(
                path = args.codex_all_uri,
                patch_size=args.patch_size, 
                biomarker_subset=biomarkers,
                nucleus_mask=False,
                subset_inds=train_subset_inds,
                )
        val_patches = EMCellPatchDataset(
                path = args.codex_all_uri,
                patch_size=args.patch_size,
                biomarker_subset=biomarkers,
                nucleus_mask=False,
                subset_inds=val_subset_inds,
                )

    else:
        train_patches = EMCellPatchDataset(
                path = args.codex_train_uri,
                patch_size=args.patch_size, 
                biomarker_subset=biomarkers,
                nucleus_mask=False
                )
        val_patches = EMCellPatchDataset(
                path = args.codex_val_uri,
                patch_size=args.patch_size,
                biomarker_subset=biomarkers,
                nucleus_mask=False
                )
        key_datapipes = {'train': train_patches.cell_key_datapipe, 'val': val_patches.cell_key_datapipe} # datapipe that provides patch identifiers
        if args.verbose:
            print('before filtering train', len(key_datapipes['train']))
        train_subset_inds = []
        for i, cell_key in enumerate(key_datapipes['train']):
            if cell_key not in train_he_dataset.acq_id_cell_id_to_idx:
                continue
            else:
                train_subset_inds.append(i)
        if args.verbose:
            print('before filtering val', len(key_datapipes['val']))
        val_subset_inds = []
        for i, cell_key in enumerate(key_datapipes['val']):
            if cell_key not in val_he_dataset.acq_id_cell_id_to_idx:
                continue
            else:
                val_subset_inds.append(i)

        # Recreate dataset with filtered cells only
        train_patches = EMCellPatchDataset(
                path = args.codex_train_uri,
                patch_size=args.patch_size, 
                biomarker_subset=biomarkers,
                nucleus_mask=False,
                subset_inds=train_subset_inds,
                )
        val_patches = EMCellPatchDataset(
                path = args.codex_val_uri,
                patch_size=args.patch_size,
                biomarker_subset=biomarkers,
                nucleus_mask=False,
                subset_inds=val_subset_inds,
                )
    key_datapipes = {'train': train_patches.cell_key_datapipe, 'val': val_patches.cell_key_datapipe} # datapipe that provides patch identifiers
    if args.verbose:
        print('after filtering train', len(key_datapipes['train']))
        print('after filtering val', len(key_datapipes['val']))
        if args.codex_all_uri is not None:
            print('after filtering all', len(key_datapipes['train']) + len(key_datapipes['val']))
    image_datapipes = {'train': train_patches.image_datapipe, 'val': val_patches.image_datapipe} # datapipe that provides patches

    # create datapipes that provides H&E images in the same order as the codex images
    train_dataset_order = []
    for i, cell_key in enumerate(key_datapipes['train']):
        train_dataset_order.append(train_he_dataset.acq_id_cell_id_to_idx[cell_key])
    val_dataset_order = []
    for i, cell_key in enumerate(key_datapipes['val']):
        val_dataset_order.append(val_he_dataset.acq_id_cell_id_to_idx[cell_key])
    he_datapipes = {
            'train': HEDataPipe(train_he_dataset, dataset_order=train_dataset_order),
            'val': HEDataPipe(val_he_dataset, dataset_order=val_dataset_order)
            } if not args.use_foundation_embeds else {
                    'train': HEFoundationDataPipe(train_he_dataset, dataset_order=train_dataset_order),
                    'val': HEFoundationDataPipe(val_he_dataset, dataset_order=val_dataset_order)
                    }

    assert len(key_datapipes['train']) == len(he_datapipes['train']), f'Length of training key datapipe ({len(key_datapipes["train"])}) != length of H&E datapipe ({len(he_datapipes["train"])})'
    assert len(key_datapipes['val']) == len(he_datapipes['val']), f'Length of validation key datapipe ({len(key_datapipes["val"])}) != length of H&E datapipe ({len(he_datapipes["val"])})'

    # combine codex image datapipe, he image datapipe, and key datapipes
    combined_datapipes = {
            'train': image_datapipes['train'].zip(he_datapipes['train'], key_datapipes['train']), 
            'val': image_datapipes['val'].zip(he_datapipes['val'], key_datapipes['val'])
            }
    
    # shuffle the combined datapipe
    combined_datapipes['train'] = combined_datapipes['train'].shuffle(buffer_size=args.shuffle_buffer_size)

    # use sharding filter to enable multiple dataloader workers
    combined_datapipes['train'] = combined_datapipes['train'].sharding_filter()
    combined_datapipes['val'] = combined_datapipes['val'].sharding_filter()

    dataloaders = {
            'train': DataLoader(dataset=combined_datapipes['train'], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, persistent_workers=args.num_workers > 0), 
            'val': DataLoader(dataset=combined_datapipes['val'], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, persistent_workers=args.num_workers > 0) 
            }
    
    if args.finetune: # supervised finetuning phase
        simclr_model = SimCLRModule.load_from_checkpoint(args.checkpoint, map_location=f'cuda:{args.which_gpu}')
        if args.finetune_checkpoint:
            model = SupervisedModule.load_from_checkpoint(
                    args.finetune_checkpoint,
                    simclr_module=simclr_model,
                    map_location=f'cuda:{args.which_gpu}',
                    )
        else:
            model = SupervisedModule(
                    simclr_module=simclr_model,
                    celltype_csv=args.celltype_csv,
                    train_acq_ids=train_acq_ids,
                    hierarchy=args.celltype_hierarchy,
                    lr=args.lr,
                    freeze_encoder=args.freeze_encoder,
                    include_contrastive_loss=args.include_contrastive_loss,
                    temperature=args.temperature,
                    enforce_same_prediction=args.enforce_same_prediction,
                    loss_term_weight=args.loss_term_weight,
                    weight_loss=args.weight_loss,
                    dropout=args.dropout,
                    )
    else: # contrastive learning phase
        model = SimCLRModule(
                he_encoder_type='cnn' if not args.use_foundation_embeds else 'fc',
                he_channels=3 if not args.use_foundation_embeds else foundation_embedding_size,
                codex_channels=len(biomarkers),
                input_patch_size=args.patch_size if not args.rescale_to_64 else 64,
                num_layers=args.num_layers, 
                embedding_size=args.embedding_size,
                encoder_filter_dim=args.encoder_filter_dim,
                batch_norm=args.batch_norm,
                track_running_stats=args.track_running_stats,
                projection_dim=args.projection_dim, 
                temperature=args.temperature, 
                lr=args.lr,
                zero_one_normalize=args.zero_one_normalize,
                diff_transform=args.diff_transform,
                rescale_to_64=args.rescale_to_64,
                )

    if not args.no_logger:
        if args.save_checkpoint_every_n_epochs:
            checkpoint_callback = ModelCheckpoint(every_n_epochs=args.save_checkpoint_every_n_epochs)
        wandb_logger.watch(model, log='all')
        trainer = pl.Trainer(
                accelerator='gpu', 
                devices=[args.which_gpu], 
                max_epochs=args.num_epochs, 
                max_time=f"00:{args.max_hours:02d}:00:00" if args.max_hours is not None else None, 
                logger=wandb_logger, 
                callbacks=[checkpoint_callback] if args.save_checkpoint_every_n_epochs else None,
                )
    else:
        trainer = pl.Trainer(
                accelerator='gpu', 
                devices=[args.which_gpu], 
                max_epochs=args.num_epochs, 
                max_time=f"00:{args.max_hours:02d}:00:00" if args.max_hours is not None else None, 
                )

    trainer.fit(model=model, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['val'], ckpt_path=args.finetune_checkpoint if args.finetune else args.checkpoint)

    if args.save_checkpoint_path:
        trainer.save_checkpoint(args.save_checkpoint_path)

    if not args.no_logger:
        wandb.finish()

