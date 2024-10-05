import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import torchmetrics
import sys

import numpy as np
import random
import pandas as pd
from scipy import ndimage

def get_class_weights(num_classes, samples_per_cls, power = 1):
    """Computes the weights for each class based on the number of samples per class.
    Classes with fewer samples are weighted more highly

    Args:
        num_classes (int): number of classes
        samples_per_cls (torch.tensor): 1D tensor with number of samples per class
        power (int): exponent value
    
    Returns:
        weights (torch.tensor): 1D tensor with weights for each class 
    """
    weights = 1.0 / torch.pow(samples_per_cls, power)
    weights = weights / torch.sum(weights) * num_classes

    return weights

def compute_binary_accuracy_metrics(preds, labels, class_of_interest):
    """Computes accuracy, precision and recall for a specific class, 
    treating it like a binary classification problem.
    
    Args:
        preds (torch.tensor): 1D tensor with model predictions
        labels (torch.tensor): 1D tensor with true labels
        class_of_interest (int): class label of interest
    
    Returns:
        accuracy, precision, recall (all float)
    """
    i = class_of_interest
    total = len(preds)
    tp = torch.sum(torch.logical_and(preds == i, labels == i))
    tn = torch.sum(torch.logical_and(preds != i, labels != i))
    fp = torch.sum(torch.logical_and(preds == i, labels != i))
    fn = torch.sum(torch.logical_and(preds != i, labels == i))

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall

class CustomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class LitClassifierModel(pl.LightningModule):
    def __init__(self,
                input_dim: int,
                patch_dim: int,
                latent_dim: int,
                num_channels: int,
                batch_size : int,
                train_acq_ids: list, 
                lr: float,
                optimizer: str,
                num_conv_layers: int = 3,
                dropout: float = 0.2,
                zscale: bool = True,
                dataset: str = 'citn-codex',
                celltype_csv: str = './data/CITN_clusters_labeled_coarse.csv',
                coarse_labels : bool = False,
                binary_problem : bool = False,
                he_input: bool = False,
                hierarchy: int = None,
                include_seg_mask: bool = False,
                cell_mask_only: bool = False,
                foundation_embed_input: bool = False,
                hidden_dim: int = 512,
                batch_norm: bool = False,
                weight_decay: float = None,
                ):
        super().__init__()

        if binary_problem and not coarse_labels:
            raise ValueError('binary_problem can only be True if coarse_labels is True. binary_problem = {binary_problem}, coarse_labels = {coarse_labels}.')

        # log hyperparameters
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.learning_rate = lr
        self.optimizer = optimizer
        self.dropout = dropout
        self.binary_problem = binary_problem
        self.zscale = zscale
        self.dataset = dataset
        self.num_conv_layers = num_conv_layers
        self.he_input = he_input
        self.hierarchy = hierarchy
        self.latent_dim = latent_dim
        self.include_seg_mask = include_seg_mask
        self.cell_mask_only = cell_mask_only
        self.foundation_embed_input = foundation_embed_input
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm
        self.weight_decay = weight_decay

        celltype_df = pd.read_csv(celltype_csv)
        celltype_df['acq_id_cell_id'] = celltype_df.acquisition_id + "_" + celltype_df.cell_id.astype(str)
        if self.dataset in ['hnscc-atlas', 'renal-transplant', 'charville', 'transplant-stanford']:
            if self.hierarchy is None:
                celltype_label_col = 'celltype_label'
            else:
                celltype_label_col = f'celltype_hierarchy_{self.hierarchy}'

            # Assigning unique integer ids to each celltype label
            celltype_df['celltype_id'] = pd.factorize(celltype_df[celltype_label_col])[0]

            self.num_classes = len(np.unique(celltype_df.celltype_id))
            class_counts = torch.from_numpy(celltype_df[celltype_df.acquisition_id.isin(train_acq_ids)].value_counts('celltype_id', sort=False).values)
            self.class_name_dict = pd.Series(celltype_df[celltype_label_col].values, index=celltype_df.celltype_id).to_dict()

            # Adding label_int column to match convention for other functions here
            celltype_df['label_int'] = celltype_df['celltype_id'].astype(int)
            self.celltype_df = celltype_df[['acq_id_cell_id', 'label_int']]
            self.celltype_df = self.celltype_df.set_index('acq_id_cell_id')
            self.label_dict = self.celltype_df.to_dict('index')
            self.label_dict = {x[0]: x[1]['label_int'] for x in self.label_dict.items()}
        else:
            if coarse_labels:
                celltype_df['label_int'] = celltype_df['coarse_label_id'].astype(int)
                if binary_problem:
                    celltype_df.loc[celltype_df.label_int != 5, 'coarse_label'] = 'Not epithelium'
                    celltype_df.loc[celltype_df.label_int != 5, 'label_int'] = 0
                    celltype_df.loc[celltype_df.label_int == 5, 'label_int'] = 1
                self.class_name_dict = pd.Series(celltype_df.coarse_label.values, index=celltype_df.label_int).to_dict()
            else:
                celltype_df['label_int'] = celltype_df['leiden.res1.0'].str[2:].astype(int)-1
                self.class_name_dict = pd.Series(celltype_df.label.values, index=celltype_df.coarse_label_id).to_dict()
            self.num_classes = len(np.unique(celltype_df.label_int))
            self.celltype_df = celltype_df[['acq_id_cell_id', 'label_int']]
            self.celltype_df = self.celltype_df.set_index('acq_id_cell_id')
            self.label_dict = self.celltype_df.to_dict('index')
            self.label_dict = {x[0]: x[1]['label_int'] for x in self.label_dict.items()}

            class_counts = torch.from_numpy(celltype_df[celltype_df.acquisition_id.isin(train_acq_ids)].value_counts('label_int', sort=False).values)

        self.class_weights = get_class_weights(self.num_classes, class_counts)

        if not self.foundation_embed_input:
            self.conv_layers = nn.ModuleList()
            self.hdims = [2**(x+6) for x in range(self.num_conv_layers)] # Conv layer output dims (64, 128, 256, 512...)
            for i in range(self.num_conv_layers):
                conv_layer = nn.Sequential(
                    nn.Conv2d(num_channels if i==0 else self.hdims[i-1], self.hdims[i], kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                self.conv_layers.append(conv_layer)

            if self.dropout is not None:
                self.dropout_conv_layers = nn.ModuleList()
                for i in range(self.num_conv_layers):
                    self.dropout_conv_layers.append(nn.Dropout(self.dropout))
                self.fc_dropout = nn.Dropout(0.5)

            n_sizes = self._get_conv_output((num_channels,patch_dim, patch_dim))

            self.fc1 = nn.Sequential(nn.Linear(n_sizes, self.latent_dim), nn.LeakyReLU())

            self.random_transformer = transforms.Compose([
                                    CustomRotationTransform(angles=[-90, 0, 90, 180]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()
                                ])
        else:
            self.fc_layers = nn.ModuleList()
            in_features = input_dim
            if hidden_dim is None or hidden_dim == 0:
                self.fc_layers.append(nn.Linear(in_features, self.latent_dim))
                self.fc_layers.append(nn.LeakyReLU())
                if batch_norm:
                    self.fc_layers.append(nn.BatchNorm1d(self.latent_dim))
                if self.dropout is not None:
                    self.fc_layers.append(nn.Dropout(dropout))
            else:
                for _ in range(2):
                    self.fc_layers.append(nn.Linear(in_features, hidden_dim))
                    self.fc_layers.append(nn.LeakyReLU())
                    if batch_norm:
                        self.fc_layers.append(nn.BatchNorm1d(hidden_dim))
                    if self.dropout is not None:
                        self.fc_layers.append(nn.Dropout(dropout))
                    in_features = hidden_dim
                    hidden_dim = self.latent_dim
            if self.dropout is not None:
                self.fc_dropout = nn.Dropout(dropout)

        if self.binary_problem:
            self.fc2 = nn.Linear(self.latent_dim, 1)
            self.last_layer = nn.Identity()
            self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights[1]/self.class_weights[0])
        else:
            self.fc2 = nn.Linear(self.latent_dim, self.num_classes)
            self.last_layer = nn.LogSoftmax(dim=1)
            self.loss = nn.NLLLoss(weight=self.class_weights)

        self.accuracy = torchmetrics.Accuracy(task='binary' if self.binary_problem else 'multiclass', num_classes=self.num_classes, top_k=1, average=None)
        self.f1 = torchmetrics.F1Score(task='binary' if self.binary_problem else 'multiclass', num_classes=self.num_classes, top_k=1, average=None)
        
        self.apply(self._init_weights) # initialize weights

    def _init_weights(self, m, init_method=nn.init.kaiming_uniform_):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init_method(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        return

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input_var = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input_var) 
        n_size = output_feat.data.view(batch_size, -1).size(1)

        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):

        if not self.foundation_embed_input:
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                if self.dropout is not None:
                    x = self.dropout_conv_layers[i](x)
        else:
            for layer in self.fc_layers:
                x = layer(x)
        return x

    # will be used during inference
    def forward(self, x):

        x = self._forward_features(x)
        if not self.foundation_embed_input:
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
        if self.dropout is not None:
            x = self.fc_dropout(x)
        x = self.fc2(x)
        x = self.last_layer(x)
        return x
    
    def embed(self, x):
        """Returns the embedding vector for the input x, layer before final classification layer
        """
        x = self._forward_features(x)
        if not self.foundation_embed_input:
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
        return x
    
    def predict_from_embedding(self, z):
        """Returns the prediction for given embedding vector z
        """
        y = self.fc2(z)
        outputs = self.last_layer(y)

        if self.binary_problem:
            preds = (outputs >= 0.5).float()
            preds.squeeze_()
        else:
            preds = torch.argmax(outputs, dim=1)

        return preds
   
    def _get_data_for_step(self, batch, train=True):
        if self.he_input:
            x = batch[0] # batch is tuple (image, celltype_id)
            labels = batch[1]
            this_batch_size = x.shape[0]

            if not self.foundation_embed_input:
                # Z normalize each R,G,B channel
                if self.zscale:
                    non_seg_x = x[:,1:,:,:] if self.include_seg_mask and not self.cell_mask_only else x
                    assert non_seg_x.shape == (this_batch_size,3,self.patch_dim,self.patch_dim)
                    im_mean = torch.mean(non_seg_x, dim=(2,3), keepdim=True)
                    assert im_mean.shape == (this_batch_size,3,1,1)
                    im_std = torch.std(non_seg_x, dim=(2,3), keepdim=True)
                    im_std[im_std == 0] = 1
                    non_seg_x = (non_seg_x - im_mean) / im_std
                    if self.include_seg_mask and not self.cell_mask_only:
                        x[:,1:,:,:] = non_seg_x
                    else:
                        x = non_seg_x
            
            # HE data is already flipped/rotated so no need to do it here

        else:
            x = batch[0] # batch is tuple (image, cell_id)
            x = x.float()
            this_batch_size = x.shape[0]
            cell_ids = batch[1] # cell id is like CITN10Co-88_c001_v001_r001_reg073_3179
            labels = np.array([self.label_dict[cell_id] for cell_id in cell_ids])
            labels = torch.from_numpy(labels).to(self.device)

            # Z normalize each image
            if self.zscale:
                im_mean = torch.mean(x, dim=(1,2,3), keepdim=True)
                im_std = torch.std(x, dim=(1,2,3), keepdim=True)
                im_std[im_std == 0] = 1
                x = (x - im_mean) / im_std

            # Randomly rotate + flip
            if train:
                x = self.random_transformer(x)

        return x, labels, this_batch_size

    def _log_metrics(self, preds, labels, loss, this_batch_size, fold='train'):

        self.log(f'{fold}_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)
        class_weights = torch.tensor([torch.sum(labels == i) for i in range(self.num_classes)], device=self.device) / this_batch_size

        acc = self.accuracy(preds, labels) # accuracy for each class
        self.log(f'{fold}_acc', torch.mean(acc), on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)

        f1score = self.f1(preds, labels) # f1 score for each class
        self.log(f'{fold}_f1', torch.mean(f1score), on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)
        self.log(f'{fold}_f1_weighted', torch.sum(f1score * class_weights) / torch.sum(class_weights), on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)

        for i in range(self.num_classes):
            prop = torch.sum(preds == i) / this_batch_size
            class_acc, prec, recall = compute_binary_accuracy_metrics(preds, labels, i)
            self.log(f'{self.class_name_dict[i]}_{fold}_prop', prop, on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)
            self.log(f'{self.class_name_dict[i]}_{fold}_acc', class_acc, on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)
            self.log(f'{self.class_name_dict[i]}_{fold}_prec', prec, on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)
            self.log(f'{self.class_name_dict[i]}_{fold}_recall', recall, on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)

            self.log(f'{self.class_name_dict[i]}_{fold}_torchmetric_acc', acc[i], on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)
            self.log(f'{self.class_name_dict[i]}_{fold}_f1', f1score[i], on_step=False, on_epoch=True, logger=True, batch_size=this_batch_size)

        return
    
    def training_step(self, batch, batch_idx):
        x, labels, this_batch_size = self._get_data_for_step(batch, train=True)

        outputs = self(x) # forward pass

        if self.binary_problem:
            labels.unsqueeze_(1)
            loss = self.loss(outputs, labels.float())
            preds = (outputs >= 0.5).float()
            preds.squeeze_()
        else:
            loss = self.loss(outputs, labels.clone())
            preds = torch.argmax(outputs, dim=1)

        # training metrics
        labels.squeeze_()
        self._log_metrics(preds, labels, loss.detach().clone(), this_batch_size, fold='train')

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels, this_batch_size = self._get_data_for_step(batch, train=False)

        outputs = self(x) # forward pass

        if self.binary_problem:
            labels.unsqueeze_(1)
            loss = self.loss(outputs, labels.float())
            preds = (outputs >= 0.5).float()
            preds.squeeze_()
        else:
            loss = self.loss(outputs, labels.clone())
            preds = torch.argmax(outputs, dim=1)

        # validation metrics
        labels.squeeze_()
        self._log_metrics(preds, labels, loss.detach().clone(), this_batch_size, fold='val')

        return loss
    
    def predict_step(self, batch, batch_idx):
        x, labels, this_batch_size = self._get_data_for_step(batch, train=False)

        outputs = self(x) # forward pass

        if self.binary_problem:
            preds = (outputs >= 0.5).float()
            preds.squeeze_()
        else:
            preds = torch.argmax(outputs, dim=1)

        return preds


    def configure_optimizers(self):
        if self.optimizer == 'adam':
            if self.weight_decay is not None:
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        return optimizer

