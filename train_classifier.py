import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import tiledb
import json

import os
os.environ["STAGE"]="prod"
import sys
#sys.path.append("/home/ubuntu/enable_rnd/rnd-utils/pythonutils/")

import emdatabase as emdb
from classifier_model import LitClassifierModel
from datamodules import HEDataModule, HEFoundationDataModule
from emimagepatch import EMCellPatchDataset

import wandb
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', help='which dataset to use', type=str, choices=['hnscc-atlas', 'upmc-hnscc', 'citn-codex', 'renal-transplant', 'charville', 'transplant-stanford'])
parser.add_argument('--he', help='set if train classifier on H&E data', action="store_true")
parser.add_argument('--he-patches-dir', type=str, help='Directory of H&E patches')
parser.add_argument('--he-foundation', help='set if using H&E foundation data', action='store_true')
parser.add_argument('--he-foundation-path', type=str, help='Path to H&E foundation dataset')
parser.add_argument('--he-foundation-ids-path', type=str, help='Path to H&E foundation embedding ids')
parser.add_argument('--randaugment', help='set if using randaugment on H&E', action='store_true')
parser.add_argument('--randaugment-params', nargs='+', help='Parameters for randaugment. 2 values: (N, M)', default=[3, 5])
parser.add_argument('--use-seg-mask', help='set if using segmentation mask as an additional channel (H&E only)', action='store_true')
parser.add_argument('--cell-mask-only', help='set if using segmentation mask to mask out cell only in image patch (H&E only)', action='store_true')
parser.add_argument('--seg-mask-dir', help='directory of segmentation masks', type=str, default='data/hnscc-atlas/seg_mask_nucl_64')
parser.add_argument('--channels', help='list of channels to use', type=str, default='./data/biomarkers_ignoreCD163.csv')
parser.add_argument('--coarse-labels', help='set if using coarser cell type labels', action='store_true')
parser.add_argument('--binary-problem', help='set if training binary classification problem. This will change the multiclass celltype classification problem to a binary problem classifying between skin epithelium and non-epithelium cells.', action='store_true')
parser.add_argument('--hierarchy', help='value of hierarchy if using hierarchical labels (currently only applies to HNSCC Atlas dataset). If left blank, defaults to most granular cell type labels.', type=int)

parser.add_argument('--epochs', help='number of epochs to train', type=int, default=5)

parser.add_argument('--patch-size', help='patch size for image inputs', type=int, default=64)
parser.add_argument('--batch-size', help='batch size used for training', type=int, default=64)
parser.add_argument('--latent-dim', help='latent dimension for model embedding', type=int, default=128)
parser.add_argument('--learning-rate', help='learning rate for optimizer', type=float, default=1e-3)
parser.add_argument('--optimizer', help='which optimizer to use', choices=['sgd', 'adam'], default='adam')
parser.add_argument('--zscale', help='set if zscaling patches', action='store_true')
parser.add_argument('--dropout', help='dropout param for conv layers', type=float, default=0.2)
parser.add_argument('--num-conv-layers', help='number of convolutional layers', type=int, default=3)
parser.add_argument('--hidden-dim', help='hidden dimension for FC model for foundation embeddings', type=int, default=512)
parser.add_argument('--batch-norm', help='set if using batch normalization for FC model', action='store_true')
parser.add_argument('--weight-decay', help='weight decay for optimizer', type=float)

parser.add_argument('--no-logger', help='set if not logging any metrics', action='store_true')

parser.add_argument('--celltype-csv', help='path to csv file with celltype labels', type=str, default='data/CITN_clusters_labeled_coarse.csv')
parser.add_argument('--train-uri', help='path to tiledb dataset for training', type=str, default='s3://research-and-development-us-east-2/ml_patch_datasets/monica_morph/CITN_subsets/CITN_train')
parser.add_argument('--val-uri', help='path to tiledb dataset for validation', type=str, default='s3://research-and-development-us-east-2/ml_patch_datasets/monica_morph/CITN_subsets/CITN_val')

parser.add_argument('--train-regions', help='path to list of training regions', type=str, default='data/CITN_test_acq_ids_train.csv')
parser.add_argument('--val-regions', type=str, help='List of acqusition ids for validation') # no default in case of no validation set

parser.add_argument('--checkpoint', help='path to training checkpoint. only set if you want to resume training', type=str)
parser.add_argument('--which-gpu', type=int, help='Which GPU to use for training', default=0)
parser.add_argument('--save-checkpoint-every-n-epochs', help='set if you want to save a checkpoint every n epochs', type=int)

parser.add_argument('--pretrain', help='set if pretraining dataset on charville.', action='store_true')

args = parser.parse_args()

epochs = args.epochs
use_coarse_labels = args.coarse_labels
binary_problem = args.binary_problem
zscale = args.zscale

patch_size = args.patch_size
batch_size = args.batch_size
latent_dim = args.latent_dim

celltype_csv = args.celltype_csv
train_regions = args.train_regions

if args.he_foundation and not args.he:
    raise ValueError("H&E foundation embeddings requires the H&E flag to be set")

if args.dataset == 'hnscc-atlas':
    sample_acq_id = 'HNSCC_c001_v001_r001_reg001'
    train_regions = 'data/hnscc-atlas/HNSCC_Atlas_acq_ids_train.csv'
elif args.dataset == 'upmc-hnscc':
    sample_acq_id = 'UPMC_c001_v001_r001_reg001'
elif args.dataset == 'citn-codex':
    sample_acq_id = "CITN10Co-88_c001_v001_r001_reg016"
elif args.dataset == 'renal-transplant':
    sample_acq_id = 's314_c001_v001_r001_reg002'
elif args.dataset == 'charville':
    sample_acq_id = 'Charvill-94_c005_v001_r001_reg001'
elif args.dataset == 'transplant-stanford':
    sample_acq_id = 's515-split1_c002_v001_r001_reg001'


if not args.no_logger:
#    wandb_logger = WandbLogger(project="he-codex-morph")
    #wandb_logger = WandbLogger(entity="mdayao", project="self-supervised-morphology")
    wandb_logger = WandbLogger(entity="mdayao", project="joint-he-codex")

# read in biomarkers
cur = emdb.connect()
biomarkers = emdb.get_all_biomarkers_for_acquisition_id(sample_acq_id, cur)
biomarkers.remove('DAPI')
if args.dataset == 'hnscc-atlas':
    biomarkers.remove('DRAQ5') # a reference marker that isn't needed for HNSCC Atlas
biomarkers = ['DAPI'] + biomarkers

# Set up train and validation patch datasets
if args.he:
    if args.dataset not in ['hnscc-atlas', 'renal-transplant', 'charville', 'transplant-stanford']:
        raise ValueError(f"H&E option can only be used with the hnscc-atlas, renal-transplant or charville dataset options. Current dataset used is {args.dataset}")
    if args.pretrain and args.dataset != 'renal-transplant':
        raise ValueError(f"Pretraining option can only be used with the renal-transplant dataset option. Current dataset used is {args.dataset}")
    # set up datamodule for H&E data
    if not args.he_foundation:
        he_dm = HEDataModule(
                data_dir = args.he_patches_dir,
                label_csv = celltype_csv,
                hierarchy = args.hierarchy,
                randaugment = args.randaugment,
                randaugment_params = args.randaugment_params,
                include_seg_mask=args.use_seg_mask,
                cell_mask_only=args.cell_mask_only,
                seg_mask_dir=args.seg_mask_dir,
                batch_size = batch_size,
                train_acq_ids_path = args.train_regions,
                val_acq_ids_path = args.val_regions,
                num_workers = 4,
                )
        he_dm.setup()
    else:
        he_dm = HEFoundationDataModule(
                embeds_path = args.he_foundation_path,
                ids_path = args.he_foundation_ids_path,
                train_acq_ids_path = args.train_regions,
                label_csv = celltype_csv,
                batch_size = batch_size,
                num_workers = 4,
                transform=None,
                verbose=True,
                val_acq_ids_path = args.val_regions,
                )
        he_dm.setup()
        foundation_input_size = he_dm.train_dataset[0][0].shape[0]
    print('train dataset size:', len(he_dm.train_dataset))
    if args.val_regions:
        print('val dataset size:', len(he_dm.val_dataset))
    else:
        print('no validation set')

    if args.pretrain:
        pretrain_dm = HEDataModule(
                data_dir = "data/charville/he_patches_64/",
                label_csv = "data/charville/charville_celltypes.csv",
                hierarchy = None,
                randaugment = args.randaugment,
                randaugment_params = args.randaugment_params,
                include_seg_mask=False,
                cell_mask_only=False,
                seg_mask_dir=None,
                batch_size = batch_size,
                train_acq_ids_path = "data/charville/charville_acq_ids_train.csv",
                val_acq_ids_path = "data/charville/charville_acq_ids_val.csv",
                num_workers = 4,
                )   
        pretrain_dm.setup()

    num_channels = 4 if args.use_seg_mask and not args.cell_mask_only else 3

else:
    train_patches = EMCellPatchDataset(
            uri = args.train_uri,
            patch_size=patch_size, 
            biomarker_subset=biomarkers,
            nucleus_mask=False
            )
    val_patches = EMCellPatchDataset(
            uri = args.val_uri,
            patch_size=patch_size, 
            biomarker_subset=biomarkers,
            nucleus_mask=False
            )
    
    key_datapipes = {'train': train_patches.cell_key_datapipe, 'val': val_patches.cell_key_datapipe} # datapipe that provides patch identifiers
    image_datapipes = {'train': train_patches.image_datapipe, 'val': val_patches.image_datapipe} # datapipe that provides patches
    combined_datapipes = {'train': image_datapipes['train'].zip(key_datapipes['train']), 'val': image_datapipes['val'].zip(key_datapipes['val']) }
    
    data_length = len(train_patches.cell_key_order)
    #print(next(iter(combined_datapipes['train']))[0].shape)
    
    dataloaders = {'train': DataLoader(dataset=combined_datapipes['train'], batch_size=batch_size, num_workers=0, pin_memory=True), 'val': DataLoader(dataset=combined_datapipes['val'], batch_size=batch_size, num_workers=0, pin_memory=True) }

    num_channels = len(biomarkers)

train_acq_ids = list(pd.read_csv(train_regions, header=None).iloc[:,0].values)
    
training_seed = 10384
pl.seed_everything(training_seed, workers=True)
model = LitClassifierModel(
            input_dim=patch_size**2 if not args.he_foundation else foundation_input_size, 
            patch_dim = patch_size, 
            latent_dim = latent_dim,
            num_channels=num_channels, 
            coarse_labels = use_coarse_labels, 
            binary_problem = binary_problem, 
            batch_size=batch_size, 
            train_acq_ids=train_acq_ids, 
            lr=args.learning_rate, 
            optimizer=args.optimizer, 
            zscale = zscale, 
            dataset=args.dataset,
            celltype_csv=celltype_csv,
            dropout=args.dropout, 
            num_conv_layers = args.num_conv_layers, 
            he_input=args.he,
            hierarchy=args.hierarchy,
            include_seg_mask=args.use_seg_mask,
            cell_mask_only=args.cell_mask_only,
            foundation_embed_input=args.he_foundation,
            hidden_dim=args.hidden_dim,
            batch_norm=args.batch_norm,
            weight_decay=args.weight_decay,
        )
if args.pretrain:
    pretrain_model = LitClassifierModel(
            input_dim=patch_size**2, 
            patch_dim = patch_size, 
            latent_dim = latent_dim,
            num_channels=num_channels, 
            coarse_labels = use_coarse_labels, 
            binary_problem = binary_problem, 
            batch_size=batch_size, 
            train_acq_ids=list(pd.read_csv("data/charville/charville_acq_ids_train.csv", header=None).iloc[:,0].values), 
            lr=args.learning_rate, 
            optimizer=args.optimizer, 
            zscale = zscale, 
            dataset="charville",
            celltype_csv="data/charville/charville_celltypes.csv",
            dropout=args.dropout, 
            num_conv_layers = args.num_conv_layers, 
            he_input=args.he,
            hierarchy=None,
            include_seg_mask=args.use_seg_mask,
            cell_mask_only=args.cell_mask_only,
        )


if not args.no_logger:
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.watch(model, log='all')

if args.no_logger:
    trainer = pl.Trainer(limit_train_batches=10, limit_val_batches=10, deterministic=True, accelerator='gpu', max_epochs=epochs, num_sanity_val_steps=0, log_every_n_steps=1, devices=[args.which_gpu])
    if args.pretrain:
        pretrainer = pl.Trainer(limit_train_batches=10, limit_val_batches=10, deterministic=True, accelerator='gpu', max_epochs=epochs, num_sanity_val_steps=0, log_every_n_steps=1, devices=[args.which_gpu])
else:
    if args.save_checkpoint_every_n_epochs:
        checkpoint_callback = ModelCheckpoint(every_n_epochs=args.save_checkpoint_every_n_epochs)
    trainer = pl.Trainer(logger=wandb_logger, deterministic=True, accelerator='gpu', max_epochs=epochs, log_every_n_steps=10, devices=[args.which_gpu], callbacks=[checkpoint_callback] if args.save_checkpoint_every_n_epochs else None)
    if args.pretrain:
        pretrainer = pl.Trainer(logger=wandb_logger, deterministic=True, accelerator='gpu', max_epochs=epochs, log_every_n_steps=10, devices=[args.which_gpu])

if args.he:
    if args.pretrain:
        pretrainer.fit(model=pretrain_model, datamodule=pretrain_dm)
        for i, conv_layer in enumerate(model.conv_layers):
            conv_layer.load_state_dict(pretrain_model.conv_layers[i].state_dict())
        model.fc1.load_state_dict(pretrain_model.fc1.state_dict())
    trainer.fit(model=model, datamodule=he_dm, ckpt_path=args.checkpoint)
else:
    trainer.fit(model=model, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['val'], ckpt_path=args.checkpoint)

if not args.no_logger:
    wandb.finish()


