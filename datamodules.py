import torch
from torch import nn, optim, Tensor
import pytorch_lightning as pl
from torchvision import transforms
import sys

import numpy as np
import pandas as pd

import glob
import tqdm
import itertools
import os

from classifier_model import CustomRotationTransform
from torch.utils.data import Dataset, DataLoader
from torchdata.datapipes.iter import IterDataPipe

from he_randaugment.randaugment import distort_image_with_randaugment

from typing import Optional, List

# Create a Pytorch Dataset for a custom dataset located in the data/hnscc-atlas/he_patches_64 folder
class HEDataset(Dataset):
    def __init__(self, data_dir, acq_ids, 
            label_csv='data/hnscc-atlas/HNSCC_Atlas_celltypes.csv',
            hierarchy=None,
            transform=None,
            randaugment=False,
            randaugment_params=None,
            include_seg_mask=False,
            cell_mask_only=False,
            seg_mask_dir=None,
            verbose=False
            ):
        self.data_dir = data_dir
        self.label_df = pd.read_csv(label_csv)
        self.hierarchy = hierarchy
        self.acq_ids = acq_ids
        self.transform = transform
        self.randaugment = randaugment
        self.randaugment_params = randaugment_params
        self.include_seg_mask = include_seg_mask 
        self.cell_mask_only = cell_mask_only
        self.seg_mask_dir = seg_mask_dir

        if self.include_seg_mask and self.seg_mask_dir is None:
            raise ValueError('seg_mask_dir must be provided if include_seg_mask is True')
        if not self.include_seg_mask and self.cell_mask_only:
            raise ValueError('cell_mask_only can only be True if include_seg_mask is True')

        # Assign cell type label column based on hierarchy
        if self.hierarchy is None or self.hierarchy == -1:
            celltype_label_col = 'celltype_label'
        else:
            celltype_label_col = f'celltype_hierarchy_{self.hierarchy}'

        # Assigning unique integer ids to each celltype label
        self.label_df['celltype_id'], unique_celltypes = pd.factorize(self.label_df[celltype_label_col])

        if verbose: 
            print('Number of celltypes:', len(self.label_df.celltype_id.unique()))
            print('Unique celltypes:', unique_celltypes)
       
        # Create a list of tuples of the form (acq_id, cell_id, cell_idx, celltype_id) for each cell in the dataset
        # Each acq_id matches a pattern of the form 'data/hnscc-atlas/he_patches_64/{acq_id}_{cell_id}.npy'
        self.acq_id_cell_id_pairs = []
        if self.include_seg_mask:
            self.all_seg_masks = []
        for acq_id in tqdm.tqdm(self.acq_ids, desc='Creating dataset', total=len(self.acq_ids), leave=verbose):
            label_df_for_acq_id = self.label_df[self.label_df.acquisition_id == acq_id]
            label_df_for_acq_id = label_df_for_acq_id.copy() # make a copy to avoid SettingWithCopyWarning

            # if the dataframe is empty (no labeled cells for this acquisition), skip it
            if label_df_for_acq_id.shape[0] == 0:
                continue

            cell_ids = [int(cell_path.split('_')[-1].replace('.npy', '')) for cell_path in sorted(glob.glob(f'{self.data_dir}/{acq_id}_*.npy'))]

            # Reordering label_df_for_acq_id to match the order of the cell_ids
            label_df_for_acq_id.loc[:,'cell_id'] = pd.Categorical(label_df_for_acq_id['cell_id'], categories=cell_ids, ordered=True)
            label_df_for_acq_id = label_df_for_acq_id.sort_values(by='cell_id')
            label_df_for_acq_id.reset_index(drop=True, inplace=True)

            nan_count = label_df_for_acq_id['cell_id'].isna().sum() # count for nans in label_df in cell_id column (only occurs labeled cell has no corresponding cell patch)

            # creating list of celltype_ids matching the order of the cell_ids
            celltype_ids = label_df_for_acq_id['celltype_id'].values

            counter = itertools.count(0)
            self.acq_id_cell_id_pairs = self.acq_id_cell_id_pairs + [(acq_id, cell_id, i, celltype_ids[next(counter)]) for i, cell_id in enumerate(cell_ids) if cell_id in label_df_for_acq_id['cell_id'].values]

            assert next(counter) + nan_count == len(celltype_ids), f'self.acq_id_cell_id_pairs counter: {next(counter)-1}, celltype_ids: {len(celltype_ids)}, nan_count: {nan_count}'

            if self.include_seg_mask:
                seg_masks = [np.load(f'{self.seg_mask_dir}/{acq_id}_{cell_id}.npy') for cell_id in cell_ids]
                self.all_seg_masks = self.all_seg_masks + seg_masks

        # Create a dictionary of numpy arrays that contain the cell patches for all cells for each acq_id
        # Each combined array matches the pattern of the form 'data/hnscc-atlas/he_patches_64/combined/{acq_id}.npy'
        self.acq_id_combined_cell_patches = {}
        for acq_id in self.acq_ids:
            # check if the combined file exists
            if not os.path.exists(f'{self.data_dir}/combined/{acq_id}.npy'):
                continue
            self.acq_id_combined_cell_patches[acq_id] = np.load(f'{self.data_dir}/combined/{acq_id}.npy')

        # Reverse mapping dictionary of acquisition id, cell id to index in the dataset
        self.acq_id_cell_id_to_idx = {f'{acq_id}_{cell_id}': idx for idx, (acq_id, cell_id, _, _) in enumerate(self.acq_id_cell_id_pairs)}


    def __len__(self):
        # Compute the total number of cells in the dataset
        return len(self.acq_id_cell_id_pairs)

    def __getitem__(self, idx):
        # Get the acq_id and cell_id for the given index
        acq_id, cell_id, cell_idx, celltype_id = self.acq_id_cell_id_pairs[idx]
        img = self.acq_id_combined_cell_patches[acq_id][cell_idx, ...]
        if self.transform or self.randaugment:
            # if the image dimensions (3, 64, 64), change it to (64, 64, 3) for transforms
            if img.shape[0] == 3 or img.shape[0] == 4:
                img = np.moveaxis(img, 0, -1)
            if self.randaugment:
                img = np.array(distort_image_with_randaugment(img, num_layers=int(self.randaugment_params[0]), magnitude=int(self.randaugment_params[1]), ra_type='Default'))
            if self.transform:
                img = self.transform(img)
            else:
                img = np.moveaxis(img, -1, 0) # change shape back to (3, 64, 64) if no transform
        if self.include_seg_mask:
            seg_mask = self.all_seg_masks[idx]
            if self.cell_mask_only:
                # Set all non-cell pixels to 0 in img
                img[:,seg_mask==0] = 0
            else:
                # Add seg_mask as the first channel of the image, now num_channels = 4
                img = np.append(np.expand_dims(seg_mask, axis=0), img, axis=0)

        return img, celltype_id

# Create a PyTorch IterDataPipe from HEDataset above
class HEDataPipe(IterDataPipe):
    def __init__(self, dataset: HEDataset, dataset_order: Optional[List[int]] = None):
        self.dataset = dataset
        self.dataset_order = dataset_order

    def __iter__(self):
        if self.dataset_order is None:
            for idx in range(len(self.dataset)):
                yield self.dataset[idx]
        else:
            for idx in self.dataset_order:
                yield self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

# Create a Lightning datamodule for a custom dataset located in the data/hnscc-atlas/he_patches_64 folder
class HEDataModule(pl.LightningDataModule):
    def __init__(self, 
            data_dir: str = 'data/hnscc-atlas/he_patches_64/combined', 
            label_csv: str = 'data/hnscc-atlas/HNSCC_Atlas_celltypes.csv',
            hierarchy: int = None,
            randaugment: bool = False,
            randaugment_params: List[int] = None,
            include_seg_mask: bool = False,
            cell_mask_only: bool = False,
            seg_mask_dir: str = 'data/hnscc-atlas/seg_mask_nucl_64',
            batch_size: int = 32,
            num_workers: int = 4,
            train_acq_ids_path: str = 'data/hnscc-atlas/HNSCC_Atlas_acq_ids_train.csv',
            val_acq_ids_path: str = None,
            verbose: bool = False,
            ):
        super().__init__()
        self.data_dir = data_dir
        self.label_csv = label_csv
        self.hierarchy = hierarchy
        self.randaugment = randaugment
        self.randaugment_params = randaugment_params
        self.include_seg_mask = include_seg_mask
        self.cell_mask_only = cell_mask_only
        self.seg_mask_dir = seg_mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose

        self.train_acq_ids = list(pd.read_csv(train_acq_ids_path, header=None).iloc[:,0].values)
        self.val_acq_ids = list(pd.read_csv(val_acq_ids_path, header=None).iloc[:,0].values) if val_acq_ids_path is not None else None

        # list transforms to randomly rotate and flip images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            CustomRotationTransform(angles=[-90, 0, 90, 180]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU
        pass
    
    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        # called on every GPU

        if stage == 'fit' or stage is None:
            if self.verbose:
                print('Setting up training dataset')
            self.train_dataset = HEDataset(
                    self.data_dir, 
                    self.train_acq_ids, 
                    self.label_csv, 
                    self.hierarchy,
                    transform=self.transform,
                    randaugment=self.randaugment,
                    randaugment_params=self.randaugment_params,
                    include_seg_mask=self.include_seg_mask,
                    cell_mask_only=self.cell_mask_only,
                    seg_mask_dir=self.seg_mask_dir,
                    verbose=self.verbose,
                    )
            if self.verbose:
                print('Setting up validation dataset')
            self.val_dataset = HEDataset(
                    self.data_dir, 
                    self.val_acq_ids, 
                    self.label_csv, 
                    self.hierarchy,
                    transform = transforms.ToTensor(),
                    randaugment=False,
                    include_seg_mask=self.include_seg_mask,
                    cell_mask_only=self.cell_mask_only,
                    seg_mask_dir=self.seg_mask_dir,
                    verbose=self.verbose,
                    ) if self.val_acq_ids is not None else None

        elif stage == 'eval':
            if self.verbose:
                print('Setting up validation dataset only')
            if self.val_acq_ids is None:
                raise ValueError('val_acq_ids_path must be provided if stage is eval')
            self.val_dataset = HEDataset(
                    self.data_dir, 
                    self.val_acq_ids, 
                    self.label_csv, 
                    self.hierarchy,
                    transform = transforms.ToTensor(),
                    randaugment=False,
                    include_seg_mask=self.include_seg_mask,
                    cell_mask_only=self.cell_mask_only,
                    seg_mask_dir=self.seg_mask_dir,
                    verbose=self.verbose,
                    )

        return
    
    def train_dataloader(self):
        # return a PyTorch DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        # return a PyTorch DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True) if self.val_dataset is not None else None
    
    def predict_dataloader(self):
        # return a PyTorch DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

# Create Pytorch dataset for foundation embedding dataset for H&E
class HEFoundationDataset(Dataset):
    def __init__(self, 
            embeds_path, 
            ids_path,
            acq_ids, 
            label_csv,
            transform=None,
            verbose=False
            ):
        self.embeds = np.load(embeds_path)
        self.cell_ids = pd.read_csv(ids_path)
        self.label_df = pd.read_csv(label_csv)
        self.acq_ids = acq_ids
        self.transform = transform

        self.label_df = self.label_df.loc[self.label_df['acquisition_id'].isin(self.acq_ids)]
        self.label_df = self.label_df.merge(self.cell_ids, left_on=['acquisition_id', 'cell_id'], right_on=['acq_id', 'cell_id'], how='right')
        self.embeds = self.embeds[~self.label_df['celltype_id'].isna(),:]
        self.labels, _ = pd.factorize(self.label_df.loc[~self.label_df['celltype_id'].isna(), 'celltype_id'], sort=True)
        assert self.embeds.shape[0] == len(self.labels)

        self.label_df = self.label_df.loc[~self.label_df['celltype_id'].isna(),:]

        self.celltype_names = pd.value_counts(self.label_df.sort_values('celltype_id').celltype_label, sort=False).index

        self.acq_id_cell_id_pairs = [(acq_id, cell_id, 0, 0) for acq_id, cell_id in zip(self.label_df['acquisition_id'], self.label_df['cell_id'])]

        self.acq_id_cell_id_to_idx = {f'{acq_id}_{cell_id}': idx for idx, (acq_id, cell_id) in enumerate(zip(self.label_df['acquisition_id'], self.label_df['cell_id']))}

    def __len__(self):
        return self.embeds.shape[0]

    def __getitem__(self, idx):
        embed = self.embeds[idx, ...]
        if self.transform:
            embed = self.transform(embed)
        return embed, self.labels[idx]

# Create a PyTorch IterDataPipe from HEFoundationDataset above
class HEFoundationDataPipe(IterDataPipe):
    def __init__(self, dataset: HEFoundationDataset, dataset_order: Optional[List[int]] = None):
        self.dataset = dataset
        self.dataset_order = dataset_order

    def __iter__(self):
        if self.dataset_order is None:
            for idx in range(len(self.dataset)):
                yield self.dataset[idx]
        else:
            for idx in self.dataset_order:
                yield self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

# Create a Lightning datamodule for foundation embeddings dataset for H&E
class HEFoundationDataModule(pl.LightningDataModule):
    def __init__(self, 
            embeds_path: str, 
            ids_path: str,
            train_acq_ids_path: str,
            label_csv: str = 'data/hnscc-atlas/HNSCC_Atlas_celltypes.csv',
            batch_size: int = 32,
            num_workers: int = 4,
            transform=None,
            verbose: bool = False,
            val_acq_ids_path: str = None,
            ):
        super().__init__()
        self.embeds_path = embeds_path
        self.ids_path = ids_path
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.verbose = verbose

        self.train_acq_ids = list(pd.read_csv(train_acq_ids_path, header=None).iloc[:,0].values)
        self.val_acq_ids = list(pd.read_csv(val_acq_ids_path, header=None).iloc[:,0].values) if val_acq_ids_path is not None else None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU
        pass
    
    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        # called on every GPU

        if stage == 'fit' or stage is None:
            if self.verbose:
                print('Setting up training dataset')
            self.train_dataset = HEFoundationDataset(
                    self.embeds_path, 
                    self.ids_path,
                    self.train_acq_ids,
                    self.label_csv, 
                    transform=self.transform,
                    verbose=self.verbose,
                    )
            if self.verbose:
                print('Setting up validation dataset')
            self.val_dataset = HEFoundationDataset(
                    self.embeds_path, 
                    self.ids_path,
                    self.val_acq_ids, 
                    self.label_csv, 
                    transform=self.transform,
                    verbose=self.verbose,
                    ) if self.val_acq_ids is not None else None

        elif stage == 'eval':
            if self.verbose:
                print('Setting up validation dataset only')
            if self.val_acq_ids is None:
                raise ValueError('val_acq_ids_path must be provided if stage is eval')
            self.val_dataset = HEFoundationDataset(
                    self.embeds_path, 
                    self.ids_path,
                    self.val_acq_ids,
                    self.label_csv, 
                    transform=self.transform,
                    verbose=self.verbose,
                    )

        return
    
    def train_dataloader(self):
        # return a PyTorch DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        # return a PyTorch DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True) if hasattr(self, 'val_dataset') else None
    
    def predict_dataloader(self):
        # return a PyTorch DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True) if hasattr(self, 'val_dataset') else None

# Dataset for KPMP data
class KPMPDataset(Dataset):
    def __init__(
            self,
            data_dir,
            metadata_file,
            ):
        self.data_dir = data_dir
        self.all_metadata = pd.read_csv(metadata_file)

        self.metadata = self.all_metadata.groupby('Participant ID').first().reset_index()
        self.acq_ids = [file_name.split('.')[0] for file_name in self.metadata['File Name'].values]

        self.acq_id_cell_id_pairs = []
        for acq_id in tqdm.tqdm(self.acq_ids, desc='Creating dataset', total=len(self.acq_ids), leave=True):
            # if directory does not exist, skip
            if not os.path.exists(f'{self.data_dir}/{acq_id}'):
                continue
            cell_ids = [int(cell_path.split('_')[-1].replace('.npy', '')) for cell_path in sorted(glob.glob(f'{self.data_dir}/{acq_id}/{acq_id}_*.npy'))]

            self.acq_id_cell_id_pairs = self.acq_id_cell_id_pairs + [(acq_id, cell_id, i) for i, cell_id in enumerate(cell_ids)]

        self.acq_id_combined_cell_patches = {}
        for acq_id in self.acq_ids:
            # check if the combined file exists
            if not os.path.exists(f'{self.data_dir}/combined/{acq_id}.npy'):
                continue
            self.acq_id_combined_cell_patches[acq_id] = np.load(f'{self.data_dir}/combined/{acq_id}.npy')

    def __len__(self):
        return len(self.acq_id_cell_id_pairs)

    def __getitem__(self, idx):
        acq_id, cell_id, cell_idx = self.acq_id_cell_id_pairs[idx]
        img = self.acq_id_combined_cell_patches[acq_id][cell_idx, ...]
        return img


