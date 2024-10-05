import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, BatchSampler
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix

import os
os.environ['STAGE'] = 'prod'
import sys
#sys.path.append("/home/ubuntu/enable_rnd/rnd-utils/pythonutils/")

import emdatabase as emdb
from datamodules import HEDataset, HEDataPipe, HEDataModule
from classifier_model import LitClassifierModel
from contrastive_model import SimCLRModule, SupervisedModule
from emimagepatch import EMCellPatchDataset

from tqdm import tqdm
import argparse


def embed_imgs(model, dataloader, he_input=False, contrastive=False, train=False):
    """Encode all images in dataloader using model, return both images and embeddings"""
    model.eval()
    if contrastive:
        label_list, he_embed_list, cell_id_list = [], [], []
        if not he_input:
            codex_embed_list = []
        for i, batch in tqdm(enumerate(dataloader), desc="Encoding images", leave=False):
            with torch.no_grad():
                if he_input:
                    he_imgs, labels = batch[0], batch[1]
                    he_imgs.float().to(model.device)
                    if model.zero_one_normalize:
                        he_imgs = he_imgs / 255.
                else:
                    codex_imgs, he_imgs, labels, _ = model._get_data_for_step(batch, train=train, embed=True)
                    codex_z = model.embed(codex_imgs.to(model.device), he=False)
                he_z = model.embed(he_imgs.to(model.device), he=True)
            label_list.append(labels)
            he_embed_list.append(he_z)
            if not he_input:
                codex_embed_list.append(codex_z)
                cell_id_list = cell_id_list + list(batch[2])
        if not he_input:
            return torch.cat(label_list, dim=0).cpu().numpy(), torch.cat(codex_embed_list, dim=0).cpu().numpy(), torch.cat(he_embed_list, dim=0).cpu().numpy(), pd.DataFrame(cell_id_list)
        cell_id_list = dataloader.dataset.acq_id_cell_id_pairs
        cell_id_list = [f"{x[0]}_{x[1]}" for x in cell_id_list]
        return torch.cat(label_list, dim=0).cpu().numpy(), None, torch.cat(he_embed_list, dim=0).cpu().numpy(), pd.DataFrame(cell_id_list)
    else:
        label_list, embed_list, cell_id_list = [], [], []
        if he_input: # H&E data
            cell_id_list = [f"{x[0]}_{x[1]}" for x in dataloader.dataset.acq_id_cell_id_pairs]
            for imgs, labels in tqdm(dataloader, desc="Encoding images", leave=False):
                with torch.no_grad():
                    z = model.embed(imgs.to(model.device))
                label_list.append(labels)
                embed_list.append(z)
        else: # CODEX data
            for batch in tqdm(dataloader, desc="Encoding images", leave=False):
                with torch.no_grad():
                    imgs, labels, _ = model._get_data_for_step(batch, train=False)
                    z = model.embed(imgs.to(model.device))
                label_list.append(labels)
                embed_list.append(z)
                cell_id_list = cell_id_list + batch[1]

        return torch.cat(label_list, dim=0).cpu().numpy(), torch.cat(embed_list, dim=0).cpu().numpy(), pd.DataFrame(cell_id_list)

def get_predictions(model, dataloader, train=False, he_only=False):
    """Get predictions for all images in dataloader using model and output confusion matrix"""
    model.eval()
    label_list, he_pred_list = [], []
    if not he_only:
        codex_pred_list = []
    for batch in tqdm(dataloader, desc="Getting predictions", leave=False):
        with torch.no_grad():
            if he_only:
                he_imgs, labels = batch[0], batch[1]
                he_imgs.float().to(model.device)
                if model.zero_one_normalize:
                    he_imgs = he_imgs / 255.
            else:
                codex_imgs, he_imgs, labels, _ = model._get_data_for_step(batch, train=train, embed=True)
                codex_preds = model.predict(codex_imgs.to(model.device), labels, he=False)
            he_preds = model.predict(he_imgs.to(model.device), labels, he=True)
        label_list.append(labels)
        he_pred_list.append(he_preds)
        if not he_only:
            codex_pred_list.append(codex_preds)

    all_labels = torch.cat(label_list, dim=0).cpu().numpy()
    all_he_preds = torch.cat(he_pred_list, dim=0).cpu().numpy()
    he_confusion = confusion_matrix(all_labels, all_he_preds)
    if not he_only:
        all_codex_preds = torch.cat(codex_pred_list, dim=0).cpu().numpy()
        codex_confusion = confusion_matrix(all_labels, all_codex_preds)
        return all_labels, all_codex_preds, all_he_preds, codex_confusion, he_confusion
    return all_labels, None, all_he_preds, None, he_confusion

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, choices=['hnscc-atlas', 'renal-transplant', 'transplant-stanford'], default='hnscc-atlas', help='Dataset to embed')
    parser.add_argument('--model-type', type=str, choices=['classifier', 'contrastive'], default='classifier', help='Model type to embed with')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--celltype-csv', type=str, default='./data/hnscc-atlas/HNSCC_Atlas_celltypes_hierarchy.csv')
    parser.add_argument('--celltype-hierarchy', type=int, help='Hierarchy level to train on')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--train-or-val', type=str, help='compute embeddings for train or val dataset. Can also choose to compute embeddings for entire dataset with "all".', choices=['train', 'val', 'all'], default='val')
    parser.add_argument('--acq-ids-path', type=str, help='List of acquisition ids (corresponding to either train or val acq ids)', default='data/renal-transplant/renal-transplant_acq_ids.csv')
    parser.add_argument('--train-acq-ids-path', type=str, help='List of acqusition ids for training', default='data/renal-transplant/renal-transplant_acq_ids_train.csv')
    parser.add_argument('--val-acq-ids-path', type=str, help='List of acqusition ids for validation', default='data/renal-transplant/renal-transplant_acq_ids_val.csv')
    parser.add_argument('--he', action='store_true', help='Use H&E input. In contrastive model, this will only embed H&E images.')
    parser.add_argument('--he-patches-dir', type=str, help='Directory of H&E patches', default='data/renal-transplant/he_patches_64')
    parser.add_argument('--train-uri', help='path to tiledb dataset for training', type=str)
    parser.add_argument('--val-uri', help='path to tiledb dataset for validation', type=str)
    parser.add_argument('--output-dir', help='path to save embeddings', type=str, default='./embeddings')
    parser.add_argument('--output-name', help='name of output file', type=str) 
    parser.add_argument('--which-gpu', type=int, help='Which GPU to for embedding', default=0)
    parser.add_argument('--verbose', help='set if want to print out dataset creation info', action='store_true')
    parser.add_argument('--finetune', help='set if using finetuned model', action='store_true')
    parser.add_argument('--simclr-checkpoint', help='path to simclr checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    
    patch_size = 64
    batch_size = args.batch_size
    celltype_csv = args.celltype_csv
    
    # read in biomarkers
    if args.model_type == 'contrastive':
        train_acq_ids = list(pd.read_csv(args.train_acq_ids_path, header=None).iloc[:,0].values)
        sample_acq_id = train_acq_ids[0]
    else:
        sample_acq_id = 'HNSCC_c001_v001_r001_reg001' if args.dataset == 'hnscc-atlas' else 's314_c001_v001_r001_reg002'
    cur = emdb.connect()
    biomarkers = emdb.get_all_biomarkers_for_acquisition_id(sample_acq_id, cur)
    biomarkers.remove('DAPI')
    if args.dataset == 'hnscc-atlas':
        biomarkers.remove('DRAQ5') # a reference marker that isn't needed for HNSCC Atlas
    biomarkers = ['DAPI'] + biomarkers
    
    # Load the model
    if args.model_type == 'classifier':
        model = LitClassifierModel.load_from_checkpoint(args.model_path, map_location='cuda'
            )
        if args.he:
            # set up datamodule for H&E data
            he_dm = HEDataModule(
                    data_dir = 'data/hnscc-atlas/he_patches_64',
                    label_csv = celltype_csv,
                    hierarchy = model.hierarchy,
                    batch_size = batch_size,
                    train_acq_ids_path = 'data/hnscc-atlas/HNSCC_Atlas_acq_ids_train.csv',
                    val_acq_ids_path = 'data/hnscc-atlas/HNSCC_Atlas_acq_ids_val.csv',
                    num_workers = 1,
                    )
            he_dm.setup()
            if args.train_or_val == 'train':
                dataloader = he_dm.train_dataloader()
            else:
                dataloader = he_dm.val_dataloader()
        
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
            
            dataloaders = {'train': DataLoader(dataset=combined_datapipes['train'], batch_size=batch_size, num_workers=0, pin_memory=True), 'val': DataLoader(dataset=combined_datapipes['val'], batch_size=batch_size, num_workers=0, pin_memory=True) }
        
            dataloader = dataloaders[args.train_or_val]

        # compute embeddings
        labels, embeddings, cell_id_df = embed_imgs(model, dataloader, args.he)
   
        # save embeddings
        np.save(os.path.join(args.output_dir, args.output_name + f'{args.train_or_val}_labels.npy'), labels)
        np.save(os.path.join(args.output_dir, args.output_name + f'{args.train_or_val}_embed.npy'), embeddings)
        cell_id_df.to_csv(os.path.join(args.output_dir, args.output_name + f'{args.train_or_val}_cell_ids.csv'), header=False, index=False)

    elif args.model_type == 'contrastive':

        if args.train_or_val == 'all':
            acq_ids = list(pd.read_csv(args.acq_ids_path, header=None).iloc[:,0].values)
            he_dataset = HEDataset(
                    data_dir=args.he_patches_dir,
                    acq_ids=acq_ids,
                    label_csv=args.celltype_csv,
                    hierarchy=args.celltype_hierarchy,
                    transform=None,
                    randaugment=False,
                    verbose=args.verbose,
                    )
        else:
            train_acq_ids = list(pd.read_csv(args.train_acq_ids_path, header=None).iloc[:,0].values)
            val_acq_ids = list(pd.read_csv(args.val_acq_ids_path, header=None).iloc[:,0].values)
            # Create H&E datasets
            train_he_dataset = HEDataset(
                    data_dir=args.he_patches_dir,
                    acq_ids=train_acq_ids,
                    label_csv=args.celltype_csv,
                    hierarchy=args.celltype_hierarchy,
                    transform=None,
                    randaugment=False,
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

        if not args.he: # Also embed codex images
            if args.train_or_val == 'all':
                raise ValueError('Cannot embed codex images for entire dataset yet. Please choose train or val or use H&E only.')
            # filter out cells that don't have H&E images
            # loop through cell keys, check if they are in he_dataset.acq_id_cell_id_to_idx
            train_patches = EMCellPatchDataset(
                    path = args.train_uri,
                    patch_size=args.patch_size, 
                    biomarker_subset=biomarkers,
                    nucleus_mask=False
                    )
            val_patches = EMCellPatchDataset(
                    path = args.val_uri,
                    patch_size=args.patch_size,
                    biomarker_subset=biomarkers,
                    nucleus_mask=False
                    )
            key_datapipes = {'train': train_patches.cell_key_datapipe, 'val': val_patches.cell_key_datapipe} # datapipe that provides patch identifiers
            print('before filtering train', len(key_datapipes['train']))
            train_subset_inds = []
            for i, cell_key in enumerate(key_datapipes['train']):
                if cell_key not in train_he_dataset.acq_id_cell_id_to_idx:
                    continue
                else:
                    train_subset_inds.append(i)
            print('before filtering val', len(key_datapipes['val']))
            val_subset_inds = []
            for i, cell_key in enumerate(key_datapipes['val']):
                if cell_key not in val_he_dataset.acq_id_cell_id_to_idx:
                    continue
                else:
                    val_subset_inds.append(i)

            # Recreate dataset with filtered cells only
            train_patches = EMCellPatchDataset(
                    path = args.train_uri,
                    patch_size=args.patch_size, 
                    biomarker_subset=biomarkers,
                    nucleus_mask=False,
                    subset_inds=train_subset_inds,
                    )
            val_patches = EMCellPatchDataset(
                    path = args.val_uri,
                    patch_size=args.patch_size,
                    biomarker_subset=biomarkers,
                    nucleus_mask=False,
                    subset_inds=val_subset_inds,
                    )
            key_datapipes = {'train': train_patches.cell_key_datapipe, 'val': val_patches.cell_key_datapipe} # datapipe that provides patch identifiers
            print('after filtering train', len(key_datapipes['train']))
            print('after filtering val', len(key_datapipes['val']))
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
                    }

            # combine codex image datapipe, he image datapipe, and key datapipes
            combined_datapipes = {
                    'train': image_datapipes['train'].zip(he_datapipes['train'], key_datapipes['train']), 
                    'val': image_datapipes['val'].zip(he_datapipes['val'], key_datapipes['val'])
                    }

            dataloaders = {'train': DataLoader(dataset=combined_datapipes['train'], batch_size=args.batch_size, num_workers=0), 'val': DataLoader(dataset=combined_datapipes['val'], batch_size=args.batch_size, num_workers=0) }
        else:
            # create dataloaders for H&E images only
            if args.train_or_val == 'all':
                he_dataloaders = {'all': DataLoader(dataset=he_dataset, batch_size=args.batch_size, num_workers=0) }
            else:
                he_dataloaders = {'train': DataLoader(dataset=train_he_dataset, batch_size=args.batch_size, num_workers=0), 'val': DataLoader(dataset=val_he_dataset, batch_size=args.batch_size, num_workers=0) }
            dataloaders = he_dataloaders

        if args.finetune:
            simclr_model = SimCLRModule.load_from_checkpoint(args.simclr_checkpoint, map_location=f'cuda:{args.which_gpu}')
            model = SupervisedModule.load_from_checkpoint(
                    args.model_path,
                    simclr_module=simclr_model,
                    map_location=f'cuda:{args.which_gpu}',
                    )
            output_dir = os.path.join(args.output_dir, 'finetune')
        else:
            model = SimCLRModule.load_from_checkpoint(args.model_path, map_location=f'cuda:{args.which_gpu}',
                    #he_channels=3, 
                    #codex_channels=len(biomarkers),
                    #num_layers=4, 
                    #encoder_filter_dim=64,
                    #projection_dim=128, 
                    #temperature=0.5, 
                    #lr=0.001,
                    )
            output_dir = args.output_dir

        dataloader = dataloaders[args.train_or_val]

        # get predictions
        if args.finetune:
            print('getting predictions')
            labels, codex_preds, he_preds, codex_confusion, he_confusion = get_predictions(model, dataloader, train=args.train_or_val=='train', he_only=args.he)
            np.save(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_he_preds.npy'), he_preds)
            np.save(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_he_confusion.npy'), he_confusion)
            if not args.he:
                np.save(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_codex_preds.npy'), codex_preds)
                np.save(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_codex_confusion.npy'), codex_confusion)

        # compute embeddings
        print('computing embeddings')
        labels, codex_embeddings, he_embeddings, cell_id_df = embed_imgs(model, dataloader, contrastive=True, train=args.train_or_val=='train', he_input=args.he)
   
        # save embeddings
        np.save(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_labels.npy'), labels)
        np.save(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_he_embed.npy'), he_embeddings)
        if not args.he:
            np.save(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_codex_embed.npy'), codex_embeddings)

        cell_id_df.to_csv(os.path.join(output_dir, args.output_name + f'{args.train_or_val}_cell_ids.csv'), header=False, index=False)


