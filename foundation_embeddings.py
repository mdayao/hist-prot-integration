import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTModel
from torchvision import transforms

from datamodules import HEDataset, KPMPDataset

import sys
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Save embeddings from foundation model')

parser.add_argument('--model-name', type=str, help='Name of model to use', choices=['phikon', 'quiltnet-b-32', 'uni', 'conch'])
parser.add_argument('--predict-dataset', type=str, choices=['renal-transplant', 'transplant-stanford', 'kpmp-healthy', 'kpmp-ckd', 'kpmp-aki'], help='Dataset to predict on')
parser.add_argument('--celltype-csv', type=str, help='Path to celltype csv file for prediction dataset.')
parser.add_argument('--hierarchy', type=int, help='Hierarchy level to predict on')
parser.add_argument('--output', type=str, help='Path to save embeddings')

parser.add_argument('--patch-size', type=int, default=64, help='Patch size for input images.')
parser.add_argument('--crop-and-rescale', action='store_true', help='Center crop to get .755 field of view and resize to 64x64')
parser.add_argument('--rescale', action='store_true', help='Resize to 64x64')

parser.add_argument('--which-gpu', type=int, help='Which GPU to use')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size for inference')

args = parser.parse_args()

if args.predict_dataset in ['renal-transplant', 'transplant-stanford']:
    # set up dataset
    he_dataset = HEDataset(
            data_dir = f'./data/{args.predict_dataset}/he_patches_{args.patch_size}',
            acq_ids = list(pd.read_csv(f'./data/{args.predict_dataset}/{args.predict_dataset}_acq_ids.csv', header=None).iloc[:,0].values),
            label_csv = args.celltype_csv,
            hierarchy = args.hierarchy,
            transform = None,
            randaugment=False,
            )
    
    # save dataset info
    dataset_df = pd.DataFrame(he_dataset.acq_id_cell_id_pairs, columns=['acq_id', 'cell_id', 'cell_idx', 'celltype_id'])
    # remove cell_idx and celltype_id
    dataset_df = dataset_df.drop(columns=['cell_idx', 'celltype_id'])
    dataset_df.to_csv(args.output.replace('.npy', '_dataset.csv'), index=False)
elif args.predict_dataset in ['kpmp-healthy', 'kpmp-ckd', 'kpmp-aki']:
    # set up dataset from KPMP
    he_dataset = KPMPDataset(
            data_dir = f'/home/ubuntu/enable_rnd/segment-he/data/{args.predict_dataset.split("-")[1]}/segment/patches_{args.patch_size}',
            metadata_file = f'/home/ubuntu/enable_rnd/segment-he/data/{args.predict_dataset.split("-")[1]}/{args.predict_dataset.split("-")[1]}_list.csv',
            )
    # save dataset info
    dataset_df = pd.DataFrame(he_dataset.acq_id_cell_id_pairs, columns=['acq_id', 'cell_id', 'cell_idx'])
    dataset_df = dataset_df.drop(columns=['cell_idx'])
    dataset_df.to_csv(args.output.replace('.npy', '_dataset.csv'), index=False)
else:
    raise NotImplementedError(f'Dataset {args.predict_dataset} not implemented')

# set up model
if args.model_name == 'phikon':
    image_processor = AutoImageProcessor.from_pretrained('owkin/phikon')
    model = ViTModel.from_pretrained('owkin/phikon', add_pooling_layer=False)
else:
    # not implemented yet
    raise NotImplementedError

# set up GPU
device = torch.device(f'cuda:{args.which_gpu}')
model.to(device)

# set up dataloader
dataloader = DataLoader(he_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# get embeddings
embeddings = []
for batch in tqdm.tqdm(dataloader):
    # if dataset is an HEDataset, not a KPMPDataset
    if isinstance(batch, tuple):
        imgs, labels = batch[0], batch[1]
    else:
        imgs = batch
    if args.crop_and_rescale:
        transform = transforms.Compose([
            transforms.CenterCrop(int(imgs.shape[1]*0.755)),
            transforms.Resize(64),
            ])
        imgs = transform(imgs)
    elif args.rescale:
        transform = transforms.Compose([
            transforms.Resize(64),
            ])
        imgs = transform(imgs)

    imgs.to(device)

    inputs = image_processor(imgs, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:,0,:].cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
np.save(args.output, embeddings)

assert embeddings.shape[0] == len(he_dataset), 'Embedding shape does not match dataset shape'

