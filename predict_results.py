import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import os
os.environ["STAGE"]="prod"

import emdatabase as emdb
from classifier_model import LitClassifierModel
from datamodules import HEDataset, HEDataModule, HEFoundationDataset, HEFoundationDataModule
from contrastive_model import SimCLRModule, SupervisedModule

import tqdm

import argparse

parser = argparse.ArgumentParser(description='Predict on a H&E dataset')

parser.add_argument('--joint', action='store_true', help='Use finetuned joint contrastive model for prediction. Default is non-constrastive model')
parser.add_argument('--model-checkpoint', type=str, help='Path to model checkpoint. For joint model, provide path to finetuned checkpoint.')
parser.add_argument('--joint-checkpoint', type=str, help='Path to joint model checkpoint (only used for initialization of finetuned model). Do not provide if not using joint model.')
parser.add_argument('--disable-track-running-stats', action='store_true', help='Disable tracking running stats for batchnorm layers. Default is to track running stats.')

parser.add_argument('--he-foundation', help='set if using H&E foundation data', action='store_true')
parser.add_argument('--he-foundation-path', type=str, help='Path to H&E foundation dataset')
parser.add_argument('--he-foundation-ids-path', type=str, help='Path to H&E foundation embedding ids')

parser.add_argument('--predict-dataset', type=str, choices=['renal-transplant', 'transplant-stanford'], help='Dataset to predict on')
parser.add_argument('--predict-full', action='store_true', help='Predict on full dataset. Default is to predict on just the validation split')
parser.add_argument('--output-csv', type=str, help='Path to save predictions')

parser.add_argument('--model-celltype-csv', type=str, help='Path to celltype csv file used to train model.')
parser.add_argument('--celltype-csv', type=str, help='Path to celltype csv file for prediction dataset.')
parser.add_argument('--hierarchy', type=int, help='Hierarchy level to predict on')

parser.add_argument('--predict-res', type=float, help='Resolution of images to predict on in microns per pixel')
parser.add_argument('--train-res', type=float, help='Resolution of images model was trained on in microns per pixel')

parser.add_argument('--which-gpu', type=int, help='Which GPU to use')

args = parser.parse_args()

def rescale(x, predict_res, train_res):
    assert predict_res > train_res, "Predict resolution must be greater than train resolution"
    scale_factor = predict_res / train_res
    scale_transform = transforms.Compose([
        transforms.Resize((int(64*scale_factor), int(64*scale_factor))),
        transforms.CenterCrop(64),
        ])
    return scale_transform(x)

track_running_stats = not args.disable_track_running_stats

# set up dataset
if args.joint:
    if args.he_foundation:
        he_dataset = HEFoundationDataset(
                embeds_path = args.he_foundation_path,
                ids_path = args.he_foundation_ids_path,
                acq_ids = list(pd.read_csv(f'./data/{args.predict_dataset}/{args.predict_dataset}_acq_ids{"" if args.predict_full else "_val"}.csv', header=None).iloc[:,0].values),
                label_csv = args.celltype_csv,
                transform = None,
                verbose = True,
                )
    else:
        he_dataset = HEDataset(
                data_dir = f'./data/{args.predict_dataset}/he_patches_64',
                acq_ids = list(pd.read_csv(f'./data/{args.predict_dataset}/{args.predict_dataset}_acq_ids{"" if args.predict_full else "_val"}.csv', header=None).iloc[:,0].values),
                label_csv = args.celltype_csv,
                hierarchy = args.hierarchy,
                transform = None,
                randaugment=False,
                )

    simclr_model = SimCLRModule.load_from_checkpoint(args.joint_checkpoint, map_location=torch.device(f'cuda:{args.which_gpu}'), track_running_stats=track_running_stats, he_encoder_type='cnn' if not args.he_foundation else 'fc', input_patch_size=64, strict=False)
    model = SupervisedModule.load_from_checkpoint(
            args.model_checkpoint,
            simclr_module = simclr_model,
            celltype_csv = args.model_celltype_csv,
            map_location=torch.device(f'cuda:{args.which_gpu}'),
            )
else:
    if args.he_foundation:
        he_dm = HEFoundationDataModule(
            embeds_path = args.he_foundation_path,
            ids_path = args.he_foundation_ids_path,
            train_acq_ids_path = f'./data/{args.predict_dataset}/{args.predict_dataset}_acq_ids_train.csv',
            label_csv = args.celltype_csv,
            batch_size = 2048,
            num_workers = 1,
            val_acq_ids_path = f'./data/{args.predict_dataset}/{args.predict_dataset}_acq_ids{"" if args.predict_full else "_val"}.csv',
            )
    else:
        he_dm = HEDataModule(
                data_dir = f'./data/{args.predict_dataset}/he_patches_64',
                label_csv = args.celltype_csv,
                hierarchy = args.hierarchy,
                batch_size = 1024,
                randaugment = False,
                train_acq_ids_path = f'./data/{args.predict_dataset}/{args.predict_dataset}_acq_ids_train.csv',
                val_acq_ids_path = f'./data/{args.predict_dataset}/{args.predict_dataset}_acq_ids{"" if args.predict_full else "_val"}.csv',
                num_workers = 1,
                )
    he_dm.setup('eval')
    he_dataset = he_dm.val_dataset

    model = LitClassifierModel.load_from_checkpoint(
            args.model_checkpoint,
            celltype_csv = args.model_celltype_csv,
            map_location=torch.device(f'cuda:{args.which_gpu}'),
            )

dataloader = DataLoader(he_dataset, batch_size=1024, shuffle=False)

model.eval()

label_list, pred_list = [], []
for batch in tqdm.tqdm(dataloader, desc="Getting predictions"):
    with torch.no_grad():
        if args.joint:
            he_imgs, labels = batch[0], batch[1]
            he_imgs.float().to(model.device)
            if args.predict_res is not None and args.train_res is not None:
                he_imgs = rescale(he_imgs, args.predict_res, args.train_res)
            if model.zero_one_normalize:
                he_imgs = he_imgs / 255.
            preds = model.predict(he_imgs.to(model.device), labels, he=True)
        else:
            he_imgs, labels, _ = model._get_data_for_step(batch, train=False)
            he_imgs.float().to(model.device)
            if args.predict_res is not None and args.train_res is not None:
                he_imgs = rescale(he_imgs, args.predict_res, args.train_res)
            outputs = model(he_imgs.to(model.device))
            preds = torch.argmax(outputs, dim=1)
    label_list.append(labels)
    pred_list.append(preds)
all_labels = torch.cat(label_list, dim=0).cpu().numpy()
all_preds = torch.cat(pred_list, dim=0).cpu().numpy()

# save predictions
data_acq_ids, data_cell_ids, _, _ = zip(*he_dataset.acq_id_cell_id_pairs)
data_acq_ids = np.array(data_acq_ids)
data_cell_ids = np.array(data_cell_ids)

pred_df = pd.DataFrame(
        data={
            'acquisition_id': data_acq_ids,
            'cell_id': data_cell_ids,
            'celltype_id': all_labels,
            'predictions': all_preds,
            })

pred_df.to_csv(args.output_csv, index=False)


