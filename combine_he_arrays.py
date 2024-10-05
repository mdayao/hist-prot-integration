import numpy as np
import pandas as pd

import os
import glob
import tqdm

import argparse

parser = argparse.ArgumentParser(description='Combine H&E arrays per acq_id')
parser.add_argument('--input_dir', type=str, default='data/hnscc-atlas/he_patches_64')
parser.add_argument('--combined_dir', type=str, default='data/hnscc-atlas/he_patches_64/combined')
parser.add_argument('--acq_ids', type=str, default='data/hnscc-atlas/HNSCC_Atlas_acq_ids.csv')
parser.add_argument('--kpmp', action='store_true', help='Use KPMP data')

args = parser.parse_args()

# Create the output directory if it doesn't exist
os.makedirs(args.combined_dir, exist_ok=True)

if args.kpmp:
    metadata = pd.read_csv(args.acq_ids)
    metadata_filtered = metadata.groupby('Participant ID').first().reset_index()
    acq_ids = [file_name.split('.')[0] for file_name in metadata_filtered['File Name'].values]
else:
    acq_ids = list(pd.read_csv(args.acq_ids, header=None).iloc[:,0].values)

# Loop through each acq_id in acq_ids and combine all the H&E arrays into a single array
for acq_id in tqdm.tqdm(acq_ids, desc=f'Combining arrays'):
    # Get a list of all the H&E arrays for the given acq_id
    if args.kpmp:
        he_arrays = [np.load(cell_patch_path) for cell_patch_path in sorted(glob.glob(f"{args.input_dir}/{acq_id}/{acq_id}_*.npy"))]
    else:
        he_arrays = [np.load(cell_patch_path) for cell_patch_path in sorted(glob.glob(f"{args.input_dir}/{acq_id}_*.npy"))]
    # Combine the H&E arrays into a single array
    if len(he_arrays) == 0:
        print('No H&E arrays found for acq_id:', acq_id)
        continue
    combined_he_array = np.stack(he_arrays, axis=0)
    # Save the combined H&E array
    np.save(f'{args.combined_dir}/{acq_id}.npy', combined_he_array)


