import numpy as np

import os
import glob
import tqdm

import argparse

parser = argparse.ArgumentParser(description='Remove empty HE images')
parser.add_argument('--input_dir', type=str, default='data/hnscc-atlas/he_patches_64')
parser.add_argument('--filtered_dir', type=str, default='data/hnscc-atlas/he_patches_64/filtered_out')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size of the HE images')

args = parser.parse_args()

# loop over files in input_dir and read in the array using numpy
# if the shape of the array is not (64, 64, 3), then move the file to filtered_dir
for filename in tqdm.tqdm(glob.glob(os.path.join(args.input_dir, '*.npy'))):
    arr = np.load(filename)
    if arr.shape != (args.patch_size, args.patch_size, 3) and arr.shape != (3, args.patch_size, args.patch_size):
        os.rename(filename, os.path.join(args.filtered_dir, os.path.basename(filename)))
