import numpy as np
import pandas as pd

from stardist import export_imagej_rois, random_label_cmap
from stardist.models import StarDist2D
from csbdeep.data import Normalizer, normalize_mi_ma
import tifffile
from tifffile import imread
from histocartography.preprocessing.tissue_mask import get_tissue_mask
from skimage.measure import regionprops

import argparse
import os
import tqdm

class MyNormalizer(Normalizer):
    def __init__(self, mi, ma):
            self.mi, self.ma = mi, ma
    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)
    def after(*args, **kwargs):
        assert False
    @property
    def do_after(self):
        return False

def compute_tissue_bbox(img, sigma):
    """
    Compute the bounding box of the tissue in the image.

    Returns:
        props: list of region properties of the tissue regions
        max_areas: indices of the regions sorted by area in descending order
    """
    tissue_mask, single_tissue_mask = get_tissue_mask(img, sigma=sigma)
    props = regionprops(tissue_mask)

    areas = np.array([prop.area for prop in props])
    max_areas = np.argsort(-areas)

    return props, max_areas

def segment_im(img, model=None):
    # Load the StarDist model
    if model is None:
        model = StarDist2D.from_pretrained('2D_versatile_he')
    
    # Normalize the image
    mi, ma = np.percentile(img, [0.5,99.8])
    normalizer = MyNormalizer(mi, ma)
    
    # Segment the image
    min_overlap = 128
    context = 128
    block_size = 4096
    assert min_overlap + 2*context < block_size, "overlap too small"
    labels, polys = model.predict_instances_big(img, axes='YXC', block_size=block_size, min_overlap=min_overlap, context=context, normalizer=normalizer, n_tiles=(4,4,1))
    
    return labels, polys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment images using StarDist')
    parser.add_argument('--data-dir', type=str, help='Path to the directory containing the images to segment')
    parser.add_argument('--output-dir', type=str, help='Path to the directory where the segmented images will be saved')
    
    parser.add_argument('--file-metadata', type=str, help='Path to the file containing the metadata of the images to segment')

    parser.add_argument('--append', action='store_true', help='Skip images that have already been segmented')

    parser.add_argument('--patch-size', type=int, default=64, help='Size of the cell patches to save')
    args = parser.parse_args()

    patch_size = args.patch_size
    
    # Read in the metadata file
    metadata = pd.read_csv(args.file_metadata)
    
    # Choose only one image per patient. For patients with multiple images, choose the first one
    metadata_filtered = metadata.groupby('Participant ID').first().reset_index()
    file_names = metadata_filtered['File Name'].values

    model = StarDist2D.from_pretrained('2D_versatile_he')
    
    # Loop through the images and segment them
    for file_idx, file_name in enumerate(file_names):
        # Skip images that have already been segmented
        if args.append:
            if os.path.exists(os.path.join(args.output_dir, file_name.split('.')[0] + '_coords.csv')):
                print(f"Skipping {file_name}, already segmented")
                continue
        print()
        print(f"Processing image {file_name}")

        # Load the image
        with tifffile.TiffFile(os.path.join(args.data_dir, file_name)) as tif:
            # get level 2 of pyramid for tissue detection
            img = tif.series[0].levels[2].asarray()

            # get level 0 of pyramid for segmentation
            full_img = tif.series[0].levels[0].asarray()
            
        scale_factor = int(full_img.shape[0] / img.shape[0])
        print(f"image shape: {full_img.shape}, tissue detection image shape: {img.shape}, scale_factor: {scale_factor}")

        # Use tissue detection to crop the image
        props, max_areas = compute_tissue_bbox(img, sigma=20)
        found_tissue = False
        for i, max_i in enumerate(max_areas):
            which_bbox = i
            if i > 2:
                print(f"Could not find a suitable bbox for {file_name}")
                break
            if props[max_i].area > 10000:
                bbox = props[max_i].bbox
                bbox = [scale_factor * i for i in bbox] # scale the bbox to the full image
                crop_image = full_img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                if crop_image.shape[0] < 4096 or crop_image.shape[1] < 4096 or (crop_image.shape[0]/crop_image.shape[1] > 100) or (crop_image.shape[1]/crop_image.shape[0] > 100):
                    print(f"bbox {i} too small or oblong, shape {crop_image.shape}, trying next bbox")
                    continue
                print(f"bbox {i}, cropped to shape {crop_image.shape}")

                # Segment the image
                labels, polys = segment_im(crop_image, model)

                # check how many cells were found
                if len(pd.unique(labels.flatten())) < 5000:
                    print(f"Found only {len(pd.unique(labels.flatten()))} cells, trying next bbox")
                else:
                    print(f"Found {len(pd.unique(labels.flatten()))} cells")
                    found_tissue = True
                    break

        if found_tissue:
            # Save the bounding box
            save_file_name = file_name.split('.')[0] + f'_{which_bbox}_bbox.csv'
            bbox_df = pd.DataFrame([bbox], columns=['ymin', 'xmin', 'ymax', 'xmax'])
            bbox_df.to_csv(os.path.join(args.output_dir, save_file_name), index=False)

            # Save coordinates of the cells in dataframe, csv saved after saving patches
            coords_df = pd.DataFrame(polys['points'], columns=['y', 'x'])
            coords_df['cell_id'] = range(len(polys['points']))

            # Save image patches
            try:
                for i, coords in tqdm.tqdm(enumerate(polys['points']), leave=False):
                    y, x = coords
                    y, x = int(y), int(x)
                    # check if patch is out of bounds
                    if y-patch_size//2 < 0 or y+patch_size//2 > crop_image.shape[0] or x-patch_size//2 < 0 or x+patch_size//2 > crop_image.shape[1]:
                        # remove the cell from the coords_df
                        coords_df = coords_df[coords_df['cell_id'] != i]
                        continue
                    patch = crop_image[int(y-patch_size//2):int(y+patch_size//2), int(x-patch_size//2):int(x+patch_size//2)]

                    save_file_name = file_name.split('.')[0] + f'_{i}.npy'
                    # make patches dir if it doesn't exist
                    if not os.path.exists(os.path.join(args.output_dir, f'patches_{patch_size}', file_name.split('.')[0])):
                        os.makedirs(os.path.join(args.output_dir, f'patches_{patch_size}', file_name.split('.')[0]))
                    np.save(os.path.join(args.output_dir, f'patches_{patch_size}', file_name.split('.')[0], save_file_name), patch.astype(np.uint8))

                # Save the coords csv
                coords_df.to_csv(os.path.join(args.output_dir, file_name.split('.')[0] + '_coords.csv'), index=False)
            except Exception as e:
                print(f"Error saving patches for {file_name}, skipping. Error below:")
                print(e)

                # save file name to list of failed files
                with open(os.path.join(args.output_dir, 'failed_files.txt'), 'a') as f:
                    f.write(file_name + '\n')

                

