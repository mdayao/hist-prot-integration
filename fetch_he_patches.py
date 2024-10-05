import numpy as np
import pandas as pd

import os
os.environ["STAGE"]="prod"
import emdatabase as emdb

import tqdm
import sys

import argparse

def get_nucl_seg_mask_single_cell(mask_arr, coords, cell_id, window_size):
    x, y = coords
    # the cell ids in this study are different from the segmentation mask ids, so this is a workaround
    seg_id = mask_arr[y,x] 
    single_mask = (mask_arr==seg_id)
    single_mask_cropped = single_mask[y-(window_size//2):y+(window_size//2), x-(window_size//2):x+(window_size//2)]

    return single_mask_cropped

def get_visual_quality_for_acquisition_id(acquisition_id, conn):
    """Given an acquisition id, returns visual quality boolean

    Args:
        acquisition_id (str): acquisition id
        conn (SnowflakeConnection/sqlalchemy.engine): snowflake/postgres connector

    Returns
        bool: visual quality 
    """
    sql_query = """
        select VISUAL_QUALITY from PORTAL.STUDY_REGIONS
        where ACQUISITION_ID = :acquisition_id
        """
    query_params = {"acquisition_id": str(acquisition_id)}
    res = emdb.query(sql_query, query_params, conn, add_datestamp=True)

    sample_id = res['VISUAL_QUALITY']
    assert len(sample_id) == 1
    return sample_id.iloc[0]

parser = argparse.ArgumentParser(description='Fetch H&E patches')
parser.add_argument('--dataset', type=str, choices=['hnscc-atlas', 'renal-transplant', 'charville', 'transplant-stanford'], default='hnscc-atlas', help='Dataset to fetch patches from')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--coords', type=str, default="./data/hnscc-atlas/xy_data.csv", help='Path to coordinates file')
parser.add_argument('--masks', action='store_true', help='Save masks instead of H&E patches')
parser.add_argument('--save-labels-only', action='store_true', help='Save cell type labels only')
args = parser.parse_args()

conn = emdb.connect()
patch_size = args.patch_size
half_patch = int(patch_size // 2)

if args.dataset == 'hnscc-atlas':
    study_id_he = 317 # HNSCC Atlas H&E
    study_id = 258 # HNSCC_Atlas
    seg_version=1
    
    acq_ids_he = sorted(emdb.get_acquisition_ids_for_study_id(study_id_he, conn))
    acq_ids = sorted(emdb.get_acquisition_ids_for_study_id(study_id, conn))
    xy_df = pd.read_csv(args.coords) # default are x,y coordinates for HNSCC_Atlas (not H&E)
    no_seg_cell_ids = []
    no_seg_acq_ids = []
    
    for i, acq_id in enumerate(acq_ids):
        print(f'coord acq id: {acq_id}')
        print(f'he acq id: {acq_ids_he[i]}')
    
        if not args.masks:
            he_channels = [emdb.get_channel_im(acq_ids_he[i], biomarker=f'H&E ({x})') for x in ['Red', 'Green', 'Blue']]
            he_im = np.stack(he_channels, axis=2)
            he_im = (he_im / he_im.max())*256
            he_im = he_im.astype(np.uint8)
        else:
            # Get segmentation mask from HNSCC_Atlas (not H&E)
            he_im = emdb.get_seg_mask(acq_ids_he[i], segmentation_version=seg_version, mask='nucleus')
        
        sub_df = xy_df[xy_df.ACQUISITION_ID == acq_id]
        
        for cell_id in tqdm.tqdm(sub_df.CELL_ID.values, total=len(sub_df.CELL_ID.values), desc=f"Saving patches for {acq_id}"):
            coords = sub_df[sub_df.CELL_ID == cell_id][['X','Y']].values[0]
            if not args.masks:
                # Getting the H&E patch using coordinates from HNSCC_Atlas but H&E images from HNSCC Atlas H&E
                cell_im = he_im[coords[1]-half_patch:coords[1]+half_patch, coords[0]-half_patch:coords[0]+half_patch]
                np.save(f"./data/hnscc-atlas/he_patches_{patch_size}/{acq_id}_{cell_id}.npy", cell_im)
            else:
                cell_im = get_nucl_seg_mask_single_cell(he_im, coords, cell_id, patch_size)
                if np.sum(cell_im) == 0:
                    no_seg_cell_ids.append(cell_id)
                    no_seg_acq_ids.append(acq_id)
                    continue
                cell_im = cell_im.astype(np.uint8)
                np.save(f"./data/hnscc-atlas/seg_mask_nucl_{patch_size}/{acq_id}_{cell_id}.npy", cell_im)
    
    
    if args.masks:
        no_seg_df = pd.DataFrame({'ACQUISITION_ID': no_seg_acq_ids, 'CELL_ID': no_seg_cell_ids})
        no_seg_df.to_csv('./data/hnscc-atlas/no_seg_mask.csv', index=False)

elif args.dataset == 'renal-transplant':
    study_id = 314 # Renal transplant
    seg_version=1
    all_acq_ids = emdb.get_acquisition_ids_for_study_id(study_id, conn)
    codex_acq_ids = [x for x in all_acq_ids if 's314' in x and 'HandE' not in x]
    
    paired_acq_ids = []
    for acq_id in codex_acq_ids:
        he_acq_id = emdb.get_overlayed_he_for_acquisition_id(acq_id, conn)
        visual_quality = get_visual_quality_for_acquisition_id(acq_id, conn)
        if he_acq_id is not None and visual_quality: # only use acq_ids with paired H&E and visual quality = True
            paired_acq_ids.append((acq_id, he_acq_id))

    xy_df = pd.read_csv(args.coords) 
    no_seg_cell_ids = []
    no_seg_acq_ids = []

    if args.save_labels_only:
        all_label_dfs = []

    for i, (acq_id, he_acq_id) in enumerate(paired_acq_ids):
        print(f'coord acq id: {acq_id}')
   
        if not args.save_labels_only:
            if not args.masks:
                he_im = emdb.get_he_image(he_acq_id, conn)
            else:
                # Get segmentation mask
                he_im = emdb.get_seg_mask(acq_id, segmentation_version=seg_version, mask='cell', biomarker_expression_version=1)
        
        sub_df = xy_df[xy_df.ACQUISITION_ID == acq_id]
       
        # only save patches for cells with labels according to Phenotype_v2 (id 3024)
        cell_labels_df = emdb.get_cell_classification_output_for_acquisition_id(acq_id, conn, annotation_id=3024)
        merged_df = sub_df.merge(cell_labels_df, how='left', on='CELL_ID')
        merged_df = merged_df.dropna() # drop cells without labels
        if len(merged_df) == 0:
            print('No cells with labels')
            continue
        if args.save_labels_only:
            cell_labels_df.rename(columns={'CELL_ID': 'cell_id', 'VALUE': 'celltype_id', 'ANNOTATION_LABEL': 'celltype_label'}, inplace=True)
            cell_labels_df = cell_labels_df.dropna()
            cell_labels_df['acquisition_id'] = acq_id
            all_label_dfs.append(cell_labels_df)
            continue
        for cell_id in tqdm.tqdm(merged_df.CELL_ID.values, total=len(merged_df.CELL_ID.values), desc=f"Saving patches for {acq_id}"):
            coords = merged_df[merged_df.CELL_ID == cell_id][['X','Y']].values[0]

            if not args.masks:
                cell_im = he_im[:, coords[1]-half_patch:coords[1]+half_patch, coords[0]-half_patch:coords[0]+half_patch]
                np.save(f"./data/renal-transplant/he_patches_{patch_size}/{acq_id}_{cell_id}.npy", cell_im)
            else:
                cell_im = get_nucl_seg_mask_single_cell(he_im, coords, cell_id, patch_size)
                if np.sum(cell_im) == 0:
                    no_seg_cell_ids.append(cell_id)
                    no_seg_acq_ids.append(acq_id)
                    continue
                cell_im = cell_im.astype(np.uint8)
                np.save(f"./data/renal-transplant/seg_mask_cell_{patch_size}/{acq_id}_{cell_id}.npy", cell_im)

    if args.save_labels_only:
        labels_df = pd.concat(all_label_dfs)
        labels_df = labels_df[['acquisition_id', 'cell_id', 'celltype_label', 'celltype_id']]
        labels_df.to_csv('./data/renal-transplant/renal-transplant_celltypes.csv', index=False)

    if args.masks:
        no_seg_df = pd.DataFrame({'ACQUISITION_ID': no_seg_acq_ids, 'CELL_ID': no_seg_cell_ids})
        no_seg_df.to_csv('./data/hnscc-atlas/no_seg_mask.csv', index=False)

elif args.dataset == 'charville':
    study_id = 1167 # charville aligned splits
    all_acq_ids = emdb.get_acquisition_ids_for_study_id(study_id, conn)
    region_labels = []
    for acq_id in all_acq_ids:
        region_labels.append(emdb.get_region_label_for_acquisition_id(acq_id, conn))
    subset_acq_id_idx = np.array([i for i, x in enumerate(region_labels) if '257' in x and 'reg 1' in x])
    subset_acq_ids = np.array(all_acq_ids)[subset_acq_id_idx]

    xy_df = pd.read_csv(args.coords)

    if args.save_labels_only:
        all_label_dfs = []

    for acq_id in subset_acq_ids:
        visual_quality = get_visual_quality_for_acquisition_id(acq_id, conn)
        if not visual_quality:
            continue # skip acq_ids with visual quality = False

        if not args.save_labels_only:
            try:
                he_im = emdb.get_he_image(acq_id, conn)
            except:
                print(f'No H&E image for {acq_id}')
                continue

        sub_df = xy_df[xy_df.ACQUISITION_ID == acq_id]

        cell_labels_df = emdb.get_cell_classification_output_for_acquisition_id(acq_id, conn, annotation_id=4342)
        merged_df = sub_df.merge(cell_labels_df, how='left', on='CELL_ID')
        merged_df = merged_df.dropna() # drop cells without labels
        if len(merged_df) == 0:
            print('No cells with labels')
            continue
        if args.save_labels_only:
            cell_labels_df.rename(columns={'CELL_ID': 'cell_id', 'VALUE': 'celltype_id', 'ANNOTATION_LABEL': 'celltype_label'}, inplace=True)
            cell_labels_df = cell_labels_df.dropna()
            cell_labels_df['acquisition_id'] = acq_id
            all_label_dfs.append(cell_labels_df)
            continue

        for cell_id in tqdm.tqdm(merged_df.CELL_ID.values, total=len(merged_df.CELL_ID.values), desc=f"Saving patches for {acq_id}"):
            coords = merged_df[merged_df.CELL_ID == cell_id][['X','Y']].values[0]

            cell_im = he_im[:, coords[1]-half_patch:coords[1]+half_patch, coords[0]-half_patch:coords[0]+half_patch]
            np.save(f"./data/charville/he_patches_{patch_size}/{acq_id}_{cell_id}.npy", cell_im)

    if args.save_labels_only:
        labels_df = pd.concat(all_label_dfs)
        labels_df = labels_df[['acquisition_id', 'cell_id', 'celltype_label', 'celltype_id']]
        labels_df.to_csv('./data/charville/charville_celltypes.csv', index=False)

elif args.dataset == 'transplant-stanford':
    study_id = 515
    all_acq_ids = list(pd.read_csv('./data/transplant-stanford/transplant-stanford_acq_ids.csv', header=None).iloc[:,0].values)
    xy_df = pd.read_csv(args.coords)

    if args.save_labels_only:
        all_label_dfs = []

    for acq_id in all_acq_ids:
        sub_df = xy_df[xy_df.ACQUISITION_ID == acq_id]
        
        if not args.save_labels_only:
            try:
                he_im = emdb.get_he_image(acq_id, conn)
            except:
                print(f'No H&E image for {acq_id}')
                continue

        cell_labels_df = emdb.get_cell_classification_output_for_acquisition_id(acq_id, conn, annotation_id=11042)
        merged_df = sub_df.merge(cell_labels_df, how='left', on='CELL_ID')
        merged_df = merged_df.dropna() # drop cells without labels
        if len(merged_df) == 0:
            print('No cells with labels')
            continue
        if args.save_labels_only:
            cell_labels_df.rename(columns={'CELL_ID': 'cell_id', 'VALUE': 'celltype_id', 'ANNOTATION_LABEL': 'celltype_label'}, inplace=True)
            cell_labels_df = cell_labels_df.dropna()
            cell_labels_df['acquisition_id'] = acq_id
            all_label_dfs.append(cell_labels_df)
            continue

        for cell_id in tqdm.tqdm(merged_df.CELL_ID.values, total=len(merged_df.CELL_ID.values), desc=f"Saving patches for {acq_id}"):
            coords = merged_df[merged_df.CELL_ID == cell_id][['X','Y']].values[0]

            cell_im = he_im[:, coords[1]-half_patch:coords[1]+half_patch, coords[0]-half_patch:coords[0]+half_patch]
            np.save(f"./data/transplant-stanford/he_patches_{patch_size}/{acq_id}_{cell_id}.npy", cell_im)

    if args.save_labels_only:
        labels_df = pd.concat(all_label_dfs)
        labels_df = labels_df[['acquisition_id', 'cell_id', 'celltype_label', 'celltype_id']]
        labels_df.to_csv('./data/transplant-stanford/transplant-stanford_celltypes.csv', index=False)


        

