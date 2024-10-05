import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import f1_score

import argparse

def naive_predictor(proportions_array, num_samples, seed=32498):
    rng = np.random.default_rng(seed)
    # Generate random predictions based on proportions
    predictions = rng.choice(len(proportions_array), size=num_samples, p=proportions_array)
    return predictions

parser = argparse.ArgumentParser(description='Plot F1 scores by cell type')
parser.add_argument('--train_acq_ids_path', type=str, default="./data/renal-transplant/renal-transplant_acq_ids_train.csv")
parser.add_argument('--train_label_path', type=str, default='./data/renal-transplant/renal-transplant_overlap_transplant-stanford_celltypes.csv')

parser.add_argument('--val_acq_ids_path', type=str, default="./data/renal-transplant/renal-transplant_acq_ids_val.csv")
parser.add_argument('--val_label_path', type=str, default='./data/renal-transplant/renal-transplant_overlap_transplant-stanford_celltypes.csv')

parser.add_argument('--joint_preds_path', type=str, default='./data/renal-transplant/predictions/joint-overlap-celltypes-164-2-165-epoch-125_renal-transplant_val.csv')
parser.add_argument('--hne_cnn_preds_path', type=str, default='./data/renal-transplant/predictions/baseline-overlap-celltypes-163_epoch-300_renal-transplant_val.csv')
parser.add_argument('--foundation_preds_path', type=str, default='./data/renal-transplant/predictions/foundation/renal-transplant_overlap_phikon-large-fov-rescaled.csv')

parser.add_argument('--output_path', type=str, default='/Users/monica/Documents/research/he-codex-morphology/figs/f1-scores/f1_scores_by_celltype.pdf')

parser.add_argument('--weird_labels', action='store_true', help='Use this flag if the foundation prediction labels dont match the format of its own ground truth labels.')
parser.add_argument('--y-axis-max', type=float, default=0.8)
parser.add_argument('--fig-width', type=float, default=10)
parser.add_argument('--fig-height', type=float, default=6)

args = parser.parse_args()

train_acq_ids_path = args.train_acq_ids_path
train_label_path = args.train_label_path
val_acq_ids_path = args.val_acq_ids_path
val_label_path = args.val_label_path
joint_preds_path = args.joint_preds_path
hne_cnn_preds_path = args.hne_cnn_preds_path
foundation_preds_path = args.foundation_preds_path

# Get training data proportions
train_acq_ids = list(pd.read_csv(train_acq_ids_path, header=None).iloc[:,0].values)
train_label_df = pd.read_csv(train_label_path)
train_new_ids, _ = pd.factorize(train_label_df['celltype_id'])
train_label_df['celltype_id'] = train_new_ids
train_df = train_label_df[train_label_df.acquisition_id.isin(train_acq_ids)]

label_mapping = train_df[['celltype_id', 'celltype_label']].drop_duplicates()
label_mapping = label_mapping.set_index('celltype_id')
sorted_labels = label_mapping.sort_values(by='celltype_id')['celltype_label'].values
sorted_labels = list(sorted_labels)
# reindex label_mapping
label_mapping = label_mapping.reset_index()

train_proportions = np.array(pd.value_counts(train_df.celltype_id).sort_index() / len(train_df))

joint_preds_df = pd.read_csv(joint_preds_path)
hne_cnn_preds_df = pd.read_csv(hne_cnn_preds_path)
foundation_preds_df = pd.read_csv(foundation_preds_path)

# map the foundation ids to the same celltype ids as the other predictions
merged_preds = foundation_preds_df.merge(joint_preds_df, on=['acquisition_id', 'cell_id'], suffixes=('_foundation', '_joint'), how='inner')
preds_mapping = merged_preds[['celltype_id_foundation', 'celltype_id_joint']].drop_duplicates()
preds_mapping = preds_mapping.set_index('celltype_id_foundation')
foundation_preds_df['celltype_id'] = foundation_preds_df['celltype_id'].map(preds_mapping['celltype_id_joint'])
if not args.weird_labels:
    foundation_preds_df['pred_celltype_id'] = foundation_preds_df['pred_celltype_id'].map(preds_mapping['celltype_id_joint'])
foundation_preds_df = foundation_preds_df.rename(columns={'pred_celltype_id': 'predictions'})

# Get validation labels
val_acq_ids = list(pd.read_csv(val_acq_ids_path, header=None).iloc[:,0].values)
val_label_df = pd.read_csv(val_label_path)
val_new_ids, _ = pd.factorize(val_label_df['celltype_id'])
val_label_df['celltype_id'] = val_new_ids
val_df = val_label_df[val_label_df.acquisition_id.isin(val_acq_ids)]

if args.weird_labels:
    weird_merged_preds = foundation_preds_df.merge(val_df, on=['acquisition_id', 'cell_id'], how='inner', suffixes=('_foundation', '_val'))
    celltype_label_id_mapping = weird_merged_preds[['celltype_label', 'celltype_id_val']].drop_duplicates()
    celltype_label_id_mapping = celltype_label_id_mapping.set_index('celltype_label')
    foundation_preds_df['predictions'] = foundation_preds_df['pred_celltype_label'].map(celltype_label_id_mapping['celltype_id_val'])

# get naive predictions on validation set
val_preds = naive_predictor(train_proportions, len(val_df))

# Get F1 scores
foundation_f1 = f1_score(foundation_preds_df.celltype_id, foundation_preds_df.predictions, average=None)
foundation_f1_weighted = f1_score(foundation_preds_df.celltype_id, foundation_preds_df.predictions, average='weighted')
joint_f1 = f1_score(joint_preds_df.celltype_id, joint_preds_df.predictions, average=None)
joint_f1_weighted = f1_score(joint_preds_df.celltype_id, joint_preds_df.predictions, average='weighted')
hne_cnn_f1 = f1_score(hne_cnn_preds_df.celltype_id, hne_cnn_preds_df.predictions, average=None)
hne_cnn_f1_weighted = f1_score(hne_cnn_preds_df.celltype_id, hne_cnn_preds_df.predictions, average='weighted')
naive_f1 = f1_score(val_df.celltype_id, val_preds, average=None)
naive_f1_weighted = f1_score(val_df.celltype_id, val_preds, average='weighted')

print("Foundation F1 weighted: ", foundation_f1_weighted)
print("Joint F1 weighted: ", joint_f1_weighted)
print("H&E CNN F1 weighted: ", hne_cnn_f1_weighted)
print("Naive F1 weighted: ", naive_f1_weighted)

# Plot the F1 scores in a bar plot
f1_scores = np.array([foundation_f1, joint_f1, hne_cnn_f1, naive_f1])
f1_scores_weighted = np.array([foundation_f1_weighted, joint_f1_weighted, hne_cnn_f1_weighted, naive_f1_weighted])

# use seaborn palette
sns.set_theme(style="whitegrid")
sns.set_palette('colorblind')

# x axis should be each predicted class, different colored bars for each model. one of the x axis should be the weighted f1 scores
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(args.fig_width, args.fig_height), sharex=True, height_ratios=[4, 1])

bar_width = 0.2
index = np.arange(len(sorted_labels) + 1)

# alphabetically sort labels
sorted_labels_alpha = sorted(sorted_labels)
foundation_f1_alpha = [foundation_f1[sorted_labels.index(label)] for label in sorted_labels_alpha]
joint_f1_alpha = [joint_f1[sorted_labels.index(label)] for label in sorted_labels_alpha]
hne_cnn_f1_alpha = [hne_cnn_f1[sorted_labels.index(label)] for label in sorted_labels_alpha]
naive_f1_alpha = [naive_f1[sorted_labels.index(label)] for label in sorted_labels_alpha]

#bar1 = ax1.bar(index, [foundation_f1_weighted] + list(foundation_f1), bar_width, label='Foundation')
#bar2 = ax1.bar(index + bar_width, [joint_f1_weighted] + list(joint_f1), bar_width, label='Contrastive')
#bar3 = ax1.bar(index + 2*bar_width, [hne_cnn_f1_weighted] + list(hne_cnn_f1), bar_width, label='H&E CNN')
#bar4 = ax1.bar(index + 3*bar_width, [naive_f1_weighted] + list(naive_f1), bar_width, label='Naive')
bar1 = ax1.bar(index, [foundation_f1_weighted] + foundation_f1_alpha, bar_width, label='Foundation')
bar2 = ax1.bar(index + bar_width, [joint_f1_weighted] + joint_f1_alpha, bar_width, label='Joint')
bar3 = ax1.bar(index + 2*bar_width, [hne_cnn_f1_weighted] + hne_cnn_f1_alpha, bar_width, label='H&E CNN')
bar4 = ax1.bar(index + 3*bar_width, [naive_f1_weighted] + naive_f1_alpha, bar_width, label='Naive')

ax1.legend()

all_bars = [bar1, bar2, bar3, bar4]

for bar in all_bars:
    bar[0].set_hatch('///')
    bar[0].set_edgecolor('black')

ax1.set_ylabel('F1 Score')
ax1.set_ylim([0, args.y_axis_max])
ax1.set_xticks([r + 1.5*bar_width for r in range(len(sorted_labels_alpha)+1)])

#ax.axvline(x=0.8, color='black', linestyle='--')

num_cells = pd.value_counts(train_df.celltype_id).sort_index().values
num_cells_alpha = [list(num_cells)[sorted_labels.index(label)] for label in sorted_labels_alpha]

# Plot the number of cells as bars beneath the F1 scores
cell_bar_width = bar_width * 4  # Adjust the width of the number of cells bars

# Plot the number of cells (centered under each cell type)
ax2.bar(index[1:] + 1.5 * bar_width , num_cells_alpha, cell_bar_width, color='gray', alpha=1)

# Set the y-axis label for the number of cells
ax2.set_ylabel('# Training Samples')
ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=4))  # Adjust nbins as needed

# Adjust the secondary axis limits if needed to match the data range
ax2.set_ylim([0, max(num_cells) * 1.1])

ax2.set_xlabel('Cell Type')
ylabels = [f'{int(y):0d}' + 'k' for y in ax2.get_yticks()//1000]
ax2.set_yticklabels(ylabels)
ax2.set_xticklabels(['Weighted Average'] + list(sorted_labels_alpha), rotation=45, ha='right')

plt.tight_layout()
#plt.show()

fig.savefig(args.output_path, bbox_inches='tight')
fig.savefig(args.output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)

plt.close()




