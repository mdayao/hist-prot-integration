import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-embeds', type=str, help='Path to the training embeddings', required=True)
    parser.add_argument('--train-ids', type=str, help='Path to the training sample ids', required=True)
    parser.add_argument('--train-labels', type=str, help='Path to the training labels', required=True)
    parser.add_argument('--train-acq-ids', type=str, help='Path to the training acquisition ids', required=True)

    parser.add_argument('--test-embeds', type=str, help='Path to the test embeddings. If not provided, will use the training embeddings')
    parser.add_argument('--test-ids', type=str, help='Path to the test sample ids. If not provided, will use the training sample ids')
    parser.add_argument('--test-labels', type=str, help='Path to the test labels. If not provided, will use the training labels')
    parser.add_argument('--test-acq-ids', type=str, help='Path to the test acquisition ids. Not required if --no-test-labels is set.')

    parser.add_argument('--no-test-labels', action='store_true', help='If provided, will only predict on the test set and not evaluate the predictions.')

    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility', default=0)

    parser.add_argument('--output', type=str, help='Path to the output file', required=True)

    parser.add_argument('--weighted', action='store_true', help='If provided, will use class weights in the logistic regression model')

    args = parser.parse_args()

    train_embeds = np.load(args.train_embeds)
    train_ids = pd.read_csv(args.train_ids)
    train_label_df = pd.read_csv(args.train_labels)
    train_acq_ids = list(pd.read_csv(args.train_acq_ids, header=None).iloc[:,0].values)

    if args.test_embeds is None:
        test_embeds = train_embeds.copy()
        test_ids = train_ids.copy()
        test_label_df = train_label_df.copy()
    else:
        test_embeds = np.load(args.test_embeds)
        test_ids = pd.read_csv(args.test_ids)
        if not args.no_test_labels:
            test_label_df = pd.read_csv(args.test_labels)

    train_label_df = train_label_df.loc[train_label_df['acquisition_id'].isin(train_acq_ids)]
    train_label_df = train_label_df.merge(train_ids, left_on=['acquisition_id', 'cell_id'], right_on=['acq_id', 'cell_id'], how='right')
    train_embeds = train_embeds[~train_label_df['celltype_id'].isna(),:]
    train_labels, _ = pd.factorize(train_label_df.loc[~train_label_df['celltype_id'].isna(), 'celltype_id'], sort=True)
    assert train_embeds.shape[0] == len(train_labels)

    if not args.no_test_labels:
        test_acq_ids = list(pd.read_csv(args.test_acq_ids, header=None).iloc[:,0].values)
        test_label_df = test_label_df.loc[test_label_df['acquisition_id'].isin(test_acq_ids)]
        test_label_df = test_label_df.merge(test_ids, left_on=['acquisition_id', 'cell_id'], right_on=['acq_id', 'cell_id'], how='right')
        test_embeds = test_embeds[~test_label_df['celltype_id'].isna(),:]
        test_labels, _ = pd.factorize(test_label_df.loc[~test_label_df['celltype_id'].isna(), 'celltype_id'], sort=True)
        assert test_embeds.shape[0] == len(test_labels)

    celltype_names = pd.value_counts(train_label_df.sort_values('celltype_id').celltype_label, sort=False).index

    clf = LogisticRegression(random_state=args.random_seed, max_iter=500, class_weight='balanced' if args.weighted else None).fit(train_embeds, train_labels)

    test_preds = clf.predict(test_embeds)

    if not args.no_test_labels:
        cm = confusion_matrix(test_labels, test_preds)
        cr = classification_report(test_labels, test_preds, target_names=celltype_names)

        # save classification report to output file
        with open(args.output, 'w') as f:
            f.write(cr)

        print(cr)

        # also save predictions to output csv
        test_results_df = test_label_df.loc[~test_label_df['celltype_id'].isna(), ['acquisition_id', 'cell_id', 'celltype_id']]
        test_results_df['pred_celltype_id'] = test_preds
        # add column with celltype names corresponding to the predicted celltype ids
        test_results_df['pred_celltype_label'] = celltype_names[test_preds]
        test_results_df.to_csv(args.output.replace('.txt', '.csv'), index=False)

    else: # save predictions to output csv using pandas
        test_results_df = pd.DataFrame({'acquisition_id': test_ids['acq_id'], 'cell_id': test_ids['cell_id'], 'pred_celltype_id': test_preds})
        # add column with celltype names corresponding to the predicted celltype ids
        test_results_df['pred_celltype_label'] = celltype_names[test_preds]
        test_results_df.to_csv(args.output, index=False)




