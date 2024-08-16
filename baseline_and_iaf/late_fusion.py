import os, sys
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from config import TASKS, PREDICTION_FOLDER, NUM_TARGETS, HUMOR, PERCEPTION_LABELS, PERCEPTION, LOG_FOLDER
from eval import mean_pearsons, calc_pearsons
from main import get_eval_fn


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=TASKS)
    parser.add_argument('--label_dim', choices=PERCEPTION_LABELS, required=False,
                        help=f'Specify the emotion dimension, only relevant for perception.')
    parser.add_argument('--model_ids', nargs='+', required=True, help='model ids')
    parser.add_argument('--seeds', nargs='+', required=False, help=f'seeds')
    parser.add_argument('--result_csv', required=False, type=str)
    parser.add_argument('--lf_dir', default='lf', help=f'Specify directory name where to save prediction files. Default: lf')
    parser.add_argument('--submission_format', action='store_true',
                        help=f'Specify whether to save predictions in submission format (default: False).')
    
    
    args = parser.parse_args()
    # TODO add again
    #assert len(set(args.model_ids)) == len(args.model_ids), "Error, duplicate model file"
    #assert len(args.model_ids) >= 2, "For late fusion, please give at least 2 different models"

    if args.task == PERCEPTION:
        assert args.label_dim

    if args.seeds and len(args.seeds) == 1:
        args.seeds = [args.seeds[0]] * len(args.model_ids)
        assert len(args.model_ids) == len(args.seeds)

    if args.task == HUMOR:
        args.prediction_dirs = [os.path.join(PREDICTION_FOLDER, args.task, args.model_ids[i]) for i in
                                range(len(args.model_ids))] # , args.seeds[i] # gp: pred is from best seed, main.py does not save seed number in log dir
    elif args.task == PERCEPTION:
        args.prediction_dirs = [os.path.join(PREDICTION_FOLDER, args.task, args.label_dim, args.model_ids[i]) for i in
                                range(len(args.model_ids))] # , args.seeds[i]
    if not args.result_csv is None:
        args.result_csv = os.path.join(LOG_FOLDER, 'lf_results', args.task if args.task==HUMOR else f'{args.task}/{args.label_dim}', args.result_csv)
        os.makedirs(os.path.dirname(args.result_csv), exist_ok=True)
        if not args.result_csv.endswith('.csv'):
            args.result_csv += '.csv'
    return args

def create_humor_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith('prediction_')]].values
    if weights is None:
        # TODO auto-compute weights based on performance
        labels = df['label'].values
        eval_fn,_ = get_eval_fn(HUMOR)
        weights = []
        for i in range(pred_arr.shape[1]):
            preds = pred_arr[:,i]
            # 0.5 chance
            weights.append(max(eval_fn(preds, labels) - 0.5, 0))
        print('Weights', weights)
        if all(w == 0 for w in weights):
            print('Only zeros')
            weights = [1 / len(weights)] * len(weights)
        #weights = [1.] * pred_arr.shape[1]
    for i, w in enumerate(weights):
        preds = pred_arr[:, i]
        # normalise and weight
        # preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
        preds = w * preds
        pred_arr[:, i] = preds
    fused_preds = np.sum(pred_arr, axis=1)
    labels = df['label'].values
    return fused_preds, labels, weights


def create_perception_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith('prediction_')]].values
    if weights is None:
        # auto-compute weights based on performance
        labels = df['label'].values
        eval_fn,_ = get_eval_fn(PERCEPTION)
        weights = []
        for i in range(pred_arr.shape[1]):
            preds = pred_arr[:,i]
            weights.append(max(eval_fn(preds, labels), 0))
        #print('Weights', weights)
        #weights = [1.] * pred_arr.shape[1]
        if all(w==0 for w in weights):
            print('Only zeros')
            weights = [1/len(weights)] * len(weights)
    weights = np.array(weights) / np.sum(weights)
    for i, w in enumerate(weights.tolist()):
        preds = pred_arr[:, i]
        preds = w * preds
        pred_arr[:, i] = preds
    fused_preds = np.sum(pred_arr, axis=1)
    labels = df['label'].values
    return fused_preds, labels, weights


if __name__ == '__main__':
    args = parse_args()
    ress = []
    weights = None # gets set to devel weights at first call
    for partition in ['devel', 'test']:
        dfs = [pd.read_csv(os.path.join(pred_dir, f'predictions_{partition}.csv')) for pred_dir in args.prediction_dirs]

        meta_cols = [c for c in list(dfs[0].columns) if c.startswith('meta_')]
        for meta_col in meta_cols:
            assert all(np.all(df[meta_col].values == dfs[0][meta_col].values) for df in dfs)
        meta_df = dfs[0][meta_cols].copy()

        label_cols = [c for c in list(dfs[0].columns) if c.startswith('label')]
        if args.task == PERCEPTION and partition=='devel':
            for label_col in label_cols:
                #for df in dfs:
                #    print((df[label_col].values, dfs[0][label_col].values))
                assert all(np.all(df[label_col].values.round(4) == dfs[0][label_col].values.round(4)) for df in dfs)
        label_df = dfs[0][label_cols].copy()

        prediction_dfs = []
        for i, df in enumerate(dfs):
            pred_df = df.drop(columns=meta_cols + label_cols)
            pred_df.rename(columns={c: f'{c}_{args.model_ids[i]}' for c in pred_df.columns}, inplace=True)
            prediction_dfs.append(pred_df)
        prediction_df = pd.concat(prediction_dfs, axis='columns')

        full_df = pd.concat([meta_df, prediction_df, label_df], axis='columns')

        if args.task == HUMOR:
            preds, labels, weights = create_humor_lf(full_df, weights=weights)
        elif args.task == PERCEPTION:
            preds, labels, weights = create_perception_lf(full_df, weights=weights)

        eval_fn, eval_str = get_eval_fn(args.task)

        # replace NaN values with 0 from labels e.g. test. Othewise RaisedException from eval_fn
        labels[np.isnan(labels)] = 0
        result = np.round(eval_fn(preds, labels), 4)
        print(f'{partition}: {result} {eval_str}')
        ress.append(result)

        # save fusion predictions
        full_df['prediction'] = preds
        task_id = args.task if args.task != PERCEPTION else os.path.join(args.task, args.label_dim)
        _filename = os.path.join(PREDICTION_FOLDER, task_id, args.lf_dir, f'predictions_{partition}.csv')
        os.makedirs(os.path.dirname(_filename), exist_ok=True)
        
        # save in submission format
        if args.submission_format:
            full_df = full_df.loc[:, ~full_df.columns.str.startswith('prediction_')]
            # if partition == 'test':
            full_df = full_df.drop(columns=['label'])
        full_df.to_csv(_filename, index=False)
        print(f'Predictions saved to {_filename}')

    if not args.result_csv is None:
        df = pd.DataFrame({
            'models': [args.model_ids],
            'weights': [str(weights)],
            # 'seeds': [args.seeds],
            'devel': [ress[0]],
            'test': [ress[1]]
        })
        if os.path.exists(args.result_csv):
            old_df = pd.read_csv(args.result_csv)
            df = pd.concat([old_df, df], axis='rows').reset_index(drop=True)
        df.to_csv(args.result_csv, index=False)
        print(f'Fusion results saved to {args.result_csv}')
