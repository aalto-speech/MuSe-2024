import argparse
import os
import random
import sys
from datetime import datetime

import numpy
import torch
from dateutil import tz
from torch import nn

import config
from config import TASKS, PERCEPTION, HUMOR
from data_parser import load_data
from dataset import MultiModalMuSeDataset, MuSeDataset, custom_collate_fn
from eval import evaluate, calc_auc, calc_pearsons
from train import train_model
from utils import Logger, seed_worker, log_results

def parse_args():

    parser = argparse.ArgumentParser(description='MuSe 2024.')

    parser.add_argument('--task', type=str, required=True, choices=TASKS,
                        help=f'Specify the task from {TASKS}.')
    parser.add_argument('--feature', default='egemaps --normalize',
                        help='Specify the features used: 1 for unimodal, 3 (a,v,t) for multimodal.')
    parser.add_argument('--label_dim', default="assertiv", choices=config.PERCEPTION_LABELS)
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--feature_length', type=int, default=None,
                        help='Specify the number of seconds at the begining (positive) or end (negative) of the feature array i.e. 1 or -1 for 1 second.')
    parser.add_argument('--model_type', default='rnn',
                        help='Specify type of model to use, (default: RNN).')
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--linear_dropout', type=float, default=0.5)
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help='Specify the kernel_size for conv1d_block (default: 3).')
    parser.add_argument('--n_attn_head', type=int, default=1, 
                        help='Specify the number of heads for attention (default: 1).')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False). (Deprecated. replaced with model.to_device)')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='Specify whether to save model check points (default: False).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')
    parser.add_argument('--combine_train_dev', action='store_true',
                        help='Specify whether to combine train and dev dataset (default: False).')
    
    # evaluation only arguments
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--eval_seed', type=str, default=None,
                        help='Specify seed to be evaluated; only considered when --eval_model is given.')

    args = parser.parse_args()
    #if not (args.result_csv is None) and args.predict:
    #    print("--result_csv is not compatible with --predict")
    #    sys.exit(-1)
    if args.eval_model:
        assert args.eval_seed
    return args


def get_loss_fn(task):
    if task == HUMOR:
        return nn.BCELoss(), 'Binary Crossentropy'
    elif task == PERCEPTION:
        return nn.MSELoss(reduction='mean'), 'MSE'


def get_eval_fn(task):
    if task == PERCEPTION:
        return calc_pearsons, 'Pearson'
    elif task == HUMOR:
        return calc_auc, 'AUC'


def main(args):
    # ensure reproducibility
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # emo_dim only relevant for stress/personalisation
    args.label_dim = args.label_dim if args.task==PERCEPTION else ''
    print('Loading data ...')
    args.paths['partition'] = os.path.join(config.PATH_TO_METADATA[args.task], f'partition.csv')

    #data input 
    collate_fn=custom_collate_fn
    if args.model_type.lower() in ['lmf', 'tfn', 'iaf', 'ltf', 'laf']:
        print('prepare multi-modal data input ')
        data={}
        print('args.feature', args.feature)
        audio_feature = args.feature.split()[0] # 'w2v-msp' # 
        video_feature = args.feature.split()[1] # 'vit-fer' #
        text_feature = args.feature.split()[2] # 'bert-base-uncased' # 
        data['audio'] = load_data(args.task, args.paths, audio_feature, args.label_dim, args.normalize, feature_length=args.feature_length, save=args.cache)
        data['video'] = load_data(args.task, args.paths, video_feature, args.label_dim, args.normalize, feature_length=args.feature_length, save=args.cache)
        data['text'] = load_data(args.task, args.paths, text_feature, args.label_dim, args.normalize, feature_length=args.feature_length, save=args.cache)
        datasets = {partition:MultiModalMuSeDataset(data, partition) for partition in data['audio'].keys()}
        args.d_in = datasets['train'].get_feature_dim() # (d_in_a, d_in_v, d_in_t)
    else:
        data = load_data(args.task, args.paths, args.feature, args.label_dim, args.normalize, feature_length=args.feature_length, save=args.cache)
        datasets = {partition:MuSeDataset(data, partition) for partition in data.keys()}
        args.d_in = datasets['train'].get_feature_dim()

    args.n_targets = config.NUM_TARGETS[args.task]
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)

    # model type
    if args.model_type.lower() == 'rnn':
        print('Use rnn (default).')
        from model import Model
        model = Model(args)
    if args.model_type.lower() == 'cnn':
        print('Use cnn.')
        from model import CnnModel  
        model = CnnModel(args)
    if args.model_type.lower() == 'crnn':
        print('Use crnn.')
        from model import CrnnModel
        model = CrnnModel(args)
    if args.model_type.lower() == 'cnn-attn':
        print('Use self attention (with cnn).')
        from model import CnnAttnModel 
        model = CnnAttnModel(args)
    if args.model_type.lower() == 'crnn-attn':
        print('Use self attention (with crnn).')
        from model import CrnnAttnModel 
        model = CrnnAttnModel(args)
    if args.model_type.lower() == 'lmf':
        print('Use low-rank multimodal fusion.')
        from model import LmfModel
        model = LmfModel(args)
    if args.model_type.lower() == 'tfn':
        print('Use tensor fusion network for multimodal.')
        from model import TfnModel
        model = TfnModel(args)
    if args.model_type.lower() == 'iaf':
        print('Use intermedial attention fusion.')
        from model import IafModel
        model = IafModel(args)

    # Train and validate for each seed
    if args.eval_model is None:  
        seeds = range(args.seed, args.seed + args.n_seeds)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)
            data_loader = {}
            
            trainset = datasets['train']
            devset = datasets['devel']
            if args.combine_train_dev: 
                # add dev to trainset, works on humor
                # ERR with collate_fn on perception
                datasets['train'] = torch.utils.data.ConcatDataset((trainset, devset))
            
            for partition, dataset in datasets.items():  # one DataLoader for each partition
                batch_size = args.batch_size if partition == 'train' else 2 * args.batch_size
                shuffle = True if partition == 'train' else False  # shuffle only for train partition                
                data_loader[partition] = torch.utils.data.DataLoader(dataset, 
                                                                     batch_size=batch_size, 
                                                                     shuffle=shuffle,
                                                                     num_workers=4,
                                                                     worker_init_fn=seed_worker,
                                                                     collate_fn=collate_fn)
            
            #print(len(trainset), len(devset))
            #print(next(iter(data_loader['train'])))
            #sys.exit(0)
            print('=' * 50)
            print(f'Training model... [seed {seed}] for at most {args.epochs} epochs')
            val_loss, val_score, best_model_file = train_model(args.task, 
                                                               model, 
                                                               data_loader, 
                                                               args.epochs,
                                                               args.lr, 
                                                               args.paths['model'], 
                                                               seed, 
                                                               loss_fn=loss_fn, 
                                                               eval_fn=eval_fn,
                                                               eval_metric_str=eval_str,
                                                               regularization=args.regularization,
                                                               early_stopping_patience=args.early_stopping_patience)
            
            model = torch.load(best_model_file) # restore best model encountered during training
            if args.eval_model:  # run evaluation only if test labels are available.
                test_loss, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn,
                                                 eval_fn=eval_fn)
            else:
                test_loss, test_score = float(0.0), float(0.0)
            test_scores.append(test_score)
            print(f'[Test {eval_str}]:  {test_score:7.4f}')

            val_losses.append(val_loss)
            val_scores.append(val_score)

            best_model_files.append(best_model_file)

        best_idx = val_scores.index(max(val_scores))  # find best performing seed
        _val_score = f'{val_scores[best_idx]:7.4f}'
        _test_score = f'{test_scores[best_idx]:7.4f}' 

        print('=' * 50)
        print(f'Best {eval_str} on [Val] for seed {seeds[best_idx]}: '
              f'[Val {eval_str}]: {_val_score}'
              f' | [Test {eval_str}]: {_test_score}')
        print('=' * 50)

        model_file = best_model_files[best_idx]  # best model of all of the seeds
        if not args.result_csv is None:
            log_results(args.result_csv, params=args, seeds=list(seeds), metric_name=eval_str,
                        model_files=best_model_files, test_results=test_scores, val_results=val_scores,
                        best_idx=best_idx)

    else:  # Evaluate existing model (No training)
        model_file = os.path.join(args.paths['model'], f'model_{args.eval_seed}.pth')
        model = torch.load(model_file, map_location=torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
        data_loader = {}
        for partition, dataset in datasets.items():  # one DataLoader for each partition
            batch_size = args.batch_size if partition == 'train' else 2 * args.batch_size
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=4,
                                                                 worker_init_fn=seed_worker,
                                                                 collate_fn=collate_fn)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        if args.eval_model:
            _, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                                     )
            print(f'[Test {eval_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting devel and test samples...')
        print(f'Predition path: {args.paths['predict']}')
        best_model = torch.load(model_file, map_location=config.device)
        _, valid_score = evaluate(args.task, best_model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        evaluate(args.task, best_model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                 predict=True, prediction_path=args.paths['predict'],
                 filename='predictions_devel.csv')
        evaluate(args.task, best_model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                 predict=True, prediction_path=args.paths['predict'], 
                 filename='predictions_test.csv')
        print(f'Predictions saved in {os.path.join(args.paths["predict"])}')

    if not args.save_ckpt:
        for _file in best_model_files:
            os.remove(_file)
            # print(f'Remove ckpt {_file}. (Disk quota)') 

    print('Done.')


if __name__ == '__main__':
    print("Start",flush=True)
    args = parse_args()

    # debug
    # print(f'args.feature: {args.feature}, id: {'_'.join(args.feature.replace(os.path.sep, "-").split())}')
    model_id=datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M")
    if args.combine_train_dev:
        model_id+='-combine-train-dev'
    feat_id='+'.join(args.feature.replace(os.path.sep, "-").split())
    if args.feature_length:
        assert not args.feature_length == 0
        if args.feature_length>0:
            feat_id+=f'+first-{str(abs(args.feature_length))}-sec'
        else:
            feat_id+=f'+last-{str(abs(args.feature_length))}-sec'
    #print(f'feat_id: {feat_id}')
    #sys.exit(0)
    args.log_file_name =  '{}_{}_[{}]_[{}_{}]'.format(args.model_type.upper(), 
                                                            model_id, 
                                                            feat_id,
                                                            args.lr,
                                                            args.batch_size)

    # adjust your paths in config.py
    task_id = args.task if args.task != PERCEPTION else os.path.join(args.task, args.label_dim)
    args.paths = {'log': os.path.join(config.LOG_FOLDER, task_id) if not args.predict else os.path.join(config.LOG_FOLDER, task_id, 'prediction'),
                  'data': os.path.join(config.DATA_FOLDER, task_id),
                  'model': os.path.join(config.MODEL_FOLDER, task_id, args.log_file_name if not args.eval_model else args.eval_model)}
    if args.predict:
        if args.eval_model:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, task_id, args.eval_model, args.eval_seed)
        else:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, task_id, args.log_file_name)

    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))

    main(args)

    #os.system(f"rm -r {config.OUTPUT_PATH}")
