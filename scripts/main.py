""" Train model from command line. """

import argparse
import os
import time

import pickle
import random

import numpy as np
from copy import deepcopy

from data_ops import read_compress
from data_ops import manual_describe
from data_ops import split_data
from data_ops import variables

from feature_select import important_features
from feature_select import get_important_features

from metrics import get_precision_score, get_accuracy_score
from metrics import get_recall_score, get_f1_score

from model_utils import import_model
from model_utils import wrap_model
from model_utils import train_model

from viz_utils import visualize_confusion_matrix
from viz_utils import correlogram
from viz_utils import univariate_plot


def configure_args():
    """ Configure cli arguments. """

    args = argparse.ArgumentParser(description='Arguments for training password strength detector.')

    args.add_argument('--t1', default=0.25, type=float, help='Train-valid/test split')
    args.add_argument('--t2', default=0.4, type=float, help='Train/valid split')
    
    args.add_argument('--n_jobs', default=-1, type=int, help='Number of threads')

    args.add_argument('--depth', default = 3, type = int, help = 'Max depth for tree model')
    
    args.add_argument('--r_state', default=42, type=int, help='Random state')

    args.add_argument('--data_dir', type=str, default = os.path.join(os.getcwd().replace('scripts', 'data'),
                                                                     'dataset', 'Sensorless_drive_diagnosis.txt'),
                      help='Data directory')

    args.add_argument('--arch_dir', type=str, default=os.path.join(os.getcwd().replace('scripts', 'data'),
                                                                   'archive', 'sensorless_dataset.zip'),
                      help='Compressed data directory')
    
    args.add_argument('--thresh', type = float, default = 1.0, help = 'Minimum feature importance')

    args.add_argument('--train', default=True, type=bool, choices=[True, False], help='Show train scores')
    args.add_argument('--valid', default=True, type=bool, choices=[True, False], help='Show valid scores')
    args.add_argument('--test', default=True, type=bool, choices=[True, False], help='Show test scores')

    args.add_argument('--visualize', default=False, type=bool, choices=[True, False],
                      help='Visualise data?')

    args.add_argument('--matrix', default=True, type=bool, choices=[True, False],
                      help= 'Show confusion matrix?')

    args.add_argument('--acc', default=True, type=bool, choices=[True, False], help='Show accuracy score')
    args.add_argument('--rec', default=True, type=bool, choices=[True, False], help='Show recall score')
    args.add_argument('--pre', default=True, type=bool, choices=[True, False], help='Show precision score')
    args.add_argument('--f1', default=True, type=bool, choices=[True, False], help='Show f1 score')

    args.add_argument('--avg', default='macro', choices=['micro', 'macro', 'samples', 'weighted', 'binary'],
                      help='Metric aggregation')

    args.add_argument('--text', default=True, choices=[True, False], help='Display text with diagnostics')

    args.add_argument('--dp', default=7, type=int, help='Rounding precision for report metrics')

    args.add_argument('--save', default=True, type=bool, help='Save trained model')

    args.add_argument('--model_name', default='trained_model.pkl', type=str,
                      help='Name for trained model')

    args.add_argument('--model_dir', default=os.getcwd().replace('scripts', 'artefacts'),
                      type=str, help='Storage location for saved model')
    
    args.add_argument('--report_dir', default=os.getcwd().replace('scripts', 'reports'),
                      type=str, help='Storage location for generated reports and visuals')

    return args


def main():
    ### CLI arguments

    start_time = time.time()
    origin_time = deepcopy(start_time)

    print('>>> Parsing CLI arguments...')
    start_time = time.time()
    args = configure_args().parse_args()
    print(f'>>> CLI arguments parsed! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    ### Dataset
    print('>>> Importing dataset...')
    start_time = time.time()

    if not os.path.exists(args.arch_dir.replace('sensorless_dataset.zip', '')):
        os.makedirs(args.arch_dir.replace('sensorless_dataset.zip', ''))

    original_data, path_to_archive = read_compress(path_to_data = args.data_dir,
                                                   path_to_archive = args.arch_dir)
    
    ### Features and target
    data, targets = variables(original_data)

    print(f'>>> Dataset successfully imported! Time elapsed : {time.time() - start_time:.5f} secs.',
          f'\n\t> Number of data observations : [{data.shape[0]}]',
          f'\n\t> Feature dimensions : [{data.shape[1]}]')
    print()

    ### Reproducibility
    print('>>> Ensuring reproducibility...')
    print('\t> Setting global and local random seeds...')

    start_time = time.time()

    random.seed(args.r_state)
    os.environ['PYTHONHASHSEED'] = str(args.r_state)
    np.random.default_rng(args.r_state)

    print('\t> Random seeds set!')
    print(f'>>> Reproducibility ensured! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()
    
    if not os.path.exists(args.report_dir):
        os.makedirs(os.path.join(args.report_dir, 'images', '1'))
        os.makedirs(os.path.join(args.report_dir, 'images', '2'))
        os.makedirs(os.path.join(args.report_dir, 'images', '3'))

        os.makedirs(os.path.join(args.report_dir, 'text'))
    else:
        pass
    
    if args.visualize:
        print(f'>>> Visualizing dataset...')
        univariate_plot(data, path = os.path.join(args.report_dir, 'images', '1', 'dist_plot'),
                        save = args.save)

        correlogram(original_data, path = os.path.join(args.report_dir, 'images', '2', 'pair_plot'),
                    save = args.save)
    print()
    
    print('>>> Describing dataset...')
    print(manual_describe(data, path = os.path.join(args.report_dir, 'text'), save = args.save))
    print()
    
    print('>>> Selecting significant feature subset...')
    start_time = time.time()
    model = import_model(max_depth=args.depth, n_jobs=args.n_jobs, random_state = args.r_state)
    
    importances = important_features(model = model, data = original_data,
                                     path = args.report_dir, save = args.save,
                                     sort = True, display = True)
    
    data = get_important_features(data = data, feature_ranking = importances,
                                  threshold = args.thresh)

    print(f'>>> Important features selected! Time elapsed: {time.time() - start_time:.5f} secs.',
          f'\n\t> Number of features selected : [{data.shape[1]}]')
    print()


    ### Data splitting
    print('>>> Splitting data into [train-valid-test] folds...')
    start_time = time.time()

    X_train, X_test, y_train, y_test = split_data(data, targets, split_size=args.t1)
    X_valid, X_test, y_valid, y_test = split_data(X_test, y_test, split_size=args.t2)

    print(f'>>> Data splits created! Time elapsed : {time.time() - start_time:.5f} secs.')
    print(f'\t> Number of Training observations: [{X_train.shape[0]}]',
          f'\n\t> Number of Validation observations: [{X_valid.shape[0]}]',
          f'\n\t> Number of Test observations: [{X_test.shape[0]}]')
    print()

    ### Model fitting
    print('>>> Importing model; Training model...')
    start_time = time.time()
    
    model = import_model(max_depth=args.depth, n_jobs=args.n_jobs, random_state = args.r_state)
    model = wrap_model(model, random_state = args.r_state)
    model = train_model(model, X_train, y_train)
    
    print(f'>>> Model trained successfully! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    if args.save:
        print('>>> Saving artefacts...')
        start_time = time.time()

        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        else:
            pass

        print('\t> Saving model artefacts...')

        with open(os.path.join(args.model_dir, args.model_name), 'wb') as f:
            pickle.dump(model, f)

        print('\t> Model artefact saved!')
        print()

        print(f'>>> Artefact redundancy achieved! Time elapsed : {time.time() - start_time:.5f} secs.')
        print()

    if args.matrix:
        if args.train:
            visualize_confusion_matrix(model, X_train, y_train, split = 'train',
                                       path = os.path.join(args.report_dir, 'images', '3'))

        if args.valid:
            visualize_confusion_matrix(model, X_valid, y_valid, split = 'valid',
                                       path = os.path.join(args.report_dir, 'images', '3'))

        if args.test:
            visualize_confusion_matrix(model, X_test, y_test, split = 'test',
                                       path = os.path.join(args.report_dir, 'images', '3'))

    if args.train:
        train_preds = model.predict(X_train)

        print('>' * 10, 'Train Diagnostics', '<' * 10)

        print(get_accuracy_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print()

    if args.valid:
        valid_preds = model.predict(X_valid)

        print('>' * 10, 'Valid Diagnostics', '<' * 10)

        print(get_accuracy_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print()

    if args.test:
        test_preds = model.predict(X_test)

        print('>' * 10, 'Test Diagnostics', '<' * 10)

        print(get_accuracy_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print()

    print(f'>>> Program run successfully! Total Time elapsed : {time.time() - origin_time :.5f} secs.')


if __name__ == '__main__':
    main()


