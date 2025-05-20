import argparse
import math
import os
import os.path as osp
import random
import time
from typing import Any, Optional

import numpy as np
import optuna
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.nn.encoder import EmbeddingEncoder, LinearBucketEncoder
from torch_frame.typing import TaskType
from models import TabPerceiver, LinearL1, LightGBM
from loaders import build_fewshot_dataset

from typing import Any

TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
GBDT_MODELS = ["LightGBM"]
BASE_MODELS = ["LinearL1"]

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument(
    '--num_repeats', type=int, default=5,
    help='Number of repeated training and eval on the best config.')
parser.add_argument(
    '--model_type', type=str, default='LightGBM', choices=[
        'LightGBM', 'TabPerceiver', 'LinearL1',
    ])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--shots', type=int, default=1)
parser.add_argument('--result_path', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
random.seed(args.seed)

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                             scale=args.scale, idx=args.idx)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()

if args.model_type in GBDT_MODELS:
    gbdt_cls_dict = {
        'LightGBM': LightGBM
    }
    model_cls = gbdt_cls_dict[args.model_type]
elif args.model_type in BASE_MODELS:
    base_cls_dict = {
        'LinearL1': LinearL1,
    }
    model_cls = base_cls_dict[args.model_type]
else:
    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary').to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = dataset.num_classes
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=dataset.num_classes).to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False).to(device)
        higher_is_better = False

    # To be set for each model
    model_cls = None
    col_stats = None

    if args.model_type == 'TabPerceiver':
        model_search_space = {
            'num_heads': [4, 8],
            'num_layers': [4, 6, 8],
            'num_latents': [4, 8, 16, 32],
            'hidden_dim': [32, 64, 128, 256],
            'mlp_ratio': [0.25, 0.5, 1, 2, 4],
            'dropout_prob': [0, 0.2],
        }
        train_search_space = {
            'batch_size': [128, 256],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = TabPerceiver
        col_stats = dataset.col_stats

    assert model_cls is not None
    assert col_stats is not None
    assert set(train_search_space.keys()) == set(TRAIN_CONFIG_KEYS)
    col_names_dict = train_tensor_frame.col_names_dict


def main_base():
    # pseudo fewshot learning of tree model with CV
    # seeds = [0, 1, 32, 42, 1024]
    seeds = list(range(30))
    num_classes = dataset.num_classes
    history = {
        "train_metric": [],
        "test_metric": [],
    }
    print(f'col stats of original train dataset: {train_dataset.col_stats}')
    for seed in tqdm(seeds):
        # CV with k-shot dataset
        fewshot_train_dataset = build_fewshot_dataset(train_dataset, args.shots, seed)
        print(fewshot_train_dataset.df)
        if args.model_type == "LightGBM":
            estimator = LightGBM(task_type=dataset.task_type, num_classes=num_classes)
        elif args.model_type == "LinearL1":
            estimator = LinearL1(task_type=dataset.task_type, num_classes=num_classes)
        # merge valid and test dataset or just use only test dataset
        # test_dataset = merge_val_tests(val_dataset, test_dataset)

        train_metric, test_metric, hyperparameters = estimator.cross_validation(fewshot_train_dataset.tensor_frame, test_dataset.tensor_frame, seed)
        print(f"Train Metric: {train_metric:.7f} | Test Metric: {test_metric:.7f}")
        print(f"best hyperparameters: {hyperparameters}")
        history["train_metric"].append(train_metric)
        history["test_metric"].append(test_metric)

    print(history)
    print(np.mean(history["test_metric"]))

    # save result
    path = f"output/{args.task_type}/{args.scale}/{args.idx}/{args.model_type}_{args.shots}.pt"
    torch.save(history, path)

if __name__ == '__main__':
    print(args)

    if args.model_type in ["XGBoost", "CatBoost", "LightGBM", "LinearL1"]:
        main_base()
    else:
        main_deep_models()