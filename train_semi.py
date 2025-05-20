import argparse
import os
import random
import copy
import numpy as np
import pandas as pd

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from tqdm import tqdm

from torch_frame import NAStrategy, stype
from torch_frame.typing import TaskType
from torch_frame.data.stats import StatType
from torch_frame.data import DataLoader
from models import TabPerceiverMultiTask
from loaders import build_dataset, build_fewshot_dataset, build_dataloader, build_datasets, build_dataloaders
from utils import create_train_setup, create_multitask_setup, init_best_metric, init_best_metrics, update_history, update_finetune_history, save_finetune_results
from utils import print_log_multitask, print_log_finetune, create_history, create_finetune_history, shuffle_task_indices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# split train_dataset into few-shot dataset and unlabeled dataset



def main(args):

    # finetune and evaluate on new task
    fewshot_history = {}
    task_idx = 13
    dataset = build_dataset(task_type=args.task_type, dataset_scale=args.scale, dataset_index=task_idx)
    num_classes, loss_fn, metric_computer, higher_is_better, task_type = create_train_setup(dataset)
    


    # multitask learning 
    model_config = {
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "num_latents": args.num_latents,
        "hidden_dim": args.hidden_dim,
        "mlp_ratio": args.mlp_ratio,
        "dropout_prob": args.dropout_prob,
    }

    # finetune and evaluate on new task
    fewshot_history = {}
    task_idx = 13
    dataset = build_dataset(task_type=args.task_type, dataset_scale=args.scale, dataset_index=task_idx)
    num_classes, loss_fn, metric_computer, higher_is_better, task_type = create_train_setup(dataset)
    
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()
    print(f"Original dataset:")
    for k, v in dataset.col_stats.items():
        print(k)
        print(v)


    print(f"\n\nSampled dataset:")
    fewshot_train_dataset = build_fewshot_dataset(train_dataset, 1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multitask training and fine-tuning script")
    parser.add_argument('--task_type', type=str,
                        choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='TabPerceiverFewShot_')
    # config
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_latents', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--mlp_ratio', type=float, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0)
    
    args = parser.parse_args()
    
    main(args)