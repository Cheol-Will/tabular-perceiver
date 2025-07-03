import argparse
import os
import time
import random
import copy
import numpy as np
import pandas as pd
import optuna

from typing import Any, Optional

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from tqdm import tqdm

import torch_frame
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabM

from encoder.piecewise_linear_encoder import compute_bins 


from loaders import build_dataset, build_dataloader
from utils import create_train_setup, init_best_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_environment(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)

def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    epoch: int,    
) -> float:
    model.train()
    metric_computer.reset()
    loss_accum = total_count = 0

    for tf in tqdm(loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        y = tf.y # (B)
        pred = model(tf) # (B, K, C)  
        B = pred.shape[0]
        K = pred.shape[1]

        # print("[Debug]")
        # print(y.shape)
        # print(pred.shape)
        if pred.size(2) == 1:
            pred = pred.view(-1, K) # (B, K) for Reg and BC
        # print("[Debug]")
        # print(f"after view pred: {pred.shape}")

        if task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float) 
            y = y.unsqueeze(1).expand(-1, K) # (B, K)
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # (B) -> (B, K)
            y = y.unsqueeze(1).expand(-1, K) # (B, K)

        # print("[Debug]")
        # print(f"after unsq y: {y.shape}")
        
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
        
        # update metric
        metric_preds = pred.mean(dim=1) # (B, K, *) -> (B, *) 
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            metric_preds = metric_preds.argmax(dim=-1)
        else:
            metric_preds = metric_preds
        # print("[Debug]")
        # print(metric_preds.shape)
        # print(tf.y.shape)        
        metric_computer.update(metric_preds, tf.y)

    train_loss = loss_accum / total_count
    train_metric = metric_computer.compute().item()

    return train_loss, train_metric


def evaluate(
    model: Module,
    loader: DataLoader,
    metric_computer: Metric,
    task_type: TaskType,
) -> float:
    model.eval()
    metric_computer.reset()

    with torch.no_grad():
        for tf in loader:
            tf = tf.to(device)
            preds = model(tf)

            if task_type == TaskType.MULTICLASS_CLASSIFICATION:
                preds = preds.argmax(dim=-1)
            elif task_type == TaskType.REGRESSION and preds.ndim > 1:
                preds = preds.view(-1)

            metric_computer.update(preds, tf.y)

    return metric_computer.compute().item()

def train_and_evaluate(
    model: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: ExponentialLR,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    higher_is_better: bool,
    trial: Optional[optuna.trial.Trial] = None,
) -> tuple[list[float], list[float]]:
    
    best_val_metric, best_test_metric = init_best_metric(higher_is_better)

    for epoch in range(epochs):
        train_loss, train_metric = train(
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            metric_computer=metric_computer,
            task_type=task_type,
            epoch=epoch
        )
        scheduler.step()
        
        val_metric = evaluate(model, valid_loader, metric_computer, task_type)
        improved = (val_metric > best_val_metric) if higher_is_better else (val_metric < best_val_metric)

        if improved:
            best_val_metric = val_metric
            best_test_metric = evaluate(model, test_loader, metric_computer, task_type)

        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Metric: {train_metric:.4f}")
        print(f"Epoch {epoch+1}/{epochs} - Val Metric: {val_metric:.4f}")
    print(
        f'Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}')

    return best_val_metric, best_test_metric

def train_and_eval_with_cfg(
    model_cfg: dict,
    train_cfg: dict,
    epochs: int,
    dataset,
    trial: Optional[optuna.trial.Trial] = None,
):
    # train_loader returns (tensor_frame, index) 
    train_loader, valid_loader, test_loader, meta, train_dataset = build_dataloader(dataset, train_cfg["batch_size"], use_train_dataset=True)
    num_classes, loss_fn, metric_computer, higher_is_better, task_type = create_train_setup(dataset)

    if torch_frame.numerical in meta["col_names_dict"]:
        if "n_bins" in train_cfg: 
            numerical_cols = meta["col_names_dict"][torch_frame.numerical]
            compute_bins_kwargs = (
                {
                    'y': train_dataset.tensor_frame.y.to(
                        torch.long if task_type != TaskType.REGRESSION else torch.float
                    ),
                    'regression': task_type != TaskType.REGRESSION,
                    'verbose': True,
                }
            )
            bin_edges = compute_bins(
                X=torch.tensor(train_dataset.df[numerical_cols].values), # pass numerical columns only.
                n_bins=train_cfg['n_bins'],
                **compute_bins_kwargs,
                tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}, # defualt tree_kwargs.
            )
            print(f"Bin edges for {numerical_cols}")
            print(bin_edges)
    else:
        bin_edges = None
        print(f"Piecewise Linear Encoding is not called since current data does not have numerical featrues.")
    
    model = TabM(
        **model_cfg,
        **meta,
        num_classes=num_classes,
        bins_list=bin_edges, # pass pre-computed bins for numerical features.
    ).to(device)

    # Use train_cfg to set up training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['base_lr'])
    scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_rate'])
    best_val_metrics, best_test_metrics = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metric_computer=metric_computer,
        task_type=task_type,
        higher_is_better=higher_is_better,
        trial=trial
    )
    return best_val_metrics, best_test_metrics

def main(args):
    
    print("Hyper-parameter search via Optuna")
    print(args)
    print(f"Device: {device}")

    # define search space
    TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr", "n_bins"]
    model_search_space = {
        'embed_dim': [4, 8, 16, 32],
        'num_layers': [1, 2, 3, 4], # With piecewise linear embedding, set num_blocks up to 4.
        'hidden_dim': [32, 64, 128, 256],
        'k': [32], # fix the number of submodels to 32
    }
    train_search_space = {
        'batch_size': [128, 256],
        'base_lr': [0.0001, 0.001],
        'gamma_rate': [0.9, 0.95, 1.],
        'n_bins': [2, 4, 8, 16, 32, 64, 128]
    }

    dataset = build_dataset(task_type=args.task_type, dataset_scale=args.scale, dataset_index=args.idx)
    higher_is_better = False if dataset.task_type == TaskType.REGRESSION else True

    def objective(trial: optuna.trial.Trial) -> float:
        model_cfg = {}
        for name, search_list in model_search_space.items():
            model_cfg[name] = trial.suggest_categorical(name, search_list)
        train_cfg = {}
        for name, search_list in train_search_space.items():
            train_cfg[name] = trial.suggest_categorical(name, search_list)
        best_val_metric, _ = train_and_eval_with_cfg(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            epochs=args.epochs,
            dataset=dataset,
            trial=trial,
        )
        return best_val_metric

    start_time = time.time()
    study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(),
            direction="maximize" if higher_is_better else "minimize")
    study.optimize(objective, n_trials = args.num_trials)
    end_time = time.time()
    search_time = end_time - start_time
    print("Hyper-parameter search done. Found the best config.")
    params = study.best_params
    best_train_cfg = {}
    for train_cfg_key in TRAIN_CONFIG_KEYS:
        best_train_cfg[train_cfg_key] = params.pop(train_cfg_key)
    best_model_cfg = params

    print(f"Repeat experiments {args.num_repeats} times with the best train "
          f"config {best_train_cfg} and model config {best_model_cfg}.")
    start_time = time.time()
    best_val_metrics, best_test_metrics = [], []
    for _ in range(args.num_repeats):
        best_val_metric, best_test_metric = train_and_eval_with_cfg(
            model_cfg=best_model_cfg,
            train_cfg=best_train_cfg,
            epochs=args.epochs,
            dataset=dataset,
        )
        best_val_metrics.append(best_val_metric)
        best_test_metrics.append(best_test_metric)
    end_time = time.time()
    final_model_time = (end_time - start_time) / args.num_repeats
    best_val_metrics = np.array(best_val_metrics)
    best_test_metrics = np.array(best_test_metrics)
    print(best_test_metrics)
    result_dict = {
        'args': args.__dict__,
        'best_val_metrics': best_val_metrics,
        'best_test_metrics': best_test_metrics,
        'best_val_metric': best_val_metrics.mean(),
        'best_test_metric': best_test_metrics.mean(),
        'best_train_cfg': best_train_cfg,
        'best_model_cfg': best_model_cfg,
        'search_time': search_time,
        'final_model_time': final_model_time,
        'total_time': search_time + final_model_time,
    }
    # Save results
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dict, args.result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TabM-PLE Hyperparameter Tuning Script")
    parser.add_argument('--task_type', type=str,
                        choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--idx', type=int, default=0,
                        help='The index of the dataset within DataFrameBenchmark')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_trials', type=int, default=20,
                        help='Number of Optuna-based hyper-parameter tuning.')
    parser.add_argument(
        '--num_repeats', type=int, default=5,
        help='Number of repeated training and eval on the best config.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='TabM')
    parser.add_argument('--result_path', type=str, default='')
    args = parser.parse_args()
    
    main(args)