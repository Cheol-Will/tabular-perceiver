import argparse
import os
import time
import random
from typing import Any, Optional

import optuna
import numpy as np
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric

from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabPerceiverMultiTask
from utils import create_multitask_setup, init_best_metrics
from loaders import build_datasets, build_dataloaders

TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_environment(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def shuffle_task_indices(loaders: list[DataLoader]) -> list[int]:
    task_order = []
    for task_idx, loader in enumerate(loaders):
        task_order.extend([task_idx] * len(loader))
    random.shuffle(task_order)
    return task_order


def train_epoch(
    model: Module,
    loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fn: Module,
    task_type: TaskType,
    task_idx_list: list[int]
) -> float:
    model.train()
    iters = [iter(loader) for loader in loaders]
    total_loss = 0.0
    total_samples = 0

    for task_idx in task_idx_list:
        tf = next(iters[task_idx]).to(device)
        y = tf.y
        preds = model(tf, task_idx)

        if preds.size(1) == 1:
            preds = preds.view(-1)
        if task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / total_samples


def evaluate_task(
    model: Module,
    loader: DataLoader,
    metric_computer: Metric,
    task_type: TaskType,
    task_idx: int
) -> float:
    model.eval()
    metric_computer.reset()

    with torch.no_grad():
        for tf in loader:
            tf = tf.to(device)
            preds = model(tf, task_idx)

            if task_type == TaskType.MULTICLASS_CLASSIFICATION:
                preds = preds.argmax(dim=-1)
            elif task_type == TaskType.REGRESSION and preds.ndim > 1:
                preds = preds.view(-1)

            metric_computer.update(preds, tf.y)

    return metric_computer.compute().item()


def train_and_evaluate(
    model: Module,
    train_loaders: list[DataLoader],
    valid_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: ExponentialLR,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    higher_is_better: bool,
    trial: Optional[optuna.trial.Trial] = None,
) -> tuple[list[float], list[float]]:
    num_tasks = len(train_loaders)
    best_val_metrics, best_test_metrics = init_best_metrics(higher_is_better, num_tasks)

    for epoch in range(epochs):
        task_idx_list = shuffle_task_indices(train_loaders)
        train_loss = train_epoch(model, train_loaders, optimizer, loss_fn, task_type, task_idx_list)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

        for task_idx in range(num_tasks):
            val_metric = evaluate_task(model, valid_loaders[task_idx], metric_computer, task_type, task_idx)
            improved = (val_metric > best_val_metrics[task_idx]) if higher_is_better else (val_metric < best_val_metrics[task_idx])

            if improved:
                best_val_metrics[task_idx] = val_metric
                best_test_metrics[task_idx] = evaluate_task(model, test_loaders[task_idx], metric_computer, task_type, task_idx)
            print(f"Epoch {epoch+1}/{epochs} [Task {task_idx}] - Best test metric: {best_test_metrics[task_idx]:.6f}")

        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_metrics, best_test_metrics


def train_and_eval_with_cfg(
    model_cfg: Module,
    train_cfg: dict,
    epochs: int,
    datasets,
    batch_size: int,
    trial: Optional[optuna.trial.Trial] = None,
):
    train_loaders, valid_loaders, test_loaders, meta = build_dataloaders(datasets, batch_size)
    num_classes, loss_fn, metric_computer, higher_is_better, task_type = create_multitask_setup(datasets)
    
    model = TabPerceiverMultiTask(
            **model_cfg,
            **meta,
            num_classes=num_classes,
        ).to(device)
    model.reset_parameters()
    # Use train_cfg to set up training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['base_lr'])
    lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_rate'])

    best_val_metrics, best_test_metrics = train_and_evaluate(
        model, train_loaders, valid_loaders, test_loaders, epochs,
        optimizer, lr_scheduler, loss_fn, metric_computer, task_type, higher_is_better, trial
    )
    return best_val_metrics, best_test_metrics


def save_results(
    args,
    train_config: dict,
    model_config: dict,
    best_val_metrics_array,
    best_test_metrics_array,
):
    for task_idx in range(best_test_metrics_array.shape[1]):
        path = os.path.join("output", args.task_type, args.scale, str(task_idx), f"{args.exp_name}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'args': vars(args),
            'train_config': train_config,
            'model_config': model_config,
            'best_val_metric': best_val_metrics_array[:,task_idx].mean(),
            'best_val_metrics': best_val_metrics_array[:,task_idx],
            'best_test_metric': best_test_metrics_array[:,task_idx].mean(),
            'best_test_metrics': best_test_metrics_array[:,task_idx]
        }, path)
        print(f"Saved results to {path}")


def main(args):
    setup_environment(args.seed)

    datasets = build_datasets(task_type=args.task_type, dataset_scale=args.scale)
    higher_is_better = False if datasets[0].task_type == TaskType.REGRESSION else True

    def objective(trial: optuna.trial.Trial) -> float:
        model_cfg = {}
        for name, search_list in model_search_space.items():
            model_cfg[name] = trial.suggest_categorical(name, search_list)
        train_cfg = {}
        for name, search_list in train_search_space.items():
            train_cfg[name] = trial.suggest_categorical(name, search_list)
        print(f"[Trial {trial.number}] model_cfg = {model_cfg}, train_cfg = {train_cfg}")

        batch_size = train_cfg["batch_size"]
        best_val_metrics, _ = train_and_eval_with_cfg(model_cfg=model_cfg,
                                                    train_cfg=train_cfg,
                                                    epochs=args.epochs,
                                                    datasets=datasets,
                                                    batch_size=batch_size,
                                                    trial=trial)
        return sum(best_val_metrics)/len(best_val_metrics)

    study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(),
            direction="maximize" if higher_is_better else "minimize")
    study.optimize(objective, n_trials = args.num_trials)
    print("Hyper-parameter search done. Found the best config.")
    params = study.best_params
    best_train_cfg = {}
    for train_cfg_key in TRAIN_CONFIG_KEYS:
        best_train_cfg[train_cfg_key] = params.pop(train_cfg_key)
    best_model_cfg = params
    batch_size = best_train_cfg["batch_size"]
    print(f"Repeat experiments {args.num_repeats} times with the best train "
          f"config {best_train_cfg} and model config {best_model_cfg}.")
    
    best_val_metrics_array, best_test_metrics_array = [], []
    for _ in range(args.num_repeats):
        best_val_metrics, best_test_metrics = train_and_eval_with_cfg(
            model_cfg=best_model_cfg,
            train_cfg=best_train_cfg,
            epochs=args.epochs,
            datasets=datasets,
            batch_size=batch_size)
        best_val_metrics_array.append(best_val_metrics)
        best_test_metrics_array.append(best_test_metrics)
    
    best_val_metrics_array = np.array(best_val_metrics_array)
    best_test_metrics_array = np.array(best_test_metrics_array)
    save_results(args, best_train_cfg, best_model_cfg, best_val_metrics_array, best_test_metrics_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multitask Hyperparameter Tuning script")
    parser.add_argument('--task_type', type=str,
                        choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_trials', type=int, default=20,
                        help='Number of Optuna-based hyper-parameter tuning.')
    parser.add_argument(
        '--num_repeats', type=int, default=5,
        help='Number of repeated training and eval on the best config.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='TabPerceiverMultiTaskOptuna_')
    args = parser.parse_args()
    
    main(args)