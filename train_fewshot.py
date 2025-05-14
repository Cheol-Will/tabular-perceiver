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

def setup_environment(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

def train_task(
    args,
    model: Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    task_idx: int,
):
    """ Train on one task."""
    model.train()
    metric_computer.reset()
    total_loss = 0.0
    total_samples = 0
    for tf in train_loader:
        tf = tf.to(device)
        y = tf.y
        if task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.float()

        preds = model(tf, task_idx)
        if preds.ndim > 1 and preds.size(1) == 1:
            preds = preds.view(-1)

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_samples = y.size(0)
        total_loss += loss.item() * num_samples
        total_samples += num_samples

        # update metric
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            metric_preds = preds.argmax(dim=-1)
        else:
            metric_preds = preds
        metric_computer.update(metric_preds, tf.y)
    
    train_loss = total_loss / total_samples
    train_metric = metric_computer.compute().item()
    return train_loss, train_metric

def train_multitask(
    model: Module,
    loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fns: list[Module],
    metric_computers: list[Metric],
    task_types: list[TaskType],
    task_idx_list: list[int],
    epoch: int,
) -> float:
    """ Train on multiple tasks."""
    num_tasks = len(loaders)
    iters = [iter(loader) for loader in loaders]
    task_total_losses = [0.0] * num_tasks
    task_num_samples = [0] * num_tasks

    model.train()
    for task_idx in tqdm(task_idx_list, desc=f'Epoch: {epoch+1}'):
        tf = next(iters[task_idx]).to(device)
        y = tf.y
        preds = model(tf, task_idx)

        if preds.size(1) == 1:
            preds = preds.view(-1)
        if task_types[task_idx] == TaskType.BINARY_CLASSIFICATION:
            y = y.float()

        loss = loss_fns[task_idx](preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        task_total_losses[task_idx] += loss.item() * y.size(0)
        task_num_samples[task_idx] += y.size(0)

        if task_types[task_idx] == TaskType.MULTICLASS_CLASSIFICATION:
            preds = preds.argmax(dim=-1)
        elif task_types[task_idx] == TaskType.REGRESSION and preds.ndim > 1:
            preds = preds.view(-1)
        metric_computers[task_idx].update(preds, tf.y)

    task_metrics = np.array([metric_computer.compute().item() for metric_computer in metric_computers])
    task_losses =  np.array(task_total_losses) / np.array(task_num_samples)

    return task_losses, task_metrics


def evaluate_task(
    model: Module,
    loader: DataLoader,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    task_idx: int
) -> tuple[float, float]:
    """Evaluate on one task"""
    model.eval()
    metric_computer.reset()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for tf in loader:
            tf = tf.to(device)
            y = tf.y
            preds = model(tf, task_idx)
            
            if preds.size(1) == 1:
                preds = preds.view(-1)
            if task_type == TaskType.BINARY_CLASSIFICATION:
                y = y.float()

            loss = loss_fn(preds, y)
            total_loss += loss.item() * y.size(0)
            num_samples += y.size(0)

            if task_type == TaskType.MULTICLASS_CLASSIFICATION:
                preds = preds.argmax(dim=-1)
            elif task_type == TaskType.REGRESSION and preds.ndim > 1:
                preds = preds.view(-1)
            metric_computer.update(preds, tf.y)
    eval_loss = total_loss / num_samples
    eval_metric = metric_computer.compute().item()
    return eval_loss, eval_metric 

def train_and_evaluate(
    args,
    model: Module,
    train_loaders: list[DataLoader],
    valid_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: ExponentialLR,
    loss_fns: list[Module],
    metric_computers: list[Metric],
    task_types: list[TaskType],
    higher_is_betters: list[bool],
) -> tuple[list[float], list[float]]:
    """Train on multiple tasks and evaluate on each tasks"""
    num_tasks = len(train_loaders)
    best_val_metrics, best_test_metrics = init_best_metrics(higher_is_betters[0], num_tasks)
    history = create_history(num_tasks)

    # train on every task and evaluate
    for epoch in range(args.epochs):
        task_idx_list = shuffle_task_indices(train_loaders)
        train_losses, train_metrics = train_multitask(
            model=model, 
            loaders=train_loaders, 
            optimizer=optimizer, 
            loss_fns=loss_fns, 
            metric_computers=metric_computers, 
            task_types=task_types, 
            task_idx_list=task_idx_list,
            epoch=epoch)
        scheduler.step()

        val_metrics = []
        val_losses = []
        for task_idx in range(num_tasks):
            loss_fn, metric_computer, task_type = loss_fns[task_idx], metric_computers[task_idx], task_types[task_idx]
            val_loss, val_metric  = evaluate_task(model, valid_loaders[task_idx], loss_fn, metric_computer, task_type, task_idx)
            val_metrics.append(val_metric)
            val_losses.append(val_loss)

            improved = (val_metric > best_val_metrics[task_idx]) if higher_is_betters[task_idx] else (val_metric < best_val_metrics[task_idx])
            if improved:
                best_val_metrics[task_idx] = val_metric
                test_loss, best_test_metrics[task_idx] = evaluate_task(model, test_loaders[task_idx], loss_fn, metric_computer, task_type, task_idx)

        update_history(history, train_losses, train_metrics, val_losses, val_metrics)

        print_log_multitask(epoch, args.epochs,
                train_losses, train_metrics,
                val_losses, val_metrics,
                metric_name="Acc")

    return history, best_test_metrics



def finetune_and_evaluate(
    args,
    model: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ExponentialLR,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    task_idx: int,
    higher_is_better,
) -> dict:

    best_val_metric, best_test_metric = init_best_metric(higher_is_better)
    history = create_finetune_history()

    for epoch in tqdm(range(args.finetune_epochs),  desc=f'Fewshots-Task{task_idx}'):
        train_loss, train_acc = train_task(
            args=args,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_computer=metric_computer,
            task_type=task_type,
            task_idx=task_idx,
        )
        scheduler.step()

        val_loss, val_metric = evaluate_task(
            model, valid_loader,
            loss_fn, metric_computer,
            task_type, task_idx
        )

        improved = (val_metric > best_val_metric) if higher_is_better else (val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            test_loss, best_test_metric = evaluate_task(
                model, test_loader,
                loss_fn, metric_computer,
                task_type, task_idx
            )

        update_finetune_history(history, train_loss, train_acc, val_loss, val_metric)

    history["finetune_test_loss"] = test_loss
    history["finetune_test_metric"]  = best_test_metric
    print_log_finetune(history=history, metric_name="Acc", task_idx=task_idx)
    return history

def replace_unseen_categories(df, col_to_stype, col_stats):
    df = df.copy()
    for col, st in col_to_stype.items():
        if st == stype.categorical:
            seen_values = set(col_stats[col][StatType.COUNT][0])
            df[col] = df[col].apply(lambda x: x if x in seen_values else -1)
    return df


def build_fewshot_dataloader(train_dataset, val_dataset, test_dataset, batch_size=128, drop_last=True):
    # replace unseen value in train_dataset with -1
    print("Debug: ")
    print(val_dataset.df.iloc[:5, :5])
    print(val_dataset.df.iloc[:5, 5:10])
    val_dataset.df = replace_unseen_categories(val_dataset.df, train_dataset.col_to_stype, train_dataset.col_stats)
    test_dataset.df = replace_unseen_categories(test_dataset.df, train_dataset.col_to_stype, train_dataset.col_stats)
    print("Debug: ")
    print(val_dataset.df.iloc[:5, :5])
    print(val_dataset.df.iloc[:5, 5:10])
    val_dataset.materialize()
    test_dataset.materialize()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame

    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    valid_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    print(f'Training set has {len(train_tensor_frame)} instances')
    print(f'Validation set has {len(val_tensor_frame)} instances')
    print(f'Test set has {len(test_tensor_frame)} instances')

    col_stats = train_dataset.col_stats
    col_names_dict = train_tensor_frame.col_names_dict
    meta_data = {
        "col_stats": col_stats,
        "col_names_dict": col_names_dict,
    }
    return train_loader, valid_loader, test_loader, meta_data


def main(args):
    
    # load datasets for pretraining 
    # do not include last task
    datasets = build_datasets(task_type=args.task_type, dataset_scale=args.scale, num_tasks=13)
    train_loaders, valid_loaders, test_loaders, meta = build_dataloaders(datasets)
    num_classes_list, loss_fns, metric_computers, higher_is_betters, task_types = create_multitask_setup(datasets)
    for metric_computer in metric_computers:
        metric_computer.to(device)
    
    # multitask learning 
    model_config = {
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "num_latents": args.num_latents,
        "hidden_dim": args.hidden_dim,
        "mlp_ratio": args.mlp_ratio,
        "dropout_prob": args.dropout_prob,
    }
    model = TabPerceiverMultiTask(
        **model_config,
        **meta,
        num_classes=num_classes_list,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    _, _ = train_and_evaluate(
        args=args,
        model=model,
        train_loaders=train_loaders,
        valid_loaders=valid_loaders,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fns=loss_fns,
        metric_computers=metric_computers,
        task_types=task_types,
        higher_is_betters=higher_is_betters,
    )
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
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

    for k, v in fewshot_train_dataset.col_stats.items():
        print(k)
        print(v)


    # During inference, need to handle unseen categories of categorical features.
    shot_trials = [1, 5, 10, 100]
    for shots in shot_trials:
        # sample k shots
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        fewshot_train_dataset = build_fewshot_dataset(train_dataset, shots)
        target_col = fewshot_train_dataset.target_col 
        print(f"Train dataset has {len(fewshot_train_dataset)} instances.")
        print(fewshot_train_dataset.df[target_col])

        # dataloader
        fewshot_train_loader, fewshot_valid_loader, fewshot_test_loader, fewshot_meta_data = build_fewshot_dataloader(fewshot_train_dataset, val_dataset, test_dataset, batch_size=128, drop_last=False)
        
        # fewshot_train_loader, fewshot_valid_loader, fewshot_test_loader, fewshot_meta_data = build_dataloader(dataset, batch_size=128, drop_last=False)
        # init model with pretrained weights and add new task head 
        model = TabPerceiverMultiTask(
            **model_config,
            **meta,
            num_classes=num_classes_list,
        )
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        model.freeze_transformer() # freeze transformer block
        model.add_new_task(num_classes, **fewshot_meta_data)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=0.95)

        history = finetune_and_evaluate(
            args=args,
            model=model,
            train_loader=fewshot_train_loader,
            valid_loader=fewshot_valid_loader,
            test_loader=fewshot_test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metric_computer=metric_computer,
            task_type=task_type,
            task_idx=task_idx,
            higher_is_better=higher_is_better,
        )
        fewshot_history[shots] = history
        # intermediate save
        path = f"output/{args.task_type}/{args.scale}/{task_idx}/{args.exp_name}.pt"
        torch.save(fewshot_history, path)

    # save result
    path = f"output/{args.task_type}/{args.scale}/{task_idx}/{args.exp_name}.pt"
    torch.save(fewshot_history, path)

    # fewshot history test metric
    for k, v in fewshot_history.items():
        test_acc = v["finetune_test_metric"] 
        print(f"{k}-shots: {test_acc}")


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