import argparse
import os
import random
import copy
import numpy as np

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from tqdm import tqdm

from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabPerceiverMultiTask
from loaders import build_datasets, build_dataloaders
from utils import create_multitask_setup, init_best_metric, init_best_metrics, update_history, update_finetune_history, save_results
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
    """ Train on one task for finetune."""
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

        # loss = loss_fn(preds, y)
        loss = loss_fns[task_idx](preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # total_loss += loss.item() * y.size(0)
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

            # update test metrics if improved
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
    higher_is_better: bool
) -> dict:

    best_val_metric, best_test_metric = init_best_metric(higher_is_better)
    history = create_finetune_history()

    for epoch in tqdm(range(args.finetune_epochs),  desc=f'Finetune-Task{task_idx}'):
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
    history["finetune_test_acc"]  = best_test_metric
    print_log_finetune(history=history, metric_name="Acc", task_idx=task_idx)
    return history


def main(args):
    setup_environment(args)
    
    datasets = build_datasets(task_type=args.task_type, dataset_scale=args.scale)
    train_loaders, valid_loaders, test_loaders, meta = build_dataloaders(datasets)
    num_classes_list, loss_fns, metric_computers, higher_is_betters, task_types = create_multitask_setup(datasets)
    for metric_computer in metric_computers:
        metric_computer.to(device)

    # parse config
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

    train_history, best_test_metrics = train_and_evaluate(
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

    best_finetune_test_metrics = []
    fientune_history = {}
    task_zips = zip(
        train_loaders,
        valid_loaders,
        test_loaders,
        loss_fns,
        metric_computers,
        task_types,
        higher_is_betters
    )
    for task_idx, task_zip in enumerate(task_zips):
        train_loader, valid_loader, test_loader, loss_fn, metric_computer, task_type, higher_is_better = task_zip
        
        # load state dict
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        history = finetune_and_evaluate(
            args=args,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metric_computer=metric_computer,
            task_type=task_type,
            task_idx=task_idx,
            higher_is_better=higher_is_better
        )
        fientune_history[task_idx] = history
        best_finetune_test_metrics.append(history["finetune_test_acc"])

    save_results(args, model_config, best_test_metrics, train_history, best_finetune_test_metrics, fientune_history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multitask training and fine-tuning script")
    parser.add_argument('--task_type', type=str,
                        choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='TabPerceiverMultiTask_')
    # config
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_latents', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--mlp_ratio', type=float, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0)
    
    args = parser.parse_args()
    
    main(args)