import argparse
import os
import random
import copy
import numpy as np

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from torch.utils.tensorboard import SummaryWriter

from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabPerceiverMultiTask
from loaders import build_datasets, build_dataloaders
from utils import create_multitask_setup, init_best_metric, init_best_metrics, update_checkpoint, tensorboard_write, print_log, save_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_environment(args):
    writer = SummaryWriter(f'output/{args.task_type}/{args.scale}/{args.exp_name}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    return writer

def shuffle_task_indices(loaders: list[DataLoader]) -> list[int]:
    task_order = []
    for idx, loader in enumerate(loaders):
        task_order.extend([idx] * len(loader))
    random.shuffle(task_order)
    return task_order


def train_multitask(
    model: Module,
    loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fns: list[Module],
    metric_computers: list[Metric],
    task_types: list[TaskType],
    task_idx_list: list[int]
) -> float:
    # manage train loss and number of samples for each task separately
    num_tasks = len(loaders)
    iters = [iter(loader) for loader in loaders]
    task_losses = [0.0] * num_tasks
    task_num_samples = [0] * num_tasks

    model.train()
    for task_idx in task_idx_list:
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
        task_losses[task_idx] += loss.item() * y.size(0)
        task_num_samples[task_idx] += y.size(0)

        if task_types[task_idx] == TaskType.MULTICLASS_CLASSIFICATION:
            preds = preds.argmax(dim=-1)
        elif task_types[task_idx] == TaskType.REGRESSION and preds.ndim > 1:
            preds = preds.view(-1)
        metric_computers[task_idx].update(preds, tf.y)

    task_metrics = np.array([metric_computer.compute().item() for metric_computer in metric_computers])
    task_losses, task_num_samples = np.array(task_losses), np.array(task_num_samples)

    return task_metrics, task_losses / task_num_samples


def evaluate_task(
    model: Module,
    loader: DataLoader,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    task_idx: int
) -> float:
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
    eval_metric = metric_computer.compute().item()
    eval_loss = total_loss / num_samples
    return eval_metric, eval_loss


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
    writer: SummaryWriter,
) -> tuple[list[float], list[float]]:

    num_tasks = len(train_loaders)
    best_val_metrics, best_test_metrics = init_best_metrics(higher_is_betters[0], num_tasks)
    checkpoint = {}
    for task_idx in range(num_tasks):
        checkpoint[task_idx] = {
            'train_losses':[],
            'train_metrics':[],
            'val_losses':[],
            'val_metrics':[],
            'test_losses':[],
            'test_metrics':[],
        }

    # train on every task and evaluate
    for epoch in range(args.epochs):
        task_idx_list = shuffle_task_indices(train_loaders)
        train_metrics, train_losses = train_multitask(model, train_loaders, optimizer, loss_fns, metric_computers, task_types, task_idx_list)
        scheduler.step()

        val_metrics = []
        val_losses = []
        for task_idx in range(num_tasks):
            loss_fn, metric_computer, task_type = loss_fns[task_idx], metric_computers[task_idx], task_types[task_idx]
            val_metric, val_loss = evaluate_task(model, valid_loaders[task_idx], loss_fn, metric_computer, task_type, task_idx)
            val_metrics.append(val_metric)
            val_losses.append(val_loss)

            improved = (val_metric > best_val_metrics[task_idx]) if higher_is_betters[task_idx] else (val_metric < best_val_metrics[task_idx])
            if improved:
                best_val_metrics[task_idx] = val_metric
                best_test_metrics[task_idx], test_loss = evaluate_task(model, test_loaders[task_idx], loss_fn, metric_computer, task_type, task_idx)

        print(train_metrics)
        # update checkpoint and tensorboard        
        update_checkpoint(checkpoint, train_losses, train_metrics, name="train")
        tensorboard_write(writer, epoch, train_metrics, train_losses, name="train")

        update_checkpoint(checkpoint, train_losses, train_metrics, name="val")
        tensorboard_write(writer, epoch, val_metrics, val_losses, name="valid")

    return checkpoint, best_test_metrics


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
    train_acc = metric_computer.compute().item()

    return train_loss, train_acc

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
    history = {
        "finetune_train_loss": [],
        "finetune_train_acc":  [],
        "finetune_valid_loss": [],
        "finetune_valid_acc":  [],
        "finetune_test_loss":  None,
        "finetune_test_acc":   None,
    }

    for epoch in range(args.finetune_epochs):
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

        val_metric, val_loss = evaluate_task(
            model, valid_loader,
            loss_fn, metric_computer,
            task_type, task_idx
        )

        improved = (val_metric > best_val_metric) if higher_is_better else (val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            best_test_metric, test_loss = evaluate_task(
                model, test_loader,
                loss_fn, metric_computer,
                task_type, task_idx
            )

        history["finetune_train_loss"].append(train_loss)
        history["finetune_train_acc"].append(train_acc)
        history["finetune_valid_loss"].append(val_loss)
        history["finetune_valid_acc"].append(val_metric)

    history["finetune_test_loss"] = test_loss
    history["finetune_test_acc"]  = best_test_metric

    return history


def main(args):
    writer = setup_environment(args)
    
    datasets = build_datasets(task_type=args.task_type, dataset_scale=args.scale)
    train_loaders, valid_loaders, test_loaders, meta = build_dataloaders(datasets)
    num_classes_list, loss_fns, metric_computers, higher_is_betters, task_types = create_multitask_setup(datasets)
    for metric_computer in metric_computers:
        metric_computer.to(device)

    # lower model's capacity
    model_config = {
        "num_heads": 4,
        "num_layers": 6,
        "num_latents": 16,
        "hidden_dim": 64,
        "mlp_ratio": 2,
        "dropout_prob": 0,
    }

    model = TabPerceiverMultiTask(
        **model_config,
        **meta,
        num_classes=num_classes_list,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    train_log, best_test_metrics = train_and_evaluate(
        args, model, train_loaders, valid_loaders, test_loaders, optimizer, scheduler, 
        loss_fns, metric_computers, task_types, higher_is_betters, writer
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }

    best_finetune_test_metrics = []
    task_zip = zip(
        train_loaders,
        valid_loaders,
        test_loaders,
        loss_fns,
        metric_computers,
        task_types,
        higher_is_betters
    )
    for task_idx, (train_loader, valid_loader, test_loader, loss_fn, metric_computer, task_type, higher_is_better) in enumerate(task_zip):
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
        print(history)
        # best_finetune_test_metrics.append(finetune_test_metric["test_metrics"])
        # print(f"[Task {idx}] Best test metric (before finetune): {best_test_metrics[idx]:.6f}, Best test metric (after finetune):  {finetune_test_metric:.6f}")

    save_results(args, model_config, best_test_metrics, best_finetune_test_metrics)


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
    args = parser.parse_args()
    
    main(args)