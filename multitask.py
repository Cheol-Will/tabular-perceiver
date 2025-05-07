import argparse
import os
import random

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric

from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabPerceiverMultiTask
from utils import create_multitask_setup, init_best_metrics
from loaders import build_datasets, build_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_environment(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def shuffle_task_indices(loaders: list[DataLoader]) -> list[int]:
    task_order = []
    for idx, loader in enumerate(loaders):
        task_order.extend([idx] * len(loader))
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
    # manage train loss and number of samples for each task separately
    iters = [iter(loader) for loader in loaders]
    total_losses = [0.0] * len*(iters)
    total_samples = [0] * len(iters)

    model.train()
    for task_idx in task_idx_list:
        tf = next(iters[task_idx]).to(device)
        y = tf.y
        preds = model(tf, task_idx)

        if preds.size(1) == 1:
            preds = preds.view(-1)
        if task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.float()

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_losses[task_idx] += loss.item() * y.size(0)
        # total_loss += loss.item() * y.size(0)
        total_samples[task_idx] += y.size(0)
    
    total_losses = np.array(total_losses)
    total_samples = np.array(total_samples)

    return total_losses / total_samples


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


def print_log(args, epoch, values, name="Train Loss"):
    print(f"Epoch {epoch+1}/{args.epochs} - {name}")
    for taks_idx, value in enumerate(values): 
        print(f"{taks_idx}: {value:.6f}", end = '\t'); 
    print()

def train_and_evaluate(
    args,
    model: Module,
    train_loaders: list[DataLoader],
    valid_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: ExponentialLR,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    higher_is_better: bool
) -> tuple[list[float], list[float]]:
    num_tasks = len(train_loaders)
    train_losses_list = []
    best_val_metrics, best_test_metrics = init_best_metrics(higher_is_better, num_tasks)

    for epoch in range(args.epochs):
        task_idx_list = shuffle_task_indices(train_loaders)
        train_losses = train_epoch(model, train_loaders, optimizer, loss_fn, task_type, task_idx_list)
        train_losses_list.append(train_losses)
        scheduler.step()
        print_log(args, epoch, train_losses, name="Train Loss")
    
        for task_idx in range(num_tasks):
            val_metric = evaluate_task(model, valid_loaders[task_idx], metric_computer, task_type, task_idx)
            improved = (val_metric > best_val_metrics[task_idx]) if higher_is_better else (val_metric < best_val_metrics[task_idx])

            if improved:
                best_val_metrics[task_idx] = val_metric
                best_test_metrics[task_idx] = evaluate_task(model, test_loaders[task_idx], metric_computer, task_type, task_idx)
            # print(f"Epoch {epoch+1}/{args.epochs} [Task {task_idx}] - Best test metric: {best_test_metrics[task_idx]:.6f}")
        print_log(args, epoch, best_test_metrics, name="Test Metrics")

    return train_losses_list, best_val_metrics, best_test_metrics


def finetune_and_evaluate(
    args,
    model: Module,
    train_loaders: list[DataLoader],
    valid_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ExponentialLR,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    task_idx: int,
    init_val: float,
    init_test: float,
    higher_is_better: bool
) -> float:
    best_val_metric, best_test_metric = init_val, init_test

    for epoch in range(args.finetune_epochs):
        finetune_indices = [task_idx] * len(train_loaders[task_idx])
        train_losses = train_epoch(model, train_loaders, optimizer, loss_fn, task_type, finetune_indices)
        scheduler.step()
        print_log(args, epoch, train_losses, name="Finetune Loss")

        val_metric = evaluate_task(model, valid_loader, metric_computer, task_type, task_idx)
        improved = (val_metric > best_val_metric) if higher_is_better else (val_metric < best_val_metric)

        if improved:
            best_val_metric = val_metric
            best_test_metric = evaluate_task(model, test_loader, metric_computer, task_type, task_idx)
        


        print_log(args, epoch, best_test_metric, name="Test Metrics")
        

    return best_test_metric


def save_results(
    args,
    model_config: dict,
    best_test_metrics: list[float],
    best_finetune_test_metrics: list[float]
):
    for idx, finetune_test_metric in enumerate(best_finetune_test_metrics):
        path = os.path.join("output", args.task_type, args.scale, str(idx), f"{args.exp_name}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'args': vars(args),
            'model_config': model_config,
            'best_test_before_finetune': best_test_metrics[idx],
            'best_test_metric': finetune_test_metric
        }, path)
        print(f"Saved results to {path}")


def main(args):
    setup_environment(args.seed)

    datasets = build_datasets(task_type=args.task_type, dataset_scale=args.scale)
    train_loaders, valid_loaders, test_loaders, meta = build_dataloaders(datasets)
    num_classes, loss_fn, metric_computer, higher_is_better, task_type = create_multitask_setup(datasets)
    metric_computer.to(device)

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
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    best_val_metrics, best_test_metrics = train_and_evaluate(
        args, model, train_loaders, valid_loaders, test_loaders,
        optimizer, scheduler, loss_fn, metric_computer, task_type, higher_is_better
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }

    best_finetune_test_metrics = []
    for idx in range(len(train_loaders)):
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        finetune_test_metric = finetune_and_evaluate(
            args, model, train_loaders,
            valid_loaders[idx], test_loaders[idx],
            optimizer, scheduler, loss_fn,
            metric_computer, task_type,
            idx, best_val_metrics[idx], best_test_metrics[idx], higher_is_better
        )
        best_finetune_test_metrics.append(finetune_test_metric)
        print(f"[Task {idx}] Best test metric (before finetune): {best_test_metrics[idx]:.6f}, Best test metric (after finetune):  {finetune_test_metric:.6f}")

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