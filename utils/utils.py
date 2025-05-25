import os
import math
import torch
import random
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torchmetrics import Accuracy, AUROC, MeanSquaredError
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from tabulate import tabulate

def create_train_setup(dataset):
    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary')
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = dataset.num_classes
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=dataset.num_classes)
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False)
        higher_is_better = False

    return out_channels, loss_fun, metric_computer, higher_is_better, dataset.task_type

def create_multitask_setup(datasets):
    num_classes_list, loss_fns, metric_computers, higher_is_betters, task_types = [], [], [], [], []
    
    for dataset in datasets:
        out_channels, loss_fn, metric_computer, higher_is_better, task_type = create_train_setup(dataset)

        num_classes_list.append(out_channels)
        loss_fns.append(loss_fn)
        metric_computers.append(metric_computer)
        higher_is_betters.append(higher_is_better)
        task_types.append(task_type)
    
    return num_classes_list, loss_fns, metric_computers, higher_is_betters, task_types

def init_best_metric(higher_is_better):
    if higher_is_better:
        best_val_metric = 0
        best_test_metric = 0
    else:
        best_val_metric = math.inf
        best_test_metric = math.inf

    return best_val_metric, best_test_metric

def init_best_metrics(higher_is_better, num_tasks):
    if higher_is_better:
        best_val_metric = 0
        best_test_metric = 0
    else:
        best_val_metric = math.inf
        best_test_metric = math.inf
    best_val_metrics = [best_val_metric] * num_tasks
    best_test_metrics = [best_test_metric] * num_tasks
    return best_val_metrics, best_test_metrics

def print_log(args, epoch, values, name="Train Loss"):
    print(f"Epoch {epoch+1}/{args.epochs} - {name}")
    for task_idx, value in enumerate(values): 
        print(f"{task_idx}: {value:.6f}", end = '\t'); 
    print()

def tensorboard_write(writer, epoch, metrics, losses, task_idx=None, name="Train"):
    if task_idx is None:
        loss_dict = {f"Task_{i}_{name}": losses[i] for i in range(len(losses))}
        metric_dict = {f"Task_{i}_{name}": metrics[i] for i in range(len(metrics))}
        writer.add_scalars("Loss", loss_dict, epoch)
        writer.add_scalars("Metric", metric_dict, epoch)
    else:
        writer.add_scalar(f"Loss/Task_{task_idx}_{name}", losses, epoch)
        writer.add_scalar(f"Metric/Task_{task_idx}_{name}", metrics, epoch)

def save_results(
    args,
    model_config: dict,
    best_test_metrics: list[float],
    train_history: dict,
    best_finetune_test_metrics: list[float] = None,
    finetune_history: dict = None,
):
    if best_finetune_test_metrics is not None:
        num_runs = len(best_finetune_test_metrics)
    else:
        num_runs = len(best_test_metrics)

    for idx in range(num_runs):
        path = os.path.join("output", args.task_type, args.scale, str(idx), f"{args.exp_name}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            'args': vars(args),
            'model_config': model_config,
            'train_history': train_history[idx],
        }

        if best_finetune_test_metrics is not None and finetune_history is not None:
            payload.update({
                'finetune_history': finetune_history[idx],
                'best_test_before_finetune': best_test_metrics[idx],
                'best_test_metric': best_finetune_test_metrics[idx],
            })
        else:
            payload['best_test_metric'] = best_test_metrics[idx]

        torch.save(payload, path)
        print(f"Saved results to {path}")

def save_finetune_results(
    args,
    model_config: dict,
    best_test_metrics: list[float],
    best_finetune_test_metrics: list[float],
    train_history: dict,
    finetune_history: dict,
):
    for idx, finetune_test_metric in enumerate(best_finetune_test_metrics):
        path = os.path.join("output", args.task_type, args.scale, str(idx), f"{args.exp_name}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'args': vars(args),
            'model_config': model_config,
            'train_history': train_history[idx],  
            'finetune_history': finetune_history[idx],  
            'best_test_before_finetune': best_test_metrics[idx],
            'best_test_metric': finetune_test_metric
        }, path)
        print(f"Saved results to {path}")



def update_history(
    history, 
    train_losses, 
    train_metrics, 
    val_losses, 
    val_metrics
):
    for task_idx in range(len(train_losses)):
        history[task_idx]["train_losses"].append(train_losses[task_idx])
        history[task_idx]["train_metrics"].append(train_metrics[task_idx])
        history[task_idx]["val_losses"].append(val_losses[task_idx])
        history[task_idx]["val_metrics"].append(val_metrics[task_idx])

def update_finetune_history(
    history, 
    train_loss,
    train_acc,
    val_loss,
    val_metric
):        
    history["finetune_train_loss"].append(train_loss)
    history["finetune_train_acc"].append(train_acc)
    history["finetune_valid_loss"].append(val_loss)
    history["finetune_valid_acc"].append(val_metric)

def print_log_multitask(epoch, epochs,
                train_losses, train_metrics,
                valid_losses, valid_metrics,
                metric_name="Acc"):
    headers = ["Task ID", "Train Loss", f"Train {metric_name}", "Valid Loss", f"Valid {metric_name}"]
    table = [
        [f"Task {i}", 
        f"{train_losses[i]:.4f}", f"{train_metrics[i]:.4f}",
        f"{valid_losses[i]:.4f}", f"{valid_metrics[i]:.4f}"]
        for i in range(len(train_losses))
    ]
    print(f"\n[Epoch {epoch+1}/{epochs}]")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
def print_log_finetune(history: dict, metric_name: str = "Acc", task_idx: int = 0):
    headers = ["Epoch", "Train Loss", f"Train {metric_name}", "Valid Loss", f"Valid {metric_name}"]
    table = [
        [epoch + 1,
         f"{history['finetune_train_loss'][epoch]:.4f}",
         f"{history['finetune_train_acc'][epoch]:.4f}",
         f"{history['finetune_valid_loss'][epoch]:.4f}",
         f"{history['finetune_valid_acc'][epoch]:.4f}"]
        for epoch in range(len(history['finetune_train_loss']))
    ]
    print(f"Task {task_idx}")
    print(tabulate(table, headers=headers, tablefmt="grid"))

def create_history(num_tasks):
    history = {}
    for task_idx in range(num_tasks):
        history[task_idx] = {
            'train_losses':[],
            'train_metrics':[],
            'val_losses':[],
            'val_metrics':[],
            'test_losses':[],
            'test_metrics':[],
        }
    return history


def create_finetune_history():
    history = {
        "finetune_train_loss": [],
        "finetune_train_acc":  [],
        "finetune_valid_loss": [],
        "finetune_valid_acc":  [],
        "finetune_test_loss":  None,
        "finetune_test_acc":   None,
    }
    return history

def shuffle_task_indices(loaders: list[DataLoader]) -> list[int]:
    task_order = []
    for idx, loader in enumerate(loaders):
        task_order.extend([idx] * len(loader))
    random.shuffle(task_order)
    return task_order