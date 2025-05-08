import math
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, AUROC, MeanSquaredError
from torch_frame.typing import TaskType


def create_train_setup(dataset):
    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary')
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = dataset.num_classes
        loss_fn = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=dataset.num_classes)
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = MSELoss()
        metric_computer = MeanSquaredError(squared=False)
        higher_is_better = False
    return out_channels, loss_fn, metric_computer, higher_is_better, dataset.task_type

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
    for taks_idx, value in enumerate(values): 
        print(f"{taks_idx}: {value:.6f}", end = '\t'); 
    print()


def tensorboard_write(writer, epoch, metrics, losses, task_idx=None, name="Train"):
    if task_idx is None:
        tasks = range(len(metrics)) # write on every task
        for taks_idx in tasks:
            writer.add_scalar(f"Task_{task_idx}_{name}_Metric", metrics[taks_idx], epoch)
            writer.add_scalar(f"Task_{task_idx}_{name}_Loss", losses[taks_idx], epoch)
    else: 
        writer.add_scalar(f"Task_{task_idx}_{name}_Metric", metrics, epoch)
        writer.add_scalar(f"Task_{task_idx}_{name}_Loss", losses, epoch)


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