import math
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torchmetrics import Accuracy, AUROC, MeanSquaredError
from torch_frame.typing import TaskType

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
    dataset = datasets[0]
    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        num_classes_list = [1 for _ in range(len(datasets))]
        loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary')
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        num_classes_list = [dataset.num_classes for dataset in datasets]
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=int(1e5))
        # no need to specify num_classes
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        num_classes_list = [1 for _ in range(len(datasets))]
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False)
        higher_is_better = False

    return num_classes_list, loss_fun, metric_computer, higher_is_better, dataset.task_type


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