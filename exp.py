import argparse
import os
import random
import copy
import numpy as np
import pandas as pd

from typing import Any

import torch
from torch.nn import Module
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torchmetrics import Accuracy, AUROC, MeanSquaredError
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from tqdm import tqdm

from torch_frame import NAStrategy, stype
from torch_frame.typing import TaskType
from torch_frame.data.stats import StatType
from torch_frame.data import DataLoader, Dataset
from models import TabPerceiverSemi
from loaders import build_dataset, build_dataloader
from utils import create_train_setup, init_best_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    for idx in range(14):
        dataset = build_dataset(task_type="binary_classification", dataset_scale="small", dataset_index=idx)
        dataset = dataset.shuffle()
        train_dataset, val_dataset, test_dataset = dataset.split()
        col_stats = train_dataset.col_stats

        y_col = dataset.target_col
        df_test = test_dataset.df
        print(f"{idx}: {df_test[y_col].value_counts(normalize=True)}")


if __name__ == "__main__":
    main()