import argparse
import os
from typing import Any, Optional

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from tqdm import tqdm

from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabPerceiverTransfer
from utils import create_train_setup, init_best_metric
from loaders import build_dataset, build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fun: Module,
    epoch: int,
    task_type,
) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        y = tf.y
        pred = model(tf)
        if pred.size(1) == 1:
            pred = pred.view(-1, )
        if task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)
        loss = loss_fun(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


def test(
    model: Module,
    loader: DataLoader,
    metric_computer: Metric,
    task_type,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()


def main(args):
    """ load pretrained TabPerceiver and finetune"""

    # build dataset
    dataset = build_dataset(task_type=args.task_type, dataset_scale=args.scale, dataset_index=args.idx)
    train_loader, valid_loader, test_loader, meta_data = build_dataloader(dataset)
    out_channels, loss_fun, metric_computer, higher_is_better, task_type = create_train_setup(dataset)
    metric_computer.to(device)

    # define model using config and load pretrained weights
    print(f"Load checkpoint from {args.pretrained_weight_path}")
    ckpt = torch.load(args.pretrained_weight_path)
    model_config = ckpt["model_config"]
    model = TabPerceiverTransfer(**model_config)
    model.load_state_dict(ckpt["model_state_dict"]) 
    model.freeze_transformer()   
    model.reconstructIO(
        out_channels=out_channels,
        num_features=meta_data["num_features"],
        col_stats=meta_data["col_stats"],
        col_names_dict=meta_data["col_names_dict"],
    )
    model.to(device)

    # train and test    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    best_val_metric, best_test_metric = init_best_metric(higher_is_better) # regression: inf, classification: 0
    print(f"higher is better: {higher_is_better}")
    print(f"default metric: {best_val_metric}, {best_test_metric}")
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, loss_fun, epoch, task_type)
        val_metric = test(model, valid_loader, metric_computer, task_type)

        if higher_is_better:
            if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test(model, test_loader, metric_computer, task_type)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, metric_computer, task_type)
        lr_scheduler.step()
            
        print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')
        # print(f'Current best_test_metric: {best_test_metric:.4F}')
    print(f"Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}")

    # save the result
    model_config["out_channels"] = out_channels
    model_config["num_features"] = meta_data["num_features"]
    model_config["col_stats"] = meta_data["col_stats"]
    model_config["col_names_dict"] = meta_data["col_names_dict"]

    checkpoint = {
        'args': args.__dict__,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "best_val_metric": best_val_metric,
        "best_test_metric": best_test_metric,
    }
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    torch.save(checkpoint, args.result_path)
    print(f"Save checkpoint into {args.result_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_type', type=str, choices=[
            'binary_classification',
            'multiclass_classification',
            'regression',
        ], default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                        default='small')
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pretrained_weight_path', type=str, default='')
    parser.add_argument('--result_path', type=str, default='')
    args = parser.parse_args()
    main(args)