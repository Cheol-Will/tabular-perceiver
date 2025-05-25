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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch_frame import NAStrategy, stype
from torch_frame.typing import TaskType
from torch_frame.data.stats import StatType
from torch_frame.data import DataLoader, Dataset
from models import TabPerceiverSemi
from loaders import build_dataset, build_dataloader
from utils import create_train_setup, init_best_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# split train_dataset into few-shot dataset and unlabeled dataset
def split_fewshot_and_unlabeled(dataset, shots, seed):
    """
    input: train_dataset
    return: (fewshot_dataset, rest_dataset)
    """
    # sample k samples for each class
    y_col = dataset.target_col
    df_train = dataset.df
    unique_classes = df_train[y_col].unique()

    sampled_dfs = []
    sampled_idx = []

    for target_class in unique_classes:
        class_df = df_train[df_train[y_col] == target_class]
        sampled = class_df.sample(n=shots, random_state=seed)
        sampled_dfs.append(sampled)
        sampled_idx.extend(sampled.index.tolist())

    fewshot_train_df = pd.concat(sampled_dfs)
    rest_df = df_train.drop(index=sampled_idx)

    fewshot_dataset = Dataset(
        df=fewshot_train_df,
        col_to_stype=dataset.col_to_stype,
        target_col=dataset.target_col,
        split_col=dataset.split_col,
        col_to_sep=dataset.col_to_sep,
        col_to_text_embedder_cfg=dataset.col_to_text_embedder_cfg,
        col_to_text_tokenizer_cfg=dataset.col_to_text_tokenizer_cfg,
        col_to_image_embedder_cfg=dataset.col_to_image_embedder_cfg,
        col_to_time_format=dataset.col_to_time_format
    )
    fewshot_dataset.materialize()

    rest_dataset = Dataset(
        df=rest_df,
        col_to_stype=dataset.col_to_stype,
        target_col=dataset.target_col,
        split_col=dataset.split_col,
        col_to_sep=dataset.col_to_sep,
        col_to_text_embedder_cfg=dataset.col_to_text_embedder_cfg,
        col_to_text_tokenizer_cfg=dataset.col_to_text_tokenizer_cfg,
        col_to_image_embedder_cfg=dataset.col_to_image_embedder_cfg,
        col_to_time_format=dataset.col_to_time_format
    )
    rest_dataset.materialize()
    return fewshot_dataset, rest_dataset

def create_semi_supervised_setup(col_stats, col_names_dict):
    # create loss_fn, metric_computer for all input feature
    loss_fn_list, metric_computer_list, task_type_list = [], [], []
    for key_stype, col_names in col_names_dict.items(): 
        if key_stype == stype.numerical:
            loss_fn_list += [MSELoss() for _ in col_names]
            metric_computer_list += [MeanSquaredError(squared=False).to(device) for _ in col_names]
            task_type_list += [TaskType.REGRESSION for _ in col_names]
            # higher_is_better = False
        elif key_stype == stype.categorical:
            num_classes_list = []
            for col_name in col_names:
                stats = col_stats[col_name]
                num_classes_list.append(len(stats[StatType.COUNT][0]))
            loss_fn_list += [CrossEntropyLoss() for _ in num_classes_list]
            metric_computer_list += [Accuracy(task='multiclass', num_classes=num_classes).to(device) for num_classes in num_classes_list]
            task_type_list += [TaskType.MULTICLASS_CLASSIFICATION for _ in num_classes_list]
            # higher_is_better = True
        else:
            print(f"Unsupported type {key_stype}")  
    return loss_fn_list, metric_computer_list, task_type_list


def sample_target_cols(
    loader: DataLoader,
    col_names: list[str], 
):
    input_col_idx_list, target_col_names_list = [], []
    num_features = len(col_names)
    for _ in range(len(loader)):
        # For each batch, sample target columns
        num_target_cols = random.randint(int(0.1*num_features), int(0.2*num_features))
        num_target_cols = min(1, num_target_cols) # sample at least one target column
        target_col_idx = random.sample(range(0, num_features), num_target_cols) 

        # Filter input columns
        input_col_idx = [i for i in range(num_features) if i not in target_col_idx]
        target_col_names = [col_names[i] for i in target_col_idx]
        
        input_col_idx_list.append(input_col_idx)
        target_col_names_list.append(target_col_names)

    return input_col_idx_list, target_col_names_list

def pretrain(
    args, 
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler, 
    loss_fn_list: list[Module],
    metric_computer_list: list[Metric],
    task_type_list: list[TaskType],
    col_names: list[str],
    col_stats: dict[str, dict[StatType, Any]],
    writer: SummaryWriter,
    seed: int
):
    """Pretrain on unlabeled dataset."""
    history = {
        "train_loss": [],
        "train_metrics": [],
    }

    for epoch in range(args.epochs):
        model.train()
        # reset metric computers
        for metric_computer in metric_computer_list:
            metric_computer.reset()
        
        input_col_idx_list, target_col_names_list = sample_target_cols(
            loader=loader,
            col_names=col_names 
        )
        total_loss = 0.0
        num_samples = 0

        for idx, tf in tqdm(enumerate(loader), desc=f"Epoch: {epoch+1}"):
            tf = tf.to(device)
            B, F = tf.num_rows, tf.num_cols

            input_col_idx, target_col_names = input_col_idx_list[idx], target_col_names_list[idx]
            out = model(tf, input_col_idx) # model uses only specified column
            loss = 0.0
            for i, target_col_name in enumerate(target_col_names):
                pred = out[i]
                if pred.size(1) == 1:
                    pred = pred.view(-1) # (batch_size) for mse
                
                y = tf.get_col_feat(target_col_name) # (batch_size, 1)
                y = y.view(-1) # (batch_size)
                
                # normalize numerical target 
                if task_type_list[col_names.index(target_col_name)] == TaskType.REGRESSION:
                    stats = col_stats[target_col_name]
                    y = (y - stats[StatType.MEAN]) / (stats[StatType.STD] + 1e-6) # 

                # get loss_fn and metric_computer
                loss_fn = loss_fn_list[col_names.index(target_col_name)]
                metric_computer = metric_computer_list[col_names.index(target_col_name)]
                loss += loss_fn(pred, y) * y.size(0)
                num_samples += y.size(0)

                if task_type_list[col_names.index(target_col_name)] == TaskType.MULTICLASS_CLASSIFICATION:
                    pred = pred.argmax(dim=-1) # (batch_size)
                elif task_type_list[col_names.index(target_col_name)] == TaskType.REGRESSION and pred.ndim > 1:
                    pred = pred.view(-1) # (batch_size, 1) -> (batch_size)
                metric_computer.update(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
       
        # total loss and metrics, note that some column may not be included 
        train_loss = total_loss/num_samples
        train_metrics = np.array([metric_computer.compute().item() for metric_computer in metric_computer_list]) 
        scheduler.step()
        
        # Log
        history["train_loss"].append(train_loss.item())
        history["train_metrics"].append(train_metrics)
        
        # Log
        print(f"Train Loss: {train_loss}")
        for idx, metric_computer in enumerate(metric_computer_list):
            if task_type_list[idx] == TaskType.MULTICLASS_CLASSIFICATION:
                print(f"{'ACC'.ljust(8)}|", end=' ')
            elif task_type_list[idx] == TaskType.REGRESSION:
                print(f"{'MSE'.ljust(8)}|", end=' ')
            else:
                print(f"{'???'.ljust(8)}|", end=' ')
        print()  
        for metric in train_metrics:
            print(f"{round(float(metric), 4):<8}|", end=' ')
        print()

        # tensorboard write
        writer.add_scalar(f"Seed{seed}/Pretrain/Loss", train_loss.item(), epoch)
        for i, metric in enumerate(train_metrics):
            writer.add_scalar(f"Seed{seed}/Pretrain/Metric_{i}", metric, epoch)


    return history    


def train(
    args, 
    model: Module,
    loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler, 
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
    writer: SummaryWriter,
    seed: int,
):
    """Finetune on target."""
    model.train()
    history = {
        "train_loss": [],
        "train_metrics": [],
        "test_loss": [],
        "test_metrics": [],
    }
    for epoch in range(args.finetune_epochs):
        total_loss = 0.0
        total_samples = 0
        metric_computer.reset()
        for tf in loader:
            tf = tf.to(device)
            y = tf.y
            if task_type == TaskType.BINARY_CLASSIFICATION:
                y = y.float()

            preds = model(tf)
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
        scheduler.step()
        train_loss = total_loss / total_samples
        train_metric = metric_computer.compute().item()

        print(f"[Finetune] Epoch {epoch+1}: Loss = {train_loss:.4f}, Metric = {train_metric:.4f}")
        history["train_loss"].append(train_loss)
        history["train_metrics"].append(train_metric)
        writer.add_scalar(f"Seed{seed}/Finetune/Loss", train_loss, epoch)
        writer.add_scalar(f"Seed{seed}/Finetune/Metric", train_metric, epoch)

        if ((epoch+1) % 5 == 0) or (epoch in [0, 1, 2, 3]):
            test_loss, test_metric = eval(
                model=model,
                loader=test_loader,
                loss_fn=loss_fn,
                metric_computer=metric_computer,
                task_type=task_type,
            )
            print(f"[Test]: Loss = {test_loss:.4f}, Metric = {test_metric:.4f}")
            history["test_loss"].append(test_loss)
            history["test_metrics"].append(test_metric)
            writer.add_scalar(f"Seed{seed}/Eval/Test_Loss", test_loss, epoch)
            writer.add_scalar(f"Seed{seed}/Eval/Test_Metric", test_metric, epoch)


    return history

def eval(
    model: Module,
    loader: DataLoader,
    loss_fn: Module,
    metric_computer: Metric,
    task_type: TaskType,
):
    model.eval()
    metric_computer.reset()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for tf in loader:
            tf = tf.to(device)
            y = tf.y
            preds = model(tf)
            
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
            metric_computer.update(preds, y)
    eval_loss = total_loss / num_samples
    eval_metric = metric_computer.compute().item()
    
    return eval_loss, eval_metric 

def main(args):
    print(args)
    log_dir = f"runs/{args.task_type}_{args.scale}_idx{args.idx}_{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # finetune and evaluate on new task
    dataset = build_dataset(task_type=args.task_type, dataset_scale=args.scale, dataset_index=args.idx)
    num_classes, loss_fn, metric_computer, higher_is_better, task_type = create_train_setup(dataset)
    metric_computer = metric_computer.to(device)
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()

    # print(f"Debug: {dataset.df['split'].value_counts()}")
    # print(f"Debug: {train_dataset.df['split'].value_counts()}")
    # print(f"Debug: {val_dataset.df['split'].value_counts()}")
    # print(f"Debug: {test_dataset.df['split'].value_counts()}")

    test_loader = DataLoader(test_dataset.tensor_frame, batch_size=128)
    col_stats = train_dataset.col_stats
    col_names_dict = train_dataset.tensor_frame.col_names_dict
    for idx, (k, v) in enumerate(col_stats.items()):
        print(f"{idx}: {k}")
        print(v)
    # pretrain setup
    loss_fn_list, metric_computer_list, task_type_list = create_semi_supervised_setup(
            col_stats, 
            col_names_dict
    )    

    # use meta data from original train_dataset, not unlabeled_train_dataset
    meta_data = {
        "col_stats": col_stats,
        "col_names_dict": col_names_dict,
    }
    model_config = {
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "num_latents": args.num_latents,
        "hidden_dim": args.hidden_dim,
        "mlp_ratio": args.mlp_ratio,
        "dropout_prob": args.dropout_prob,
    }
    # get col_names to access target values during pretrain
    col_names = []
    for v in col_names_dict.values():
        col_names += v

    seed_list = range(0,30)
    history = []
    for seed in seed_list:
        # define model for each seed
        model = TabPerceiverSemi(
            **model_config,
            **meta_data,
            num_classes=num_classes,
        )    
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.95) 

        # split original train dataset into fewshot and unlabeld datasest
        fewshot_train_dataset, unlabeled_train_dataset = split_fewshot_and_unlabeled(train_dataset, args.shots, seed=seed)
        unlabeled_train_loader = DataLoader(unlabeled_train_dataset.tensor_frame, batch_size=128, shuffle=True, drop_last=True)
        fewshot_train_loader = DataLoader(fewshot_train_dataset.tensor_frame, batch_size=128, shuffle=True, drop_last=False)

        # pretrain with unlabeled dataset
        pretrain_history = pretrain(
            args, 
            model=model,
            loader=unlabeled_train_loader,
            optimizer=optimizer,
            scheduler=scheduler, 
            loss_fn_list=loss_fn_list,
            metric_computer_list=metric_computer_list,
            task_type_list=task_type_list,
            col_names=col_names,
            col_stats=col_stats,
            writer=writer,
            seed=seed,
        )
        print(pretrain_history)

        # finetune with fewshot dataset
        model.freeze()
        model.to(device)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.shape}")
        # print(fewshot_train_dataset.df.shape)
        # print(fewshot_train_dataset.df)
        # redefine optimizer and scheduler since it shows better performance
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_finetune)
        scheduler = ExponentialLR(optimizer, gamma=0.95) 
        finetune_history = train(
            args=args,
            model=model,
            loader=fewshot_train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler, 
            loss_fn=loss_fn,
            metric_computer=metric_computer,
            task_type=task_type,
            writer=writer,
            seed=seed,            
        )
        for k, v in finetune_history.items():
            print(f"{k}: {v}")
        # eval
        # test_loss, test_metric = eval(
        #     model=model,
        #     loader=test_loader,
        #     loss_fn=loss_fn,
        #     metric_computer=metric_computer,
        #     task_type=task_type,
        #     writer=writer,
        #     seed=seed,            
        # )
        

        seed_history = {
            "pretrain": pretrain_history,
            "finetune": finetune_history,
            "meta": {
                "col_names": col_names,
                "task_type_list": task_type_list,
                "args": vars(args),
            }
        }
        history.append(seed_history)

    path = f"output/{args.task_type}/{args.scale}/{args.idx}/{args.exp_name}.pt"
    torch.save(history, path)
    for hist in history:
        print(hist["finetune"]["test_metrics"])

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tabular Perceiver Semi-supervised Learning Script")
    parser.add_argument('--task_type', type=str,
                        choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
    parser.add_argument('--finetune_epochs', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='TabPerceiverSemi_')
    # config
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_latents', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--mlp_ratio', type=float, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_finetune', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    main(args)