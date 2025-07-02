import os
from collections import Counter
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_finetune_result(
    dir_name : str = "output",
):
    task_types = ["binary_classification", "regression", "multiclass_classification"]
    scale_types = ["small", "medium", "large"]
    model_type = "TabPerceiver"
    result_list = []
    for task_type in task_types:
        for scale_type in scale_types:
            path = os.path.join(dir_name, f"{task_type}/{scale_type}/{model_type}.pt")
            if os.path.exists(path):
                print(f"Load {path}")
                result = torch.load(path, weights_only=False)
                result_list.append(result)

    return result_list

def load_result_multitask(
    dir_name: str = "output",
    task_type: str = None,
):
    if task_type is None:
        task_types = ["binary_classification", "regression", "multiclass_classification"]
    else:
        task_types = [task_type]
    scale_types = ["small", "medium", "large"]
    model_types = [
        "TPMT_250509_config_d4_0",
        "TPMT_250509_config_d6_5",
    ]
    dataset_index_ranges = {
        'binary_classification': {
            'small':   range(0, 14),  'medium': range(0, 9),  'large': range(0, 1),
        },
        'multiclass_classification': {
            'small':   [],            'medium': range(0, 3),  'large': range(0, 3),
        },
        'regression': {
            'small':   range(0, 13),  'medium': range(0, 6),  'large': range(0, 6),
        },
    }

    result_dict = {model_type : [] for model_type in model_types}
    for task_type in task_types:
        for model_type in model_types:
            for scale_type in scale_types:
                index_list = dataset_index_ranges[task_type][scale_type]        

                for idx in index_list:
                    path = os.path.join(dir_name, f"{task_type}/{scale_type}/{idx}/{model_type}.pt")

                    if os.path.exists(path):
                        result = torch.load(path, weights_only=False)
                        result_dict[model_type].append({
                            "task_type": task_type,
                            "scale_type": scale_type,
                            "idx": idx,
                            "model_type": model_type,
                            "train_losses": result["train_history"]["train_losses"],
                            "train_metrics": result["train_history"]["train_metrics"],
                            "val_losses": result["train_history"]["val_losses"],
                            "val_metrics": result["train_history"]["val_metrics"],
                            "finetune_train_losses": result["finetune_history"]["finetune_train_loss"],
                            "finetune_train_metrics": result["finetune_history"]["finetune_train_acc"],
                            "finetune_val_losses": result["finetune_history"]["finetune_valid_loss"],
                            "finetune_val_metrics": result["finetune_history"]["finetune_valid_acc"],
                            "best_test_metric": result["best_test_metric"],
                            "best_test_before_finetune": result["best_test_before_finetune"],
                        })

    return result_dict


def load_result(
    dir_name : str = "output", 
    task_type : str = None,
    scale_type : str = None,
):  
    if task_type is None:
        task_types = ["binary_classification", "regression", "multiclass_classification"]
    else:
        task_types = [task_type]
    if scale_type is None:
        scale_types = ["small", "medium", "large"]
    else:
        scale_types = [scale_type]
    model_types = [
        "LightGBM",
        # "TabNet", 
        "FTTransformer", 
        "ResNet", 
        # "TabTransformer", 
        "Trompt", 
        "ExcelFormer",
        "LinearL1",
        "TabPerceiver", 
        "MemPerceiver",
        "MemGlovalAvgPool",
        # "MemPerceiver_ens",
        # "MemPerceiver_Ens_Attn",
        # "TabPerceiverMultiTask_1",
        # "TabPerceiverMultiTaskMOE_1",
    ]

    # model_types += [f"TPMT_250509_config_d4_{i}" for i in range(12)]
    # model_types += [f"TPMT_250509_config_d6_{i}" for i in range(12)]
    # model_types += [f"TPMOE_250512_config_{i}" for i in range(12)]
    
    dataset_index_ranges = {
        'binary_classification': {
            'small':   range(0, 14),  'medium': range(0, 9),  'large': range(0, 1),
        },
        'multiclass_classification': {
            'small':   [],            'medium': range(0, 3),  'large': range(0, 3),
        },
        'regression': {
            'small':   range(0, 13),  'medium': range(0, 6),  'large': range(0, 6),
        },
    }

    result_dict = {model_type : [] for model_type in model_types}
    for task_type in task_types:
        for model_type in model_types:
            for scale_type in scale_types:
                index_list = dataset_index_ranges[task_type][scale_type]        

                for idx in index_list:
                    path = os.path.join(dir_name, f"{task_type}/{scale_type}/{idx}/{model_type}.pt")

                    if os.path.exists(path):
                        result = torch.load(path, weights_only=False)
                        if model_type == "LightGBM":
                            result_dict[model_type].append({
                                "task_type": task_type,
                                "scale_type": scale_type,
                                "idx": idx,
                                "model_type": model_type,
                                "best_test_metric": result["best_test_metric"],
                                "best_cfg": result["best_cfg"],
                            })
                        else:
                            try: 
                                result_dict[model_type].append({
                                    "task_type": task_type,
                                    "scale_type": scale_type,
                                    "idx": idx,
                                    "model_type": model_type,
                                    "best_test_metric": result["best_test_metric"],
                                    "best_model_cfg": result["best_model_cfg"],
                                    "best_train_cfg": result["best_train_cfg"],
                                })
                            except:
                                result_dict[model_type].append({
                                    "task_type": task_type,
                                    "scale_type": scale_type,
                                    "idx": idx,
                                    "model_type": model_type,
                                    "best_test_metric": result["best_test_metric"],
                                })

    return result_dict

def plot_result(
    result_dict: dict[str, list[dict]], 
    task_type: str,
    file_name: str = None,
):
    # Flatten
    rows = []
    for model_type, entries in result_dict.items():
        for entry in entries:
            dataset_index = f"{entry['task_type']}-{entry['scale_type']}-{entry['idx']}"
            rows.append({
                "dataset_index": dataset_index,
                "model_type": model_type,
                "best_test_metric": entry["best_test_metric"]
            })
    result_df = pd.DataFrame(rows)

    # Pivot
    pivot_df = result_df.pivot(index="dataset_index", columns="model_type", values="best_test_metric")
    if "LightGBM" not in pivot_df.columns:
        raise ValueError("LightGBM is not included in result_dict.")

    # Prepare plot
    plt.figure(figsize=(9, 9))
    x = np.linspace(0.45, 1.05, 500)  # 확장된 x 범위
    plt.fill_between(x, x, 1.05, color='blue', alpha=0.1)  # 위쪽 삼각형
    plt.fill_between(x, 0.45, x, color='red', alpha=0.1)   # 아래쪽 삼각형
    plt.plot([0.45, 1.05], [0.45, 1.05], 'r--')            # 대각선 기준선

    for model_type in pivot_df.columns:
        if model_type == "LightGBM":
            continue

        if model_type == "TabPerceiver":
            color = "red"
        elif model_type == "MemPerceiver":
            color = "blue" 
        elif "MultiTaskMOE" in model_type:
            color = "blue"
        elif "MultiTask" in model_type:
            color = "green"
        else:
            color = None          
        if "Perceiver" in model_type:
            s = 360
            alpha = 1.0
        else:
            s = 240  
            alpha = 0.5
        plt.scatter(
            pivot_df[model_type],
            pivot_df["LightGBM"],
            label=model_type,
            s=s,
            alpha=alpha,
            color=color, 
            marker="X",
        )

    plt.xlim(0.45, 1.05)
    plt.ylim(0.45, 1.05)
    plt.text(0.52, 0.97, "LightGBM Better", fontsize=28, fontweight='bold', color='blue')
    plt.text(0.72, 0.53, "Deep model Better", fontsize=28, fontweight='bold', color='red')
    plt.xlabel("ROC-AUC for deep tabular models", fontsize=32)
    plt.ylabel("ROC-AUC for LightGBM", fontsize=32)
    plt.title(f"Deep Models vs LightGBM across {task_type} Task", fontsize=20)
    plt.legend(loc='center right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    if file_name is None:
        out_path = f"output/plots/deep_vs_lightgbm_{task_type}.png"
    else:
        out_path = f"output/plots/{file_name}.png"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved plot to {out_path}")

def plot_multitask_history(result_dict, model_key, file_name):
    result = result_dict[model_key]
    num_tasks = len(result)
    total_subplots = num_tasks + 2  # task + total loss + total metric

    fig, axes = plt.subplots(1, total_subplots, figsize=(6 * total_subplots, 5), squeeze=False)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # loss and metric color for separate task plot
    loss_color = 'tab:blue'
    metric_color = 'tab:orange'

    for task_id in range(num_tasks):
        ax = axes[0, task_id]
        history = result[task_id]

        train_losses = history["train_losses"]
        train_metrics = history["train_metrics"]
        val_losses = history["val_losses"]
        val_metrics = history["val_metrics"]

        epochs = list(range(len(train_losses)))

        ax.plot(epochs, train_losses, label="Train Loss", linestyle='-', color=loss_color)
        ax.plot(epochs, val_losses, label="Val Loss", linestyle='--', color=loss_color)
        ax.plot(epochs, train_metrics, label="Train Metric", linestyle='-', color=metric_color)
        ax.plot(epochs, val_metrics, label="Val Metric", linestyle='--', color=metric_color)

        ax.set_title(f"Task {task_id}", fontsize=24)
        ax.set_xlabel("Epoch", fontsize=24)
        ax.set_ylabel("Value", fontsize=24)
        ax.legend()
        ax.grid(True)

        # total loss and total metric
        color = color_cycle[task_id % len(color_cycle)]
        axes[0, num_tasks].plot(epochs, train_losses, linestyle='-', color=color, label=f"Task {task_id} Train")
        axes[0, num_tasks].plot(epochs, val_losses, linestyle='--', color=color, label=f"Task {task_id} Val")

        axes[0, num_tasks + 1].plot(epochs, train_metrics, linestyle='-', color=color, label=f"Task {task_id} Train")
        axes[0, num_tasks + 1].plot(epochs, val_metrics, linestyle='--', color=color, label=f"Task {task_id} Val")

    axes[0, num_tasks].set_title("All Tasks - Loss")
    axes[0, num_tasks].set_xlabel("Epoch")
    axes[0, num_tasks].set_ylabel("Loss")
    axes[0, num_tasks].grid(True)

    axes[0, num_tasks + 1].set_title("All Tasks - Metric")
    axes[0, num_tasks + 1].set_xlabel("Epoch")
    axes[0, num_tasks + 1].set_ylabel("Metric")
    axes[0, num_tasks + 1].grid(True)

    plt.tight_layout()

    os.makedirs("output/plots", exist_ok=True)
    out_path = f"output/plots/{file_name}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")




def save_result_dataframe(
    result_dict: dict[str, list[dict]],
    task_type: str = None,
    file_name: str = None,
):
    rows = []
    for model_type, entries in result_dict.items():
        for entry in entries:
            rows.append({
                "task_type": entry["task_type"],
                "scale_type": entry["scale_type"],
                "idx": entry["idx"],
                "model_type": model_type,
                "best_test_metric": entry["best_test_metric"]
            })
    result_df = pd.DataFrame(rows)

    # Save full results
    # out_path = f"output/metric/leaderboard_{task_type}.csv"
    if file_name is None:
        out_path = f"output/metric/leaderboard.csv"
    else:
        out_path = f"output/metric/{file_name}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"Saved leaderboard to {out_path}")

def plot_hyperparameter_distribution(
    result_dict: dict[str, list[dict]],
    model_type: str,
    path_name: str = None,
):
    entries = result_dict.get(model_type, [])

    # Define search spaces
    model_search_space = {
        'num_heads': [4, 8],
        'num_layers': [4, 6, 8],
        'num_latents': [4, 8, 16, 32],
        'hidden_dim': [32, 64, 128, 256],
        'mlp_ratio': [0.25, 0.5, 1, 2, 4],
        'dropout_prob': [0.0, 0.2],
    }
    train_search_space = {
        'batch_size': [128, 256],
        'base_lr': [0.0001, 0.001],
        'gamma_rate': [0.9, 0.95, 1.0],
    }

    all_keys = list(model_search_space) + list(train_search_space)
    n_cols   = 3
    n_rows   = len(all_keys) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols//2, 4*n_rows))
    axes = axes.flatten()

    for idx, key in enumerate(all_keys):
        ax = axes[idx]
        if key in model_search_space:
            values = [run['best_model_cfg'].get(key) for run in entries]
            categories = model_search_space[key]
        else:
            values = [run['best_train_cfg'].get(key) for run in entries]
            categories = train_search_space[key]

        sns.countplot(x=values, order=categories, ax=ax)
        ax.set_title(key)
        ax.set_xlabel(key)
        ax.set_ylabel("Frequency")

    # Hide any extra subplots
    for ax in axes[len(all_keys):]:
        ax.set_visible(False)

    fig.suptitle(f"Hyperparameter Distributions for {model_type}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if path_name is None:
        path = f"output/plots/hyperparam_distribution_{model_type}.png"
    else:
        path = f"output/plots/hyperparam_distribution_{model_type}_{path_name}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"Saved plot to {path}")