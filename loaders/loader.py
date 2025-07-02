import os.path as osp
import pandas as pd
import torch
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader, Dataset, TensorFrame
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.typing import IndexSelectType

class CustomDataLoader(DataLoader):
    r"""A custom data loader which creates mini-batches and indicies 
    so that it is possible to track each instance in the memory bank.
    """

    def collate_fn(self, index: IndexSelectType) -> TensorFrame:
        return self.tensor_frame[index], torch.as_tensor(index)

def build_dataset(task_type, dataset_scale, dataset_index):
    """ 
        Build dataset from specified path
    """
    print(f"Start building {task_type} dataset_{dataset_scale}_{dataset_index}")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    dataset = DataFrameBenchmark(root=path, task_type=TaskType(task_type),
                                scale=dataset_scale, idx=dataset_index)
    dataset.materialize()
    return dataset

def build_fewshot_dataset(dataset, shots, seed):
    """
        input: train_dataset
        return: sampled train_dataset
    """
    # sample k samples for each class
    y_col = dataset.target_col
    df_train = dataset.df
    unique_classses = df_train[y_col].unique()
    sampled_dfs = []
    for target_class in unique_classses:
        sampled = df_train[df_train[y_col] == target_class].sample(n=shots, random_state=seed) 
        sampled_dfs.append(sampled)
    fewshot_train_df = pd.concat(sampled_dfs)

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
    return fewshot_dataset

def build_dataloader(
    dataset, 
    batch_size: int = 128, 
    drop_last: bool = True, 
    is_custom: bool = False, 
    use_train_dataset: bool = False
):
    """ 
        Build dataloader 
    """
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    if not is_custom:
        train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        valid_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
        test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)
    else:
        # train_loader returns (tensor_frame, index)
        train_loader = CustomDataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        valid_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
        test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    print(f'Training set has {len(train_tensor_frame)} instances')
    print(f'Validation set has {len(val_tensor_frame)} instances')
    print(f'Test set has {len(test_tensor_frame)} instances')

    col_stats = dataset.col_stats
    col_names_dict = train_tensor_frame.col_names_dict
    if not is_custom:
        meta_data = {
            "col_stats": col_stats,
            "col_names_dict": col_names_dict,
        }
    else: 
        num_samples = len(train_tensor_frame)
        meta_data = {
            "col_stats": col_stats,
            "col_names_dict": col_names_dict,
            "num_samples": num_samples,
        }

    if not use_train_dataset: 
        return train_loader, valid_loader, test_loader, meta_data
    else:
        return train_loader, valid_loader, test_loader, meta_data, train_dataset

def build_datasets(task_type, dataset_scale, num_tasks = None):
    """
        Build datasets for specified task_type and dataset_scale for multitask learning.
        Return a list of dataset.
    """
    dataset_index_ranges = {
        'binary_classification': {
            # 'small':   range(0, 2),  'medium': range(0, 9),  'large': range(0, 1),
            'small':   range(0, 14),  'medium': range(0, 9),  'large': range(0, 1),
        },
        'multiclass_classification': {
            'small':   [],            'medium': range(0, 3),  'large': range(0, 3),
        },
        'regression': {
            'small':   range(0, 13),  'medium': range(0, 6),  'large': range(0, 6),
        },
    }
    datasets = []
    if num_tasks is None:
        dataset_indices = list(dataset_index_ranges[task_type][dataset_scale])
    else:
        dataset_indices = list(range(0, num_tasks))
    for dataset_index in dataset_indices:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
        dataset = build_dataset(task_type, dataset_scale, dataset_index)
        datasets.append(dataset)

    return datasets

def build_dataloaders(datasets, batch_size=128):
    """
        Build dataloaders from given list of dataset.
        Return a list of dataloader.
    """
    train_loaders, valid_loaders, test_loaders = [], [], []
    meta_datas = {"col_stats": [], "col_names_dicts": []}     
    
    for dataset in datasets:

        dataset = dataset.shuffle()
        train_dataset, val_dataset, test_dataset = dataset.split()

        train_tensor_frame = train_dataset.tensor_frame
        val_tensor_frame = val_dataset.tensor_frame
        test_tensor_frame = test_dataset.tensor_frame

        train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
        test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

        col_stats = dataset.col_stats
        col_names_dict = train_tensor_frame.col_names_dict

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)
        meta_datas["col_stats"].append(col_stats)
        meta_datas["col_names_dicts"].append(col_names_dict)
    return train_loaders, valid_loaders, test_loaders, meta_datas