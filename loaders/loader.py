import os.path as osp
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark

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


def build_dataloader(dataset, batch_size=128):
    """ 
        Build dataloader 
    """
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame

    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_tensor_frame, batch_size=batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size)

    print(f'Training set has {len(train_tensor_frame)} instances')
    print(f'Validation set has {len(val_tensor_frame)} instances')
    print(f'Test set has {len(test_tensor_frame)} instances')

    col_stats = dataset.col_stats
    col_names_dict = train_tensor_frame.col_names_dict
    num_features = 0
    for k, v in col_names_dict.items():
        num_features += len(v)   

    meta_data = {
        "col_stats": col_stats,
        "col_names_dict": col_names_dict,
        "num_features": num_features,
    }

    return train_loader, valid_loader, test_loader, meta_data



def build_datasets(task_type, dataset_scale):
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
    dataset_indices = list(dataset_index_ranges[task_type][dataset_scale])
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