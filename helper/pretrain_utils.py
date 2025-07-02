import torch

def load_pretrain_weigth(path):
    checkpoint = torch.load(path)

    return checkpoint