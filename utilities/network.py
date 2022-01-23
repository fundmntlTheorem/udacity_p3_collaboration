import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def layer_init(layer, limits=None):
    if not limits:
        limits = hidden_init(layer)
    layer.weight.data.uniform_(*limits)
    return layer

def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x

def to_np(t):
    return t.cpu().detach().numpy()

def empty_gate(x):
    return x

