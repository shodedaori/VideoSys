import os
import pickle
import torch

from videosys.core.parallel_mgr import initialize
from videosys.core.mp_utils import get_distributed_init_method, get_open_port


def get_compute_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def tensor_check(x, y):
    device = x.device
    error = torch.max(torch.abs(x - y))
    if device.type == 'cpu':
        assert torch.equal(x, y), f"Equal check: the error is {error}"
    else:
        assert torch.allclose(x, y, atol=1e-6), f"Allclose check: the error is {error}"


def init_env():
    distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
    initialize(rank=0, world_size=1, init_method=distributed_init_method)


def save_obj(obj, path, name):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f)
