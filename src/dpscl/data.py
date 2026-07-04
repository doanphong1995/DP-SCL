"""Data loading, seeding, and DataLoader helpers."""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_full_temporal_data(input_dir, ds_config):
    npz_path = os.path.join(input_dir, "datastore", ds_config["npz_filename"])
    data = np.load(npz_path)
    x = np.concatenate([data["t_data"], data["v_data"]], axis=0).astype(np.float32)
    y = np.concatenate([data["t_label"], data["v_label"]], axis=0).astype(np.int64)
    expected_suffix = (ds_config["week_count"], ds_config["days_per_week"], ds_config["activity_num"])
    if x.ndim != 4 or x.shape[1:] != expected_suffix:
        raise ValueError(f"Expected data shape (N,{expected_suffix[0]},{expected_suffix[1]},{expected_suffix[2]}), got {x.shape}")
    return x, y, npz_path


def make_loaders(x, y, train_idx, val_idx, test_idx, batch_size, num_workers):
    x_flat = x.reshape(x.shape[0], -1)
    tensors = {
        "train": (torch.from_numpy(x_flat[train_idx]).float(), torch.from_numpy(y[train_idx]).float()),
        "val": (torch.from_numpy(x_flat[val_idx]).float(), torch.from_numpy(y[val_idx]).float()),
        "test": (torch.from_numpy(x_flat[test_idx]).float(), torch.from_numpy(y[test_idx]).float()),
    }
    train_loader = DataLoader(TensorDataset(*tensors["train"]), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(TensorDataset(*tensors["val"]), batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset(*tensors["test"]), batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

