"""Dataset split helpers."""

import os

import numpy as np
from sklearn.model_selection import train_test_split


def make_split_indices(y, seed, split):
    train_ratio, val_ratio, test_ratio = split
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {split}")

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=y,
        random_state=seed,
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=y[train_val_idx],
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def save_split_indices(split_dir, seed, train_idx, val_idx, test_idx):
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, f"seed_{seed}_train.npy"), train_idx)
    np.save(os.path.join(split_dir, f"seed_{seed}_val.npy"), val_idx)
    np.save(os.path.join(split_dir, f"seed_{seed}_test.npy"), test_idx)


def class_ratio(y, idx):
    labels = y[idx]
    pos = int(labels.sum())
    total = len(labels)
    return {"total": total, "pos": pos, "neg": total - pos, "pos_ratio": pos / max(total, 1)}

