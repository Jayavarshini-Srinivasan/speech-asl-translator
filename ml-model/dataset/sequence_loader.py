import os
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_hand(seq):
    """
    Normalize hand landmarks to be invariant to position and scale.

    Steps:
      1. Translate all landmarks relative to the wrist (landmark 0)
      2. Scale by the L2-norm of the landmark vector (proxy for hand size)

    Parameters
    ----------
    seq : np.ndarray, shape (T, 63)

    Returns
    -------
    np.ndarray, shape (T, 63)
    """
    s = seq.reshape(-1, 21, 3)          # (T, 21, 3)
    wrist = s[:, 0:1, :]               # (T, 1, 3)
    s = s - wrist                       # translate to wrist origin
    flat = s.reshape(-1, 63)           # (T, 63)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6  # (T, 1)
    flat = flat / norm
    return flat                         # (T, 63)


def add_velocity(seq):
    """
    Append frame-to-frame velocity (delta) to each frame.

    Parameters
    ----------
    seq : np.ndarray, shape (T, 63)

    Returns
    -------
    np.ndarray, shape (T, 126)  — 63 coords + 63 deltas
    """
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]       # first frame delta stays zero
    return np.concatenate([seq, vel], axis=1)   # (T, 126)


class ASLDataset(Dataset):

    def __init__(self, data_dir, augment=True):

        self.samples = []
        self.labels = []
        self.label_map = {}
        self.augment = augment

        label_names = sorted(os.listdir(data_dir))

        for idx, label in enumerate(label_names):

            self.label_map[label] = idx

            label_dir = os.path.join(data_dir, label)

            if not os.path.isdir(label_dir):
                continue

            for file in os.listdir(label_dir):

                if file.endswith(".npy"):

                    path = os.path.join(label_dir, file)

                    self.samples.append(path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x = np.load(self.samples[idx])  # (30, 63)

        # --- Feature engineering ---
        x = normalize_hand(x)           # (30, 63) — position/scale invariant
        x = add_velocity(x)             # (30, 126) — add motion information

        # --- Augmentation (training only) ---
        if self.augment:
            # Small additive noise on the coordinate part
            noise = np.random.normal(0, 0.005, x.shape)
            x = x + noise

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx])

        return x, y