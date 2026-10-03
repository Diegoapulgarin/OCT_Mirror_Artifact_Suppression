"""PyTorch datasets of paired (mirror-corrupted, clean) complex B-scans.

Both datasets store pre-normalized arrays (H, W, 2) and return float32 tensors (2, H, W)
with channels (real, imag).
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from octmirror.preprocessing import (
    extract_center_roi,
    load_tomogram,
    log_scale,
    mirror_artifact,
    select_bscan_indices,
)


class PairedComplexDatasetWithScale(Dataset):
    """PC-CGAN dataset: (input, target) pairs plus the target's log-amplitude limits.

    Args:
        data: List of tuples (input, target), each array (H, W, 2).
        max_min: List of tuples (smax, smin) per sample, used to invert the
            normalization of the target (see :func:`octmirror.preprocessing.inverse_log_scale`).
    """

    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]],
                 max_min: List[Tuple[np.ndarray, np.ndarray]]):
        self.data = data
        self.max_min = max_min

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        input_array, output_array = self.data[idx]
        max_val, min_val = self.max_min[idx]

        # Convert to tensors with 2 channels (real, imag)
        input_tensor = torch.tensor(input_array.transpose(2, 0, 1), dtype=torch.float32)
        output_tensor = torch.tensor(output_array.transpose(2, 0, 1), dtype=torch.float32)

        return input_tensor, output_tensor, max_val, min_val

    def __len__(self) -> int:
        return len(self.data)


class PairedComplexDataset(Dataset):
    """Diffusion dataset: (input, target) pairs of normalized complex ROIs.

    Args:
        data: List of tuples (input, target), each array (H, W, 2).
    """

    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]]):
        self.data = data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_array, output_array = self.data[idx]
        # Channels (real, imag) already exist in the last axis; move to (2, H, W)
        input_tensor = torch.tensor(input_array.transpose(2, 0, 1), dtype=torch.float32)
        output_tensor = torch.tensor(output_array.transpose(2, 0, 1), dtype=torch.float32)
        return input_tensor, output_tensor

    def __len__(self) -> int:
        return len(self.data)


def build_pccgan_datasets(
    data_root: str,
    folders: Sequence[str],
    files: Sequence[str],
    val_fraction: float = 0.1,
    split_seed: Optional[int] = 0,
) -> Tuple[PairedComplexDatasetWithScale, PairedComplexDatasetWithScale]:
    """Build the PC-CGAN train/validation datasets as in the original training script.

    For every listed file in every folder: the mirror artifact is simulated on the full
    tomogram along Z, both volumes are transposed to (Y, Z, X) so that axis 0 indexes
    B-scans, and each B-scan is log-scaled independently. The target's per-B-scan
    (smax, smin) are kept. B-scans of all volumes are pooled and split at random
    (per B-scan, not per volume) into train/validation.

    Files are visited in ``os.listdir`` order, as in the original script.

    Args:
        data_root: Root directory containing ``folders``.
        folders: Sub-folders to scan.
        files: File names (``.npy``) to include; other files are ignored.
        val_fraction: Fraction of B-scans assigned to validation (``int`` truncation).
        split_seed: Seed of the random permutation used for the split. ``None``
            reproduces the original unseeded ``random.shuffle``.

    Returns:
        ``(train_ds, val_ds)``.
    """
    tomList = []
    tomccList = []
    maxData = []
    minData = []
    nfile = 0
    for folder in tqdm(folders):
        folder_files = os.listdir(os.path.join(data_root, folder))
        for file in folder_files:
            if file not in files:
                continue
            tom = load_tomogram(os.path.join(data_root, folder, file), verbose=False)  # (Z, X, Y)
            tomcc = mirror_artifact(tom)
            tom = np.transpose(tom, (2, 0, 1))  # (Y, Z, X)
            tomcc = np.transpose(tomcc, (2, 0, 1))  # (Y, Z, X)
            logslicesData, slicesmaxData, slicesminData, _, _ = log_scale(tom)
            logslicesCxData, _, _, _, _ = log_scale(tomcc)
            tomList.append(logslicesData)
            tomccList.append(logslicesCxData)
            maxData.append(slicesmaxData)
            minData.append(slicesminData)
            nfile += 1
    print(f'number of files loaded: {nfile}')
    tomTarget = np.concatenate(tomList, axis=0)
    tomccInput = np.concatenate(tomccList, axis=0)
    maxTarget = np.concatenate(maxData, axis=0)
    minTarget = np.concatenate(minData, axis=0)
    del tomList, tomccList, maxData, minData
    print(f'len of dataset total: {tomTarget.shape[0]}')
    pairs = []
    for i in range(tomTarget.shape[0]):
        pairs.append((tomccInput[i], tomTarget[i]))
    del tomTarget, tomccInput
    max_min = []
    for i in range(maxTarget.shape[0]):
        max_min.append((maxTarget[i, :, :], minTarget[i, :, :]))
    del maxTarget, minTarget

    total_samples = len(pairs)
    val_size = int(val_fraction * total_samples)
    train_size = total_samples - val_size

    indices = list(range(total_samples))
    if split_seed is None:
        random.shuffle(indices)
    else:
        # Same permutation as random.seed(split_seed); random.shuffle(indices),
        # without touching the global random state.
        random.Random(split_seed).shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = [pairs[i] for i in train_indices]
    val_data = [pairs[i] for i in val_indices]
    train_max_min = [max_min[i] for i in train_indices]
    val_max_min = [max_min[i] for i in val_indices]
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    return (PairedComplexDatasetWithScale(train_data, train_max_min),
            PairedComplexDatasetWithScale(val_data, val_max_min))


def build_diffusion_dataset(
    data_root: str,
    folders: Sequence[str],
    roi_size: int = 256,
    n_bscans_per_volume: int = 40,
) -> PairedComplexDataset:
    """Build the diffusion training dataset as in the original training script.

    Every ``.npy`` file in every folder is used. For each volume, up to
    ``n_bscans_per_volume`` evenly spaced B-scans are taken; the central
    ``roi_size`` x ``roi_size`` ROI is cropped at native resolution, the mirror
    artifact is simulated on the ROI itself (axis 0 == Z), and input and target are
    log-scaled independently.

    Args:
        data_root: Root directory containing ``folders``.
        folders: Sub-folders to scan.
        roi_size: Side of the central ROI.
        n_bscans_per_volume: Maximum number of B-scans per volume.

    Returns:
        The training dataset (there is no validation split for diffusion).
    """
    n_volumes = 0
    n_rois = 0
    pairs = []
    for folder in tqdm(folders):
        folder_files = os.listdir(os.path.join(data_root, folder))
        for file in folder_files:
            if not file.endswith('.npy'):
                continue
            tom = load_tomogram(os.path.join(data_root, folder, file), verbose=False)  # (Z, X, Y)
            n_volumes += 1
            y_indices = select_bscan_indices(tom.shape[2], n_bscans_per_volume)
            for y_idx in y_indices:
                bscan_complex = tom[:, :, y_idx]  # (Z, X), native resolution
                target_roi = extract_center_roi(bscan_complex, roi_size=roi_size)
                # Mirror artifact applied directly on the complex ROI, axis 0 == Z
                input_roi = mirror_artifact(target_roi)
                target_norm, _, _, _, _ = log_scale(target_roi[np.newaxis, ...])
                input_norm, _, _, _, _ = log_scale(input_roi[np.newaxis, ...])
                pairs.append((input_norm[0], target_norm[0]))
                n_rois += 1
    print(f'Total volumes loaded: {n_volumes}')
    print(f'Total B-scans/ROIs: {n_rois}')
    return PairedComplexDataset(pairs)
