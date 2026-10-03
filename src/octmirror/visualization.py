"""Figures: training snapshots, training curves, and evaluation plots.

All functions save figures to disk with the non-interactive Agg backend.
"""
from __future__ import annotations

import os
import traceback
from typing import Callable, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from octmirror.preprocessing import inverse_log_scale  # noqa: E402


def phase_correlation(slices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Phase difference between neighbouring pixels of a B-scan (ex ``Correlation``).

    Args:
        slices: Array (Z, X, 2) with (real, imag) channels.

    Returns:
        ``(correlation_x, correlation_y)``: wrapped phase differences along axis 0
        (shape (Z-1, X)) and along axis 1 (shape (Z, X-1)).
    """
    slices = slices[:,:,0] + 1j*slices[:,:,1]

    correlationy = np.angle(slices[:,1:] * np.conjugate(slices[:,:-1]))
    correlationx = np.angle(slices[1:, :] * np.conjugate(slices[:-1, :]))
    return correlationx, correlationy


def saving_img(fake_img, input_img, real_img, smax, smin, e: int, savePath: str) -> None:
    """Save amplitude (dB) and phase-difference snapshots of the first sample of a batch.

    The three images (prediction, input, target) are de-normalized with the target's
    ``smax`` / ``smin`` (no clipping, as in the PC-CGAN pipeline) and written to
    ``savePath/generated``.
    """
    if fake_img.shape[0] > 1:
        fake_img = fake_img[0:1]
        input_img = input_img[0:1]
        real_img = real_img[0:1]
        smax = smax[0:1]
        smin = smin[0:1]
    os.makedirs(os.path.join(savePath, "generated"), exist_ok=True)

    def recover_and_save(img_tensor, smax, smin, name):
        if img_tensor.ndim == 4:
            img_tensor = img_tensor.squeeze(0)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (z, x, 2)
        img_np = img_np[np.newaxis, ...]  # (1, z, x, 2)
        if isinstance(smax, torch.Tensor):
            smax = smax.cpu().numpy()
        if isinstance(smin, torch.Tensor):
            smin = smin.cpu().numpy()
        if smax.ndim == 1:
            smax = smax[:, np.newaxis, np.newaxis]
            smin = smin[:, np.newaxis, np.newaxis]
        img_rec = inverse_log_scale(img_np, smax, smin, max_log_amp=None)
        img_rec = img_rec[0]
        intImage = 20 * np.log10(np.abs(img_rec[..., 0] + 1j * img_rec[..., 1]))
        cx, cy = phase_correlation(img_rec)
        return intImage, cx, cy

    intImageFake, cx, cy = recover_and_save(fake_img, smax, smin, "fake")
    intImageInput, cx_input, cy_input = recover_and_save(input_img, smax, smin, "input")
    intImageReal, cx_real, cy_real = recover_and_save(real_img, smax, smin, "real")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(intImageFake, cmap='gray')
    axs[0].set_title("Fake Image")
    axs[0].axis("off")
    axs[1].imshow(intImageInput, cmap='gray')
    axs[1].set_title("Input Image")
    axs[1].axis("off")
    axs[2].imshow(intImageReal, cmap='gray')
    axs[2].set_title("Real Image")
    axs[2].axis("off")
    fig.savefig(os.path.join(savePath, "generated", f"epoch_{e+1}.png"))
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cx, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[0].set_title("Correlation Fake Image")
    axs[0].axis("off")
    axs[1].imshow(cx_input, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Correlation Input Image")
    axs[1].axis("off")
    axs[2].imshow(cx_real, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[2].set_title("Correlation Real Image")
    axs[2].axis("off")
    fig.savefig(os.path.join(savePath, "generated", f"correlation_x_epoch_{e+1}.png"))
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cy, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[0].set_title("Correlation Fake Image")
    axs[0].axis("off")
    axs[1].imshow(cy_input, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Correlation Input Image")
    axs[1].axis("off")
    axs[2].imshow(cy_real, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[2].set_title("Correlation Real Image")
    axs[2].axis("off")
    fig.savefig(os.path.join(savePath, "generated", f"correlation_y_epoch_{e+1}.png"))
    plt.close(fig)


def plot_training_curves(result: Mapping[str, Sequence[float]], savePath: str) -> None:
    """Plot and save training/validation curves of all loss components."""
    epochs = np.arange(1, len(result['loss_g']) + 1)
    has_phase_grad = 'train_phase_grad' in result and len(result['train_phase_grad']) == len(epochs)

    if has_phase_grad:
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Generator and Discriminator total loss
    axes[0, 0].plot(epochs, result['loss_g'], label='Generator', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, result['loss_d'], label='Discriminator', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Train components
    axes[0, 2].plot(epochs, result['train_adv'], label='Adversarial', color='orange')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Train Adversarial Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(epochs, result['train_l1'], label='L1', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Train L1 Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, result['train_phase'], label='Phase', color='purple')
    axes[1, 1].plot(epochs, result['val_phase_mae'], label='Val Phase MAE', color='purple', linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss / MAE')
    axes[1, 1].set_title('Phase Loss (Train) & Val MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[1, 2].plot(epochs, result['train_amp'], label='Amp (train)', color='brown')
    axes[1, 2].plot(epochs, result['val_amp_mae'], label='Val Amp MAE', color='brown', linestyle='--')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss / MAE')
    axes[1, 2].set_title('Amplitude Loss (Train) & Val MAE')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    if has_phase_grad:
        axes[0, 3].plot(epochs, result['train_phase_grad'], label='PhaseGrad', color='black')
        axes[0, 3].set_xlabel('Epoch')
        axes[0, 3].set_ylabel('Loss')
        axes[0, 3].set_title('Train Phase Gradient Loss')
        axes[0, 3].legend()
        axes[0, 3].grid(True)

        axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(savePath, 'training_curves.png'), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {os.path.join(savePath, 'training_curves.png')}")


def save_amplitude_phase_png(cx: np.ndarray, amplitude_path: str, phase_path: str,
                             title: Optional[str] = None) -> None:
    """Save the amplitude in dB (``20*log10|cx|``) and the phase of a complex B-scan as PNGs."""
    intensity = 20 * np.log10(np.abs(cx) + 1e-12)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(intensity, cmap='gray')
    fig.colorbar(im, ax=ax, label='Amplitude (dB)')
    ax.set_title(f"{title} - amplitude" if title else "Amplitude")
    ax.axis('off')
    fig.savefig(amplitude_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.angle(cx), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im, ax=ax, label='Phase (rad)')
    ax.set_title(f"{title} - phase" if title else "Phase")
    ax.axis('off')
    fig.savefig(phase_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


TRAINING_CURVE_METRICS = ['loss_g', 'val_amp_mae', 'val_phase_mae', 'train_phase', 'train_phase_grad', 'train_amp']


def build_training_curve_plots(ablation_root: str, config_names: Sequence[str], out_dir: str,
                               metrics: Sequence[str] = TRAINING_CURVE_METRICS,
                               log_error: Callable[[str], None] = print) -> None:
    """One figure per training metric overlaying every configuration's ``metrics_epoch.csv``."""
    os.makedirs(out_dir, exist_ok=True)
    for metric in metrics:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            for config_name in config_names:
                csv_path = os.path.join(ablation_root, config_name, 'metrics_epoch.csv')
                if not os.path.exists(csv_path):
                    log_error(f"Missing metrics_epoch.csv for {config_name}, skipping in {metric} plot.")
                    continue
                df = pd.read_csv(csv_path)
                if metric not in df.columns:
                    log_error(f"Column {metric} missing in {csv_path}, skipping.")
                    continue
                ax.plot(df['epoch'], df[metric], label=config_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'Unified training curve: {metric}')
            ax.legend(fontsize=8)
            ax.grid(True)
            out_path = os.path.join(out_dir, f'unified_{metric}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
        except Exception as ex:
            log_error(f"Failed to build unified training curve plot for {metric}: {ex}\n{traceback.format_exc(limit=5)}")


def build_metric_boxplots(per_bscan_df: pd.DataFrame, config_names: Sequence[str], out_dir: str,
                          metrics: Sequence[str], checkpoint_type: str = 'G_best',
                          log_error: Callable[[str], None] = print) -> None:
    """One box plot per evaluation metric across configurations (per-B-scan values)."""
    os.makedirs(out_dir, exist_ok=True)
    for metric in metrics:
        try:
            data_per_config = []
            labels = []
            for config_name in config_names:
                subset = per_bscan_df[
                    (per_bscan_df['config'] == config_name) &
                    (per_bscan_df['checkpoint_type'] == checkpoint_type)
                ]
                values = subset[metric].dropna().values
                data_per_config.append(values)
                labels.append(config_name)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data_per_config, tick_labels=labels, showmeans=True)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} across configurations ({checkpoint_type})')
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            out_path = os.path.join(out_dir, f'boxplot_{metric}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
        except Exception as ex:
            log_error(f"Failed to build boxplot for {metric}: {ex}\n{traceback.format_exc(limit=5)}")
