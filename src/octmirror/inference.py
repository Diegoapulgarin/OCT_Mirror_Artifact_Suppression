"""Loading trained models and reconstructing complex B-scans."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from octmirror.models.diffusion import UNET, DDPMScheduler
from octmirror.models.pccgan import UNetGenerator
from octmirror.preprocessing import inverse_log_scale, log_scale

DIFFUSION_MAX_LOG_AMP = 30.0


# ============================== PC-CGAN ============================== #

def load_pccgan(weights: str, device) -> UNetGenerator:
    """Load a PC-CGAN generator ``state_dict`` (e.g. ``weight/G_best.pth``) in eval mode."""
    G = UNetGenerator(in_ch=2, out_ch=2)
    state_dict = torch.load(weights, map_location=device)
    G.load_state_dict(state_dict)
    G.to(device)
    G.eval()
    return G


def run_pccgan(G: nn.Module, input_cx: np.ndarray, smax: np.ndarray, smin: np.ndarray,
               device) -> np.ndarray:
    """Reconstruct one complex B-scan with the PC-CGAN generator.

    The input is log-scaled with its own limits, passed through ``G``, and the output is
    de-normalized with the given ``smax`` / ``smin`` (no clipping). In the paper's
    evaluation protocol these are the clean target's limits.

    Args:
        G: Generator in eval mode.
        input_cx: Mirror-corrupted complex B-scan (Z, X); Z and X must be multiples of 128.
        smax: Maximum log10 amplitude, shape (1, 1, 1).
        smin: Minimum log10 amplitude, shape (1, 1, 1).
        device: Torch device of ``G``.

    Returns:
        Reconstructed complex B-scan (Z, X).
    """
    logslices_in, _, _, _, _ = log_scale(input_cx[np.newaxis, ...])
    input_tensor = torch.tensor(
        logslices_in[0].transpose(2, 0, 1), dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_img = G(input_tensor)  # (1, 2, Z, X)

    fake_np = fake_img[0].detach().cpu().numpy().transpose(1, 2, 0)[np.newaxis, ...]  # (1, Z, X, 2)
    pred_rec = inverse_log_scale(fake_np, smax, smin, max_log_amp=None)
    return pred_rec[0, :, :, 0] + 1j * pred_rec[0, :, :, 1]


# ============================== Diffusion ============================== #

def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str = 'module.') -> "OrderedDict[str, torch.Tensor]":
    """Remove a leading ``prefix`` from every key if all keys carry it."""
    keys = list(state_dict.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return OrderedDict((k[len(prefix):], v) for k, v in state_dict.items())
    return OrderedDict(state_dict)


def load_diffusion(checkpoint: str, device, use_ema: bool = True) -> UNET:
    """Load the diffusion U-Net from a ``checkpoint.pth`` written by ``train_diffusion``.

    Args:
        checkpoint: Path to a dict with keys ``weights``, ``optimizer``, ``ema``.
        device: Torch device.
        use_ema: Load the EMA weights (as the original sampler does) instead of the raw
            weights.

    The ``module.`` prefix added by ``nn.DataParallel`` is removed from ``weights``.
    ``ema`` is the ``state_dict`` of ``timm.utils.ModelEmaV3``, whose keys carry one
    ``module.`` prefix from the EMA wrapper and a second one if the model was wrapped in
    ``DataParallel``; both are removed.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if use_ema:
        state_dict = _strip_prefix(_strip_prefix(ckpt['ema']))
    else:
        state_dict = _strip_prefix(ckpt['weights'])
    model = UNET()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def sample_ddpm(model: nn.Module, condition: torch.Tensor, device,
                num_time_steps: int = 1000, seed: Optional[int] = 0,
                scheduler: Optional[DDPMScheduler] = None) -> torch.Tensor:
    """Ancestral DDPM sampling conditioned on a mirror-corrupted ROI.

    Same reverse update as the original training script's sampler, starting from
    Gaussian noise; only the final sample is returned.

    Args:
        model: Diffusion U-Net in eval mode.
        condition: Normalized condition (B, 2, H, W) with H and W equal to the training
            ROI size.
        device: Torch device of ``model``.
        num_time_steps: Number of diffusion steps (1000 in all experiments).
        seed: Passed to ``torch.manual_seed`` before drawing any noise; ``None`` keeps
            the current RNG state.
        scheduler: Noise schedule; built with ``num_time_steps`` if None.

    Returns:
        Normalized sample (B, 2, H, W) on CPU.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if scheduler is None:
        scheduler = DDPMScheduler(num_time_steps=num_time_steps, device=device)
    with torch.no_grad():
        z = torch.randn(condition.shape[0], 2, condition.shape[2], condition.shape[3]).to(device)  # initial noise (real, imag)
        condition = condition.to(device)
        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.tensor([t], device=z.device)
            temp = (scheduler.beta[t_tensor] / ((torch.sqrt(1 - scheduler.alpha[t_tensor])) * (torch.sqrt(1 - scheduler.beta[t_tensor]))))
            z = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor]))) * z - (temp * model(z, t_tensor, condition))
            e = torch.randn_like(z)
            z = z + (e * torch.sqrt(scheduler.beta[t_tensor]))

        # last step
        t_tensor = torch.tensor([0], device=z.device)
        temp = scheduler.beta[t_tensor] / ((torch.sqrt(1 - scheduler.alpha[t_tensor])) * (torch.sqrt(1 - scheduler.beta[t_tensor])))
        x = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor]))) * z - (temp * model(z, t_tensor, condition))
    return x.cpu()


def run_diffusion(model: nn.Module, input_roi_cx: np.ndarray, smax: np.ndarray, smin: np.ndarray,
                  device, num_time_steps: int = 1000, seed: Optional[int] = 0) -> np.ndarray:
    """Reconstruct one complex ROI with the diffusion model.

    The input ROI is log-scaled with its own limits and used as condition; the sample is
    de-normalized with the given ``smax`` / ``smin`` and the log-amplitude clip at 30.

    Returns:
        Reconstructed complex ROI (H, W).
    """
    input_norm, _, _, _, _ = log_scale(input_roi_cx[np.newaxis, ...])
    condition = torch.tensor(input_norm[0].transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    sample = sample_ddpm(model, condition, device, num_time_steps=num_time_steps, seed=seed)
    sample_np = sample[0].numpy().transpose(1, 2, 0)[np.newaxis, ...]
    rec = inverse_log_scale(sample_np, smax, smin, max_log_amp=DIFFUSION_MAX_LOG_AMP)
    return rec[0, :, :, 0] + 1j * rec[0, :, :, 1]


def scale_limits(cx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-slice log10-amplitude limits ``(smax, smin)`` of one complex B-scan, shape (1, 1, 1)."""
    _, smax, smin, _, _ = log_scale(cx[np.newaxis, ...])
    return smax, smin


def evaluate_diffusion(*args, **kwargs):
    """Evaluation of the diffusion ablations (D1-D4) on the validation set.

    Not part of this release: only training and inference code are provided for the
    diffusion model.
    """
    raise NotImplementedError  # TODO: pending the authors' diffusion evaluation code
