"""Loss functions: hinge adversarial losses and circular phase losses.

Phase losses take tensors (B, 2, H, W) with channels (real, imag) and compute the phase
as ``atan2(imag, real)`` of the channels as given (see docs/KNOWN_ISSUES.md about the
[0, 1] representation used during training).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PHASE_WEIGHT_MODES = ('geo', 'none')


def hinge_d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss of the discriminator. ``pred_*``: (B, 1, H, W) patch scores."""
    return F.relu(1 - pred_real).mean() + F.relu(1 + pred_fake).mean()


def hinge_g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss of the generator."""
    return -pred_fake.mean()


def _check_weight_mode(weight_mode: str) -> None:
    if weight_mode not in PHASE_WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {PHASE_WEIGHT_MODES}, got {weight_mode!r}")


def phase_circular_loss(fake_img: torch.Tensor, real_img: torch.Tensor,
                        weight_mode: str = 'geo') -> torch.Tensor:
    """Amplitude-weighted circular phase loss.

    ``L = sum(w * (1 - cos(dphi))) / sum(w)`` with ``dphi`` wrapped to [-pi, pi].

    Args:
        fake_img: Prediction (B, 2, H, W), channels (real, imag).
        real_img: Reference (B, 2, H, W), channels (real, imag).
        weight_mode: ``'geo'``: ``w = sqrt(|fake| * |real|)``; ``'none'``: ``w = 1``.
    """
    _check_weight_mode(weight_mode)
    fr, fi = fake_img[:,0], fake_img[:,1]
    rr, ri = real_img[:,0], real_img[:,1]
    phase_fake = torch.atan2(fi, fr)
    phase_real = torch.atan2(ri, rr)
    delta = phase_fake - phase_real
    # wrap to [-pi, pi]
    delta = torch.remainder(delta + math.pi, 2*math.pi) - math.pi
    amp_f = torch.sqrt(fr**2 + fi**2 + 1e-8)
    amp_r = torch.sqrt(rr**2 + ri**2 + 1e-8)
    if weight_mode == 'geo':
        w = torch.sqrt(amp_f * amp_r + 1e-8)
    else:
        w = torch.ones_like(delta)
    loss = (w * (1 - torch.cos(delta))).sum() / (w.sum() + 1e-8)
    return loss


def phase_gradient_loss(fake_img: torch.Tensor, real_img: torch.Tensor,
                        weight_mode: str = 'geo') -> torch.Tensor:
    """Differentiable phase-gradient constraint (axial + lateral).

    Compares finite phase differences along H (axial) and W (lateral) with a circular
    ``1 - cos`` penalty on their wrapped difference, weighted as in
    :func:`phase_circular_loss`, and averages the two directions.

    Args:
        fake_img: Prediction (B, 2, H, W), channels (real, imag).
        real_img: Reference (B, 2, H, W), channels (real, imag).
        weight_mode: ``'geo'`` or ``'none'``.
    """
    _check_weight_mode(weight_mode)
    fr, fi = fake_img[:,0], fake_img[:,1]
    rr, ri = real_img[:,0], real_img[:,1]

    phase_fake = torch.atan2(fi, fr)
    phase_real = torch.atan2(ri, rr)

    # gradients in axial (z/H) and lateral (x/W)
    grad_ax_fake = phase_fake[:, 1:, :] - phase_fake[:, :-1, :]
    grad_ax_real = phase_real[:, 1:, :] - phase_real[:, :-1, :]
    grad_lat_fake = phase_fake[:, :, 1:] - phase_fake[:, :, :-1]
    grad_lat_real = phase_real[:, :, 1:] - phase_real[:, :, :-1]

    # wrapped gradient differences
    delta_ax = grad_ax_fake - grad_ax_real
    delta_ax = torch.remainder(delta_ax + math.pi, 2*math.pi) - math.pi
    delta_lat = grad_lat_fake - grad_lat_real
    delta_lat = torch.remainder(delta_lat + math.pi, 2*math.pi) - math.pi

    amp_f = torch.sqrt(fr**2 + fi**2 + 1e-8)
    amp_r = torch.sqrt(rr**2 + ri**2 + 1e-8)

    if weight_mode == 'geo':
        w = torch.sqrt(amp_f * amp_r + 1e-8)
    else:
        w = torch.ones_like(phase_fake)

    w_ax = w[:, 1:, :]
    w_lat = w[:, :, 1:]

    loss_ax = (w_ax * (1 - torch.cos(delta_ax))).sum() / (w_ax.sum() + 1e-8)
    loss_lat = (w_lat * (1 - torch.cos(delta_lat))).sum() / (w_lat.sum() + 1e-8)
    return 0.5 * (loss_ax + loss_lat)
