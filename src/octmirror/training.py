"""Training loops for the PC-CGAN and for the conditional diffusion model."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time
from statistics import mean
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from octmirror.losses import hinge_d_loss, hinge_g_loss, phase_circular_loss, phase_gradient_loss
from octmirror.models.diffusion import UNET, DDPMScheduler
from octmirror.visualization import plot_training_curves, saving_img


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA, MPS) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================== PC-CGAN ============================== #

def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    """Update the exponential moving average of the model parameters (buffers are not averaged)."""
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name in model_params:
            if name in ema_params:
                ema_params[name].data.mul_(decay).add_(model_params[name].data, alpha=1 - decay)


def validate_fn(val_dl: DataLoader, G: nn.Module, device, use_ema: bool = False,
                G_ema: Optional[nn.Module] = None) -> Dict[str, float]:
    """Validation metrics averaged over batches: amplitude MAE, circular phase MAE, L1.

    If ``use_ema`` is True and ``G_ema`` is given, the EMA generator is evaluated
    instead of ``G``.
    """
    model = G_ema if use_ema and G_ema is not None else G
    model.eval()

    total_amp_mae = 0.0
    total_phase_mae = 0.0
    total_l1 = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_img, real_img, smax, smin in val_dl:
            input_img = input_img.to(device)
            real_img = real_img.to(device)

            fake_img = model(input_img)

            # Amplitude MAE
            fr, fi = fake_img[:,0], fake_img[:,1]
            rr, ri = real_img[:,0], real_img[:,1]
            amp_fake = torch.sqrt(fr**2 + fi**2 + 1e-8)
            amp_real = torch.sqrt(rr**2 + ri**2 + 1e-8)
            amp_mae = (amp_fake - amp_real).abs().mean()

            # Phase circular MAE (wrapped difference)
            phase_fake = torch.atan2(fi, fr)
            phase_real = torch.atan2(ri, rr)
            delta = phase_fake - phase_real
            delta = torch.remainder(delta + math.pi, 2*math.pi) - math.pi
            phase_mae = delta.abs().mean()

            # L1
            l1 = (fake_img - real_img).abs().mean()

            total_amp_mae += amp_mae.item()
            total_phase_mae += phase_mae.item()
            total_l1 += l1.item()
            n_batches += 1

    return {
        'amp_mae': total_amp_mae / n_batches if n_batches > 0 else 0.0,
        'phase_mae': total_phase_mae / n_batches if n_batches > 0 else 0.0,
        'l1': total_l1 / n_batches if n_batches > 0 else 0.0
    }


def _compute_amp(tensor: torch.Tensor) -> torch.Tensor:
    """Amplitude of a (B, 2, H, W) tensor whose (real, imag) channels are in [0, 1]."""
    t = tensor * 2.0 - 1.0
    amp = torch.sqrt(t[:,0]**2 + t[:,1]**2 + 1e-8)
    return amp


def train_fn(train_dl: DataLoader, G: nn.Module, D: nn.Module,
             criterion_l1: nn.Module,
             optimizer_g: optim.Optimizer, optimizer_d: optim.Optimizer,
             device,
             epoch: int = 0, num_epochs: int = 0,
             lambda_l1: float = 100.0,
             use_phase: bool = True,
             lambda_phase: float = 10.0,
             lambda_amp: float = 10.0,
             phase_weight_mode: str = 'geo',
             use_phase_grad: bool = False,
             lambda_phase_grad: float = 0.0,
             grad_clip: float = 1.0):
    """Run one PC-CGAN epoch (hinge adversarial loss).

    For every batch, the generator is updated first with
    ``adv + lambda_l1*L1 + lambda_phase*phase + lambda_amp*amp + lambda_phase_grad*phase_grad``,
    then the discriminator with the hinge loss. The discriminator is conditioned on the
    mirror-corrupted input: ``D(input, target)`` vs ``D(input, G(input))``.

    Returns:
        ``(mean_loss_g, mean_loss_d, fake, input, target, smax, smin, avg_components)``
        where the tensors come from the last batch (on CPU) and ``avg_components`` holds
        the per-batch average of each generator loss term.
    """
    G.train()
    D.train()
    total_loss_g, total_loss_d = [], []
    # accumulators for loss components
    sum_adv = 0.0
    sum_l1 = 0.0
    sum_phase = 0.0
    sum_amp = 0.0
    sum_phase_grad = 0.0
    n_batches = 0

    for i, (input_img, real_img, smax, smin) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
        input_img = input_img.to(device)
        real_img = real_img.to(device)

        # =================== GENERATOR =================== #
        fake_img = G(input_img)
        pred_fake_for_g = D(real=input_img, fake=fake_img)
        loss_g_adv = hinge_g_loss(pred_fake_for_g)

        loss_g_l1 = criterion_l1(fake_img, real_img) * lambda_l1

        # Amplitude loss: channels in [0,1] after Sigmoid are mapped back to [-1,1]
        amp_fake = _compute_amp(fake_img)
        amp_real = _compute_amp(real_img)
        loss_g_amp = F.l1_loss(amp_fake, amp_real) * lambda_amp

        if use_phase:
            loss_g_phase = phase_circular_loss(fake_img, real_img, weight_mode=phase_weight_mode) * lambda_phase
        else:
            loss_g_phase = torch.zeros(1, device=fake_img.device)

        if use_phase_grad:
            loss_g_phase_grad = phase_gradient_loss(fake_img, real_img, weight_mode=phase_weight_mode) * lambda_phase_grad
        else:
            loss_g_phase_grad = torch.zeros(1, device=fake_img.device)

        loss_g = loss_g_adv + loss_g_l1 + loss_g_phase + loss_g_amp + loss_g_phase_grad
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=grad_clip)
        optimizer_g.step()

        # =================== DISCRIMINATOR =================== #
        with torch.no_grad():
            fake_detached = fake_img.detach()
        pred_real = D(real=input_img, fake=real_img)
        pred_fake = D(real=input_img, fake=fake_detached)

        loss_d = hinge_d_loss(pred_real, pred_fake)

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip)
        optimizer_d.step()

        total_loss_g.append(loss_g.item())
        sum_adv += loss_g_adv.item()
        sum_l1 += loss_g_l1.item()
        sum_phase += loss_g_phase.item()
        sum_amp += loss_g_amp.item()
        sum_phase_grad += loss_g_phase_grad.item()
        n_batches += 1
        total_loss_d.append(loss_d.item())

    if n_batches > 0:
        avg_components = dict(
            adv=sum_adv / n_batches,
            l1=sum_l1 / n_batches,
            phase=sum_phase / n_batches,
            amp=sum_amp / n_batches,
            phase_grad=sum_phase_grad / n_batches
        )
    else:
        avg_components = dict(adv=0.0, l1=0.0, phase=0.0, amp=0.0, phase_grad=0.0)

    return (mean(total_loss_g),
            mean(total_loss_d),
            fake_img.detach().cpu(),
            input_img.detach().cpu(),
            real_img.detach().cpu(),
            smax.detach().cpu(),
            smin.detach().cpu(),
            avg_components)


def saving_model(D: nn.Module, G: nn.Module, e: int, savePath: str) -> None:
    """Save ``weight/G{e+1}.pth`` and ``weight/D{e+1}.pth``."""
    os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
    torch.save(G.state_dict(), os.path.join(savePath, f"weight/G{str(e+1)}.pth"))
    torch.save(D.state_dict(), os.path.join(savePath, f"weight/D{str(e+1)}.pth"))


def train_loop(train_dl: DataLoader,
               val_dl: DataLoader,
               G: nn.Module,
               D: nn.Module,
               num_epoch: int,
               savePath: str,
               device,
               lr: float = 5e-5,
               dlr: float = 1e-5,
               betas: Tuple[float, float] = (0.5, 0.999),
               lambda_l1: float = 100.0,
               use_phase: bool = True,
               lambda_phase: float = 10.0,
               lambda_amp: float = 10.0,
               phase_weight_mode: str = 'geo',
               use_phase_grad: bool = False,
               lambda_phase_grad: float = 0.0,
               use_ema: bool = True,
               ema_decay: float = 0.999,
               grad_clip: float = 1.0,
               checkpoint_every: int = 5,
               scheduler_patience: int = 10,
               scheduler_factor: float = 0.5,
               scheduler_min_lr: float = 1e-7,
               image_log_every: int = 25):
    """Train the PC-CGAN.

    Per epoch: one pass of :func:`train_fn`, one EMA update of the generator
    parameters (if ``use_ema``), validation with :func:`validate_fn` (on the EMA
    generator when ``use_ema``), a ``ReduceLROnPlateau`` step of the generator learning
    rate on the validation amplitude MAE, and a row in ``metrics_epoch.csv``.
    ``weight/G_best.pth`` / ``weight/D_best.pth`` hold the raw (non-EMA) networks at the
    epoch with the lowest validation amplitude MAE. See docs/KNOWN_ISSUES.md.

    Returns:
        ``(G, D, best_amp_mae, best_epoch)``.
    """
    G.to(device)
    D.to(device)
    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=dlr, betas=betas)
    criterion_l1 = nn.L1Loss()

    # Learning rate scheduler (reduces LR when amp_mae plateaus)
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=scheduler_factor, patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )

    # EMA model
    G_ema = None
    if use_ema:
        G_ema = copy.deepcopy(G).to(device)
        G_ema.eval()
        for param in G_ema.parameters():
            param.requires_grad = False

    os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)

    total_loss_d, total_loss_g = [], []
    val_amp_maes, val_phase_maes, val_l1s = [], [], []
    train_advs, train_l1s, train_phases, train_amps, train_phase_grads = [], [], [], [], []
    result = {}

    csv_path = os.path.join(savePath, 'metrics_epoch.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'loss_g', 'loss_d',
                'train_adv', 'train_l1', 'train_phase', 'train_amp', 'train_phase_grad',
                'val_amp_mae', 'val_phase_mae', 'val_l1'
            ])

    best_amp_mae = float('inf')
    best_epoch = -1

    for e in range(num_epoch):
        loss_g, loss_d, fake_img, input_img, real_img, smax, smin, comps = train_fn(
            train_dl, G, D,
            criterion_l1,
            optimizer_g, optimizer_d,
            device,
            epoch=e, num_epochs=num_epoch,
            lambda_l1=lambda_l1,
            use_phase=use_phase,
            lambda_phase=lambda_phase,
            lambda_amp=lambda_amp,
            phase_weight_mode=phase_weight_mode,
            use_phase_grad=use_phase_grad,
            lambda_phase_grad=lambda_phase_grad,
            grad_clip=grad_clip
        )
        total_loss_d.append(loss_d)
        total_loss_g.append(loss_g)
        train_advs.append(comps['adv'])
        train_l1s.append(comps['l1'])
        train_phases.append(comps['phase'])
        train_amps.append(comps['amp'])
        train_phase_grads.append(comps['phase_grad'])

        # Update EMA (once per epoch, see docs/KNOWN_ISSUES.md)
        if use_ema and G_ema is not None:
            update_ema(G_ema, G, decay=ema_decay)

        # Validation
        val_metrics = validate_fn(val_dl, G, device, use_ema=use_ema, G_ema=G_ema)
        val_amp_maes.append(val_metrics['amp_mae'])
        val_phase_maes.append(val_metrics['phase_mae'])
        val_l1s.append(val_metrics['l1'])

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                e + 1,
                loss_g,
                loss_d,
                comps['adv'],
                comps['l1'],
                comps['phase'],
                comps['amp'],
                comps['phase_grad'],
                val_metrics['amp_mae'],
                val_metrics['phase_mae'],
                val_metrics['l1']
            ])

        # Scheduler step (based on validation amp MAE)
        scheduler_g.step(val_metrics['amp_mae'])

        # Save images sparsely to reduce I/O pressure
        if (e + 1) % image_log_every == 0 or e == 0 or e == num_epoch - 1:
            try:
                saving_img(fake_img, input_img, real_img, smax, smin, e, savePath)
            except Exception as ex:
                print(f"Warning: saving_img failed at epoch {e+1}: {ex}")

        # Periodic checkpoint (every N epochs)
        if (e + 1) % checkpoint_every == 0:
            saving_model(D, G, e, savePath)
            if use_ema and G_ema is not None:
                os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
                torch.save(G_ema.state_dict(), os.path.join(savePath, f"weight/G_ema{str(e+1)}.pth"))

        # Save best model based on validation amp MAE
        if val_metrics['amp_mae'] < best_amp_mae:
            best_amp_mae = val_metrics['amp_mae']
            best_epoch = e + 1
            os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
            torch.save(G.state_dict(), os.path.join(savePath, "weight/G_best.pth"))
            torch.save(D.state_dict(), os.path.join(savePath, "weight/D_best.pth"))
            print(f"  --> New best amp_mae: {best_amp_mae:.6f} at epoch {best_epoch}")

        result["loss_d"] = total_loss_d
        result["loss_g"] = total_loss_g
        result["val_amp_mae"] = val_amp_maes
        result["val_phase_mae"] = val_phase_maes
        result["val_l1"] = val_l1s
        result["train_adv"] = train_advs
        result["train_l1"] = train_l1s
        result["train_phase"] = train_phases
        result["train_amp"] = train_amps
        result["train_phase_grad"] = train_phase_grads

        print(f"Epoch {e+1}/{num_epoch} "
              f"- G: {loss_g:.4f} (adv={comps['adv']:.4f}, l1={comps['l1']:.4f}, phase={comps['phase']:.4f}, amp={comps['amp']:.4f}, phase_grad={comps['phase_grad']:.4f}) "
              f"- D: {loss_d:.4f} | Val: amp_mae={val_metrics['amp_mae']:.6f}, phase_mae={val_metrics['phase_mae']:.4f}, l1={val_metrics['l1']:.4f}")

    # Final save
    saving_model(D, G, num_epoch-1, savePath)
    if use_ema and G_ema is not None:
        os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
        torch.save(G_ema.state_dict(), os.path.join(savePath, "weight/G_ema_final.pth"))

    try:
        plot_training_curves(result, savePath)
    except Exception as ex:
        print(f"Error plotting curves: {ex}")
    print(f"Model Saved Successfully. Best amp_mae: {best_amp_mae:.6f} at epoch {best_epoch}")
    return G, D, best_amp_mae, best_epoch


# ============================== Diffusion ============================== #

DIFFUSION_CSV_HEADER = ['epoch', 'loss_total', 'mse_noise', 'phase_loss', 'seconds_per_epoch',
                        'samples_per_second', 'peak_gpu_memory_gb', 'experiment_id', 'roi_size',
                        'batch_size', 'lr']


def check_diffusion_overwrite(savePath: str, allow_overwrite: bool = False) -> None:
    """Refuse to reuse a directory that already holds a diffusion run.

    Raises:
        RuntimeError: If ``savePath`` contains ``checkpoint.pth`` or
            ``training_metrics.csv`` and ``allow_overwrite`` is False.
    """
    existing_checkpoint = os.path.join(savePath, 'checkpoint.pth')
    existing_metrics = os.path.join(savePath, 'training_metrics.csv')
    if not allow_overwrite and (os.path.exists(existing_checkpoint) or os.path.exists(existing_metrics)):
        raise RuntimeError(
            f"savePath '{savePath}' already contains a checkpoint.pth or training_metrics.csv "
            f"and allow_overwrite is False. Refusing to overwrite a previous experiment."
        )


def write_run_metadata(savePath: str, metadata: dict) -> None:
    """Write ``run_metadata.json`` (status ``running`` / ``ok`` / ``failed``)."""
    with open(os.path.join(savePath, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def write_config(savePath: str, config: dict) -> None:
    """Write ``config.json`` with the resolved run configuration."""
    with open(os.path.join(savePath, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def train_diffusion(train_ds: Dataset,
                    savePath: str,
                    device,
                    batch_size: int = 1,
                    num_time_steps: int = 1000,
                    num_epochs: int = 15,
                    seed: int = -1,
                    ema_decay: float = 0.9999,
                    lr: float = 2e-5,
                    use_phase_loss: bool = False,
                    lambda_phase_diff: float = 0.0,
                    experiment_id: int = 0,
                    roi_size: int = 256,
                    num_workers: int = 2,
                    data_parallel: bool = True) -> None:
    """Train the conditional diffusion U-Net (noise prediction, MSE).

    Adam, mixed precision (``GradScaler``, enabled on CUDA devices), optional
    ``DataParallel`` when several GPUs are visible, and an EMA (``timm`` ``ModelEmaV3``)
    updated after every optimizer step. If ``use_phase_loss``, the circular phase loss
    (``'geo'`` weighting) between the predicted x0 and the clean target is added with
    weight ``lambda_phase_diff``. Per-epoch metrics go to ``training_metrics.csv``; the
    final ``checkpoint.pth`` holds ``{'weights', 'optimizer', 'ema'}``.

    Args:
        train_ds: Dataset of (condition, target) pairs (2, roi_size, roi_size).
        savePath: Output directory.
        device: Torch device.
        seed: Seed for :func:`set_seed`; ``-1`` draws a random seed.
        num_workers: DataLoader workers.
        data_parallel: Wrap the model in ``nn.DataParallel`` if more than one GPU is visible.
    """
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=num_workers)

    device = torch.device(device)
    scheduler = DDPMScheduler(num_time_steps=num_time_steps, device=device)
    model = UNET()

    if data_parallel and device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    # mixed precision scaler (automatic)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = nn.MSELoss(reduction='mean')

    metrics_path = os.path.join(savePath, 'training_metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(DIFFUSION_CSV_HEADER)

    n_samples = len(train_ds)
    checked_batch_shape = False
    for i in range(num_epochs):
        total_loss = 0
        total_mse_noise = 0
        total_phase_loss = 0
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()
        with tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}") as pbar:
            for condition, x in pbar:
                if not checked_batch_shape:
                    expected_shape = (x.shape[0], 2, roi_size, roi_size)
                    if tuple(condition.shape) != expected_shape or tuple(x.shape) != expected_shape:
                        raise ValueError(
                            f"Expected input/target batches of shape {expected_shape}, "
                            f"got condition={tuple(condition.shape)}, target={tuple(x.shape)}"
                        )
                    checked_batch_shape = True
                x = x.to(device)
                condition = condition.to(device)
                current_batch_size = x.shape[0]
                t = torch.randint(0, num_time_steps, (current_batch_size,))
                e = torch.randn_like(x, requires_grad=False)
                a = scheduler.alpha[t].view(current_batch_size, 1, 1, 1).to(x.device)
                x0_clean = x.clone()
                x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

                # Mixed precision forward/backward to reduce memory
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    output = model(x_noisy, t, condition)
                    mse_noise = criterion(output, e)
                    if use_phase_loss:
                        x0_pred = (x_noisy - torch.sqrt(1 - a) * output) / torch.sqrt(a)
                        phase_loss = phase_circular_loss(x0_pred, x0_clean, weight_mode='geo')
                        loss = mse_noise + lambda_phase_diff * phase_loss
                    else:
                        phase_loss = torch.zeros((), device=mse_noise.device)
                        loss = mse_noise
                total_loss += loss.item()
                total_mse_noise += mse_noise.item()
                total_phase_loss += phase_loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
        epoch_seconds = time.time() - epoch_start
        samples_per_second = n_samples / epoch_seconds if epoch_seconds > 0 else 0.0
        peak_gpu_memory_gb = (torch.cuda.max_memory_allocated() / 1024**3) if use_amp else 0.0
        avg_loss = total_loss / len(train_loader)
        avg_mse_noise = total_mse_noise / len(train_loader)
        avg_phase_loss = total_phase_loss / len(train_loader)
        print(f'Epoch {i+1} | Loss {avg_loss:.5f} '
              f'| mse_noise {avg_mse_noise:.5f} '
              f'| phase_loss {avg_phase_loss:.5f} '
              f'| sec/epoch {epoch_seconds:.2f} '
              f'| samples/sec {samples_per_second:.2f} '
              f'| peak_gpu_mem_gb {peak_gpu_memory_gb:.2f}')
        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([i+1, avg_loss, avg_mse_noise, avg_phase_loss, epoch_seconds,
                                    samples_per_second, peak_gpu_memory_gb, experiment_id, roi_size,
                                    batch_size, lr])

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }

    torch.save(checkpoint, os.path.join(savePath, 'checkpoint.pth'))
    print(f"Model saved to {savePath}")
