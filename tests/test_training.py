import copy
import csv
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from legacy_loader import load_legacy
from octmirror.datasets import PairedComplexDataset, PairedComplexDatasetWithScale
from octmirror.models.pccgan import PatchGANDiscriminator, UNetGenerator
from octmirror.training import (
    DIFFUSION_CSV_HEADER,
    set_seed,
    train_diffusion,
    train_fn,
    train_loop,
    update_ema,
    validate_fn,
)

GAN = "torchPCCGAN_AblationM2_M8.py"
DIFF = "diffComplexField_ROI256_D1_D4.py"


def _gan_dataset(n=4, size=128, seed=0):
    rng = np.random.default_rng(seed)
    data = [(rng.uniform(size=(size, size, 2)), rng.uniform(size=(size, size, 2))) for _ in range(n)]
    max_min = [(rng.uniform(3, 4, size=(1, 1)), rng.uniform(0, 1, size=(1, 1))) for _ in range(n)]
    return PairedComplexDatasetWithScale(data, max_min)


def test_update_ema_matches_legacy():
    legacy = load_legacy(GAN, ["update_ema"])
    torch.manual_seed(0)
    G = UNetGenerator()
    ema_a, ema_b = copy.deepcopy(G), copy.deepcopy(G)
    with torch.no_grad():
        for p in G.parameters():
            p.add_(torch.randn_like(p))
    update_ema(ema_a, G, 0.999)
    legacy.update_ema(ema_b, G, 0.999)
    for (k, a), b in zip(ema_a.state_dict().items(), ema_b.state_dict().values()):
        assert torch.equal(a, b), k


@pytest.mark.parametrize("use_ema", [False, True])
def test_validate_fn_matches_legacy(use_ema):
    legacy = load_legacy(GAN, ["validate_fn"])
    torch.manual_seed(0)
    G, G_ema = UNetGenerator(), UNetGenerator()
    dl = DataLoader(_gan_dataset(n=3), batch_size=2)
    assert validate_fn(dl, G, "cpu", use_ema, G_ema) == legacy.validate_fn(dl, G, "cpu", use_ema, G_ema)


@pytest.mark.parametrize("cfg", [
    dict(lambda_phase=20.0, phase_weight_mode="geo", use_phase_grad=False, lambda_phase_grad=0.0),
    dict(lambda_phase=0.0, phase_weight_mode="none", use_phase_grad=False, lambda_phase_grad=0.0),
    dict(lambda_phase=20.0, phase_weight_mode="geo", use_phase_grad=True, lambda_phase_grad=20.0),
])
def test_train_fn_matches_legacy(cfg):
    legacy = load_legacy(
        GAN,
        ["train_fn", "hinge_d_loss", "hinge_g_loss", "phase_circular_loss", "phase_gradient_loss"],
        extra_globals={"device": "cpu"},
    )
    ds = _gan_dataset(n=4)
    results = []
    for impl in ("legacy", "new"):
        set_seed(0)
        G = UNetGenerator()
        D = PatchGANDiscriminator()
        opt_g = torch.optim.Adam(G.parameters(), lr=3e-5, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=5e-6, betas=(0.5, 0.999))
        dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)
        common = dict(lambda_l1=100.0, use_phase=True, lambda_amp=5.0, **cfg)
        if impl == "legacy":
            out = legacy.train_fn(dl, G, D, torch.nn.L1Loss(), opt_g, opt_d, use_hinge=True, **common)
        else:
            out = train_fn(dl, G, D, torch.nn.L1Loss(), opt_g, opt_d, "cpu", **common)
        results.append((out, G.state_dict(), D.state_dict()))
    (out_a, g_a, d_a), (out_b, g_b, d_b) = results
    assert out_a[0] == out_b[0] and out_a[1] == out_b[1] and out_a[7] == out_b[7]
    for a, b in zip(out_a[2:7], out_b[2:7]):
        assert torch.equal(a, b)
    for k in g_a:
        assert torch.equal(g_a[k], g_b[k]), k
    for k in d_a:
        assert torch.equal(d_a[k], d_b[k]), k


def test_train_loop_smoke(tmp_path):
    set_seed(0)
    ds = _gan_dataset(n=4)
    train_dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)
    val_dl = DataLoader(_gan_dataset(n=2, seed=1), batch_size=2)
    G, D = UNetGenerator(), PatchGANDiscriminator()
    _, _, best, best_epoch = train_loop(
        train_dl, val_dl, G, D, 2, str(tmp_path), "cpu", lr=3e-5, dlr=5e-6,
        lambda_phase=20.0, lambda_amp=5.0, checkpoint_every=1, scheduler_patience=15,
    )
    df = pd.read_csv(tmp_path / "metrics_epoch.csv")
    assert list(df.columns) == ['epoch', 'loss_g', 'loss_d', 'train_adv', 'train_l1', 'train_phase',
                                'train_amp', 'train_phase_grad', 'val_amp_mae', 'val_phase_mae', 'val_l1']
    assert len(df) == 2 and 1 <= best_epoch <= 2
    w = tmp_path / "weight"
    for name in ["G_best.pth", "D_best.pth", "G1.pth", "G2.pth", "G_ema1.pth", "G_ema_final.pth"]:
        assert (w / name).exists(), name
    assert not (w / "G_ema_best.pth").exists()
    assert (tmp_path / "training_curves.png").exists()
    assert (tmp_path / "generated" / "epoch_1.png").exists()


def test_train_diffusion_matches_legacy(tmp_path, monkeypatch):
    legacy = load_legacy(DIFF, ["train", "UNET", "UnetLayer", "ResBlock", "Attention",
                                "SinusoidalEmbeddings", "DDPM_Scheduler", "set_seed"])
    g = legacy.train.__globals__
    real_loader, real_sched = g["DataLoader"], g["DDPM_Scheduler"]
    g["DataLoader"] = lambda *a, **k: real_loader(*a, **{**k, "num_workers": 0})
    g["DDPM_Scheduler"] = lambda num_time_steps=1000: real_sched(num_time_steps=num_time_steps, device="cpu")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)

    rng = np.random.default_rng(0)
    ds = PairedComplexDataset([(rng.uniform(size=(32, 32, 2)), rng.uniform(size=(32, 32, 2)))
                               for _ in range(3)])
    kw = dict(batch_size=2, num_time_steps=1000, num_epochs=1, seed=0, lr=2e-5,
              use_phase_loss=True, lambda_phase_diff=20.0, experiment_id=2, roi_size=32)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy.train(ds, savePath=str(old_dir), trainSize=32, interpolate=False, **kw)
    train_diffusion(ds, savePath=str(new_dir), device="cpu", num_workers=0, **kw)

    old = torch.load(old_dir / "checkpoint.pth", weights_only=False)
    new = torch.load(new_dir / "checkpoint.pth", weights_only=False)
    assert set(new) == {"weights", "optimizer", "ema"}
    for part in ("weights", "ema"):
        assert old[part].keys() == new[part].keys()
        for k in old[part]:
            assert torch.equal(old[part][k], new[part][k]), (part, k)

    with open(new_dir / "training_metrics.csv") as f:
        rows_new = list(csv.reader(f))
    with open(old_dir / "training_metrics.csv") as f:
        rows_old = list(csv.reader(f))
    assert rows_new[0] == rows_old[0] == DIFFUSION_CSV_HEADER
    deterministic = [0, 1, 2, 3, 7, 8, 9, 10]  # every column except timing / memory
    assert [rows_new[1][i] for i in deterministic] == [rows_old[1][i] for i in deterministic]
