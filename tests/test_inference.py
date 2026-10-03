import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from timm.utils import ModelEmaV3

from legacy_loader import load_legacy
from octmirror.inference import load_diffusion, load_pccgan, run_pccgan, sample_ddpm, scale_limits
from octmirror.models.diffusion import UNET, DDPMScheduler
from octmirror.models.pccgan import UNetGenerator
from octmirror.preprocessing import inverse_log_scale, log_scale

DIFF = "diffComplexField_ROI256_D1_D4.py"
ROOT = Path(__file__).resolve().parents[1]


def _save_checkpoint(path, data_parallel):
    torch.manual_seed(0)
    model = UNET()
    if data_parallel:
        model = nn.DataParallel(model)
    ema = ModelEmaV3(model, decay=0.9999)
    with torch.no_grad():  # make raw weights differ from the EMA weights
        for p in model.parameters():
            p.add_(1.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.save({"weights": model.state_dict(), "optimizer": opt.state_dict(),
                "ema": ema.state_dict()}, path)
    raw = model.module if data_parallel else model
    ema_model = ema.module.module if data_parallel else ema.module
    return raw.state_dict(), ema_model.state_dict()


# (f)
@pytest.mark.parametrize("data_parallel", [True, False])
def test_load_diffusion_strips_module_prefix(tmp_path, data_parallel):
    path = tmp_path / "checkpoint.pth"
    raw, ema = _save_checkpoint(path, data_parallel)
    if data_parallel:
        saved = torch.load(path, weights_only=True)
        assert all(k.startswith("module.") for k in saved["weights"])
        assert all(k.startswith("module.module.") for k in saved["ema"])
    for use_ema, expected in [(True, ema), (False, raw)]:
        model = load_diffusion(str(path), "cpu", use_ema=use_ema)
        assert not model.training
        for k, v in model.state_dict().items():
            assert torch.equal(v, expected[k]), k


def test_legacy_inference_cannot_load_data_parallel_ema(tmp_path, monkeypatch):
    """Documents KNOWN_ISSUES: the original sampler fails on a DataParallel EMA state."""
    legacy = load_legacy(DIFF, ["inference", "UNET", "UnetLayer", "ResBlock", "Attention",
                                "SinusoidalEmbeddings", "DDPM_Scheduler", "display_reverse"])
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *a, **k: self)
    _save_checkpoint(tmp_path / "checkpoint.pth", data_parallel=True)
    with pytest.raises(RuntimeError):
        legacy.inference(savePath=str(tmp_path), num_time_steps=5,
                         input_img=torch.zeros(1, 2, 32, 32), train_size=32)


def test_sample_ddpm_matches_legacy_inference(tmp_path, monkeypatch):
    legacy = load_legacy(DIFF, ["inference", "UNET", "UnetLayer", "ResBlock", "Attention",
                                "SinusoidalEmbeddings", "DDPM_Scheduler"])
    g = legacy.inference.__globals__
    captured = {}
    g["display_reverse"] = lambda images, savePath: captured.setdefault("images", images)
    real_sched = g["DDPM_Scheduler"]
    g["DDPM_Scheduler"] = lambda num_time_steps=1000: real_sched(num_time_steps=num_time_steps, device="cpu")
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)

    _save_checkpoint(tmp_path / "checkpoint.pth", data_parallel=False)
    torch.manual_seed(5)
    condition = torch.rand(1, 2, 32, 32)
    steps = 12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        torch.manual_seed(7)
        legacy.inference(savePath=str(tmp_path), num_time_steps=steps, input_img=condition, train_size=32)
    expected = captured["images"][-1]

    model = load_diffusion(str(tmp_path / "checkpoint.pth"), "cpu", use_ema=True)
    # The legacy sampler builds UNET() (consuming RNG) between seeding and drawing the
    # initial noise; replay that consumption, then sample without reseeding.
    torch.manual_seed(7)
    UNET()
    out = sample_ddpm(model, condition, "cpu", num_time_steps=steps, seed=None,
                      scheduler=DDPMScheduler(steps, device="cpu"))
    assert out.shape == (1, 2, 32, 32)
    assert torch.equal(out, expected)
    # reproducible with the same explicit seed
    a = sample_ddpm(model, condition, "cpu", num_time_steps=steps, seed=7)
    assert torch.equal(sample_ddpm(model, condition, "cpu", num_time_steps=steps, seed=7), a)


def test_load_and_run_pccgan_matches_evaluation_path(tmp_path):
    torch.manual_seed(0)
    G = UNetGenerator()
    torch.save(G.state_dict(), tmp_path / "G_best.pth")
    G2 = load_pccgan(str(tmp_path / "G_best.pth"), "cpu")
    assert not G2.training

    rng = np.random.default_rng(0)
    inp = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    tgt = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    smax, smin = scale_limits(tgt)
    pred = run_pccgan(G2, inp, smax, smin, "cpu")

    log_in, _, _, _, _ = log_scale(inp[np.newaxis])
    with torch.no_grad():
        fake = G.eval()(torch.tensor(log_in[0].transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0))
    rec = inverse_log_scale(fake[0].numpy().transpose(1, 2, 0)[np.newaxis], smax, smin)
    np.testing.assert_array_equal(pred, rec[0, ..., 0] + 1j * rec[0, ..., 1])


def _run_infer(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "infer.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_infer_help():
    res = _run_infer("--help")
    assert res.returncode == 0 and "--scale-from" in res.stdout


@pytest.mark.parametrize("extra, message", [
    (["--no-simulate-mirror", "--scale-from", "target"], "--scale-from target requires"),
    (["--no-simulate-mirror", "--compute-metrics"], "--compute-metrics requires"),
    (["--use-ema"], "only apply to --model diffusion"),
])
def test_infer_rejects_invalid_combinations(extra, message):
    res = _run_infer("--model", "pccgan", "--weights", "w.pth", "--input", "x.npy",
                     "--out-dir", "o", *extra)
    assert res.returncode == 2 and message in res.stderr


def test_infer_pccgan_end_to_end(tmp_path):
    torch.manual_seed(0)
    torch.save(UNetGenerator().state_dict(), tmp_path / "G.pth")
    rng = np.random.default_rng(0)
    tom = rng.standard_normal((128, 128, 3)) + 1j * rng.standard_normal((128, 128, 3))
    np.save(tmp_path / "vol.npy", np.stack([tom.real, tom.imag], axis=-1))
    out = tmp_path / "out"
    res = _run_infer("--model", "pccgan", "--weights", str(tmp_path / "G.pth"),
                     "--input", str(tmp_path / "vol.npy"), "--out-dir", str(out),
                     "--bscan-indices", "0", "2", "--device", "cpu", "--compute-metrics")
    assert res.returncode == 0, res.stderr
    rec = np.load(out / "vol_pccgan_reconstruction.npy")
    assert rec.shape == (128, 128, 2) and np.iscomplexobj(rec)
    df = pd.read_csv(out / "metrics.csv")
    assert list(df["bscan_index"]) == [0, 2]
    assert set(df["despeckle_filter"]) == {"nlm_h010"}
    assert (out / "png" / "vol_y0002_phase.png").exists()
    assert (out / "png" / "vol_y0000_amplitude_db.png").exists()
