import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from legacy_loader import load_legacy
from octmirror.models.pccgan import UNetGenerator

ROOT = Path(__file__).resolve().parents[1]
EVAL = "evaluate_ablation_unified.py"

PCCGAN_TABLE = {
    "m2": ("M2_ComplexCGAN_NoPhase", 0.0, "none", False, 0.0),
    "m3": ("M3_PCCGAN_Reference", 20.0, "geo", False, 0.0),
    "m4": ("M4_NoAmplitudeWeighting", 20.0, "none", False, 0.0),
    "m5": ("M5_PhaseWeight_Low", 2.0, "geo", False, 0.0),
    "m6": ("M6_PhaseWeight_MidHigh", 60.0, "geo", False, 0.0),
    "m7": ("M7_PhaseWeight_High", 150.0, "geo", False, 0.0),
    "m8": ("M8_PhaseGradientConstraint", 20.0, "geo", True, 20.0),
}
DIFFUSION_TABLE = {
    "d1": (1, "D1_ROI256_NoPhase", False, 0.0, 8, 2e-5),
    "d2": (2, "D2_ROI256_PhaseLoss", True, 20.0, 8, 2e-5),
    "d3": (3, "D3_ROI256_PhaseLoss_LRScaled", True, 20.0, 16, 4e-5),
    "d4": (4, "D4_ROI256_PhaseLoss_LRHigh", True, 20.0, 16, 8e-5),
}
FILES = ["OpticNerveAOld.npy", "OpticNerveBOld.npy", "unpairCadaverhearth.npy", "ChickenBreastA.npy",
         "ChickenBreastB.npy", "NailA.npy", "NailB.npy", "S.Eye2A.npy", "S.Eye2B.npy"]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


@pytest.mark.parametrize("key", sorted(PCCGAN_TABLE))
def test_pccgan_configs_match_paper_table(key):
    cfg = yaml.safe_load((ROOT / "configs" / "pccgan" / f"{key}.yaml").read_text())
    name, lp, mode, grad, lpg = PCCGAN_TABLE[key]
    tr = cfg["training"]
    assert cfg["name"] == name and cfg["seed"] == 0 and cfg["split_seed"] == 0 and cfg["val_fraction"] == 0.1
    assert cfg["data"] == {"folders": ["phase"], "files": FILES}
    assert (tr["lambda_phase"], tr["phase_weight_mode"], tr["use_phase_grad"], tr["lambda_phase_grad"]) == (lp, mode, grad, lpg)
    assert tr["lambda_l1"] == 100.0 and tr["lambda_amp"] == 5.0
    assert tr["lr_g"] == 3e-5 and tr["lr_d"] == 5e-6 and tr["betas"] == [0.5, 0.999]
    assert tr["epochs"] == 300 and tr["batch_size"] == 32 and tr["adversarial_loss"] == "hinge"
    assert tr["use_ema"] is True and tr["ema_decay"] == 0.999 and tr["grad_clip"] == 1.0
    assert tr["scheduler"] == {"patience": 15, "factor": 0.5, "min_lr": 1e-7}
    assert tr["checkpoint_every"] == 5 and tr["image_log_every"] == 25


@pytest.mark.parametrize("key", sorted(DIFFUSION_TABLE))
def test_diffusion_configs_match_paper_table(key):
    cfg = yaml.safe_load((ROOT / "configs" / "diffusion" / f"{key}.yaml").read_text())
    eid, name, upl, lam, bs, lr = DIFFUSION_TABLE[key]
    tr, data = cfg["training"], cfg["data"]
    assert (cfg["experiment_id"], cfg["name"], cfg["seed"]) == (eid, name, 0)
    assert (tr["use_phase_loss"], tr["lambda_phase_diff"], tr["batch_size"], tr["lr"]) == (upl, lam, bs, lr)
    assert tr["epochs"] == 80 and tr["num_time_steps"] == 1000 and tr["ema_decay"] == 0.9999
    assert tr["beta_start"] == 1e-4 and tr["beta_end"] == 0.02
    assert data == {"folders": ["noPhase", "phase", "synthetic"], "roi_size": 256,
                    "roi_mode": "center", "n_bscans_per_volume": 40}
    _load_script("train_diffusion").load_config(str(ROOT / "configs" / "diffusion" / f"{key}.yaml"))


def test_evaluation_config_names_match_yaml():
    ev = _load_script("evaluate_pccgan")
    assert ev.CONFIG_NAMES == [PCCGAN_TABLE[k][0] for k in sorted(PCCGAN_TABLE)]


@pytest.mark.parametrize("script", ["train_pccgan.py", "train_diffusion.py", "evaluate_pccgan.py"])
def test_scripts_help(script):
    assert _run(script, "--help").returncode == 0


def _write_volumes(folder, names, shape, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in names:
        tom = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        np.save(folder / name, tom)


def test_train_pccgan_script_smoke(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "pccgan" / "m3.yaml").read_text())
    cfg["training"].update(epochs=1, batch_size=2)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    _write_volumes(tmp_path / "data" / "phase", ["NailA.npy", "other.npy"], (128, 128, 10))
    res = _run("train_pccgan.py", "--config", str(tmp_path / "cfg.yaml"), "--data-root",
               str(tmp_path / "data"), "--out-dir", str(tmp_path / "runs"), "--device", "cpu")
    assert res.returncode == 0, res.stderr
    run = tmp_path / "runs" / "M3_PCCGAN_Reference"
    assert (run / "weight" / "G_best.pth").exists() and (run / "metrics_epoch.csv").exists()
    summary = json.loads((tmp_path / "runs" / "ablation_summary.json").read_text())
    assert summary[0]["name"] == "M3_PCCGAN_Reference" and summary[0]["best_epoch"] == 1


def test_train_diffusion_script_smoke_and_overwrite_guard(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "diffusion" / "d2.yaml").read_text())
    cfg["training"].update(epochs=1, batch_size=2)
    cfg["data"].update(roi_size=32, n_bscans_per_volume=2)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    for folder in ["noPhase", "phase", "synthetic"]:
        _write_volumes(tmp_path / "data" / folder, ["a.npy"], (40, 40, 3))
    args = ["--config", str(tmp_path / "cfg.yaml"), "--data-root", str(tmp_path / "data"),
            "--out-dir", str(tmp_path / "runs"), "--device", "cpu", "--num-workers", "0"]
    res = _run("train_diffusion.py", *args)
    assert res.returncode == 0, res.stderr
    run = tmp_path / "runs" / "D2_ROI256_PhaseLoss"
    assert json.loads((run / "run_metadata.json").read_text())["status"] == "ok"
    assert json.loads((run / "config.json").read_text())["batch_size"] == 2
    ckpt = torch.load(run / "checkpoint.pth", weights_only=False)
    assert set(ckpt) == {"weights", "optimizer", "ema"}
    assert len(pd.read_csv(run / "training_metrics.csv")) == 1

    res = _run("train_diffusion.py", *args)
    assert res.returncode != 0 and "Refusing to overwrite" in res.stderr
    assert _run("train_diffusion.py", *args, "--allow-overwrite").returncode == 0


LEGACY_EVAL_NAMES = [
    "run_full_evaluation", "build_validation_samples", "load_tomogram_complex", "select_bscan_indices",
    "mirrorArtifact", "logScale", "inverseLogScale", "GeneratorTf", "evaluate_single_bscan",
    "simcos", "calculate_ssim", "calculate_mse", "calculate_psnr", "histogram_difference",
    "weighted_phase_coherence", "masked_complex_coherence", "phase_gradient_ssim", "despeckle_nlm",
    "bootstrap_ci", "holm_bonferroni", "build_summary_metrics", "build_checkpoint_comparison",
    "build_wilcoxon_tests", "build_training_curve_plots", "build_metric_boxplots", "log_error",
    "CONFIG_NAMES", "REFERENCE_CONFIG", "CHECKPOINT_TYPES", "N_BSCANS_PER_VOLUME",
    "DESPECKLE_FILTER_NAME", "METRIC_COLUMNS", "TRAINING_CURVE_METRICS", "BOOTSTRAP_ITERS",
    "BOOTSTRAP_SEED",
]


def test_evaluate_pccgan_reproduces_legacy_outputs(tmp_path):
    ev = _load_script("evaluate_pccgan")
    ablation = tmp_path / "ablation"
    for k, name in enumerate(ev.CONFIG_NAMES):
        (ablation / name / "weight").mkdir(parents=True)
        torch.manual_seed(k)
        torch.save(UNetGenerator().state_dict(), ablation / name / "weight" / "G_best.pth")
        pd.DataFrame({"epoch": [1, 2], "loss_g": [1.0, 0.5 + k], "val_amp_mae": [0.2, 0.1],
                      "val_phase_mae": [1, 1], "train_phase": [1, 1], "train_phase_grad": [0, 0],
                      "train_amp": [1, 1]}).to_csv(ablation / name / "metrics_epoch.csv", index=False)
    _write_volumes(tmp_path / "val", ["v1.npy", "v2.npy"], (128, 128, 4))

    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    new_dir.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # scipy, on synthetic data
        ev.run_full_evaluation(str(ablation), str(tmp_path / "val"), str(new_dir), "cpu")

    old_dir.mkdir()
    legacy = load_legacy(EVAL, LEGACY_EVAL_NAMES, extra_globals=dict(
        ABLATION_ROOT=str(ablation), VALIDATION_DATA_DIR=str(tmp_path / "val"),
        EVALUATION_DIR=str(old_dir), TRAINING_CURVES_PLOTS_DIR=str(old_dir / "curves"),
        BOXPLOTS_DIR=str(old_dir / "box"), ERROR_LOG_PATH=str(old_dir / "errors.log"), device="cpu"))
    (old_dir / "curves").mkdir()
    (old_dir / "box").mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy.run_full_evaluation()

    new_rows = pd.read_csv(new_dir / "per_bscan_metrics.csv")
    old_rows = pd.read_csv(old_dir / "per_bscan_metrics.csv")
    assert len(new_rows) == 7 * 2 * 4
    assert set(new_rows["despeckle_filter"]) == {"nlm_h010"}
    assert set(old_rows["despeckle_filter"]) == {"nlm_h020"}  # mislabelled in the original
    pd.testing.assert_frame_equal(new_rows.drop(columns="despeckle_filter"),
                                  old_rows.drop(columns="despeckle_filter"))
    for name in ["summary_metrics.csv", "wilcoxon_tests.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(new_dir / name), pd.read_csv(old_dir / name))
    assert len(pd.read_csv(new_dir / "wilcoxon_tests.csv")) == 42
    for metric in ev.METRIC_COLUMNS:
        assert (new_dir / "plots" / "metric_boxplots" / f"boxplot_{metric}.png").exists()
    assert (new_dir / "plots" / "training_curves_unified" / "unified_loss_g.png").exists()
