import numpy as np
import pandas as pd
import pytest

from legacy_loader import load_legacy
from octmirror import metrics, stats

EVAL = "evaluate_ablation_unified.py"
METRIC_FNS = ["simcos", "calculate_ssim", "calculate_mse", "calculate_psnr", "histogram_difference",
              "weighted_phase_coherence", "masked_complex_coherence", "phase_gradient_ssim",
              "despeckle_nlm", "bootstrap_ci", "holm_bonferroni", "BOOTSTRAP_ITERS", "BOOTSTRAP_SEED"]


@pytest.fixture(scope="module")
def legacy():
    return load_legacy(EVAL, METRIC_FNS)


def _images(seed=0, shape=(64, 48)):
    rng = np.random.default_rng(seed)
    return rng.uniform(size=shape), rng.uniform(size=shape)


def _fields(seed=0, shape=(64, 48)):
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal(shape) + 1j * rng.standard_normal(shape) for _ in range(2))


# (e) known results on synthetic arrays
def test_identity_values():
    img, _ = _images()
    cx, _ = _fields()
    assert metrics.calculate_ssim(img, img) == pytest.approx(1.0)
    assert metrics.calculate_mse(img, img) == 0.0
    assert metrics.calculate_psnr(img, img) == float("inf")
    assert metrics.histogram_difference(img, img) == pytest.approx(1.0)
    assert metrics.weighted_phase_coherence(cx, cx) == pytest.approx(1.0)
    assert metrics.masked_complex_coherence(cx, cx) == pytest.approx(1.0)
    assert metrics.phase_gradient_ssim(cx, cx)[0] == pytest.approx(1.0)


def test_ccc_is_invariant_to_global_phase_and_wpc_is_not():
    cx, _ = _fields()
    rotated = cx * np.exp(1j * 0.7)
    assert metrics.masked_complex_coherence(rotated, cx) == pytest.approx(1.0)
    assert metrics.weighted_phase_coherence(rotated, cx) == pytest.approx(np.cos(0.7))


def test_psnr_known_value():
    a = np.zeros((10, 10))
    b = np.full((10, 10), 0.1)
    assert metrics.calculate_psnr(a, b) == pytest.approx(20.0)


def test_despeckle_default_h_is_0_1():
    import inspect
    assert inspect.signature(metrics.despeckle_nlm).parameters["h"].default == 0.1
    assert metrics.DESPECKLE_FILTER_NAME == "nlm_h010"


# golden tests
def test_image_metrics_match_legacy(legacy):
    a, b = _images(1)
    assert metrics.simcos(a, b) == legacy.simcos(a, b)
    assert metrics.calculate_ssim(a, b) == legacy.calculate_ssim(a, b)
    assert metrics.calculate_mse(a, b) == legacy.calculate_mse(a, b)
    assert metrics.calculate_psnr(a, b) == legacy.calculate_psnr(a, b)
    for method in ["cosine-similarity", "chi-squared", "kullback-leibler"]:
        assert metrics.histogram_difference(a, b, method) == legacy.histogram_difference(a, b, method)
    np.testing.assert_array_equal(metrics.despeckle_nlm(a), legacy.despeckle_nlm(a))


def test_phase_metrics_match_legacy(legacy):
    p, g = _fields(2)
    assert metrics.weighted_phase_coherence(p, g) == legacy.weighted_phase_coherence(p, g)
    assert metrics.masked_complex_coherence(p, g) == legacy.masked_complex_coherence(p, g)
    assert metrics.phase_gradient_ssim(p, g) == legacy.phase_gradient_ssim(p, g)


def test_compute_bscan_metrics_matches_legacy_evaluate_single_bscan(legacy):
    """Run legacy evaluate_single_bscan with an identity 'generator' and compare."""
    import torch

    ns = load_legacy(EVAL, ["evaluate_single_bscan", "logScale", "inverseLogScale"] + METRIC_FNS,
                     extra_globals={"device": "cpu"})
    from octmirror.preprocessing import inverse_log_scale, log_scale

    target, inp = _fields(3, shape=(64, 64))

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    old = ns.evaluate_single_bscan(Identity(), inp, target)
    log_in, _, _, _, _ = log_scale(inp[np.newaxis])
    log_t, smax, smin, _, _ = log_scale(target[np.newaxis])
    fake = torch.tensor(log_in[0].transpose(2, 0, 1), dtype=torch.float32).numpy().transpose(1, 2, 0)[np.newaxis]
    pred = inverse_log_scale(fake, smax, smin)
    gt = inverse_log_scale(log_t, smax, smin)
    new = metrics.compute_bscan_metrics(pred[0, ..., 0] + 1j * pred[0, ..., 1], gt[0, ..., 0] + 1j * gt[0, ..., 1])
    assert new == old


def test_bootstrap_and_holm_match_legacy(legacy):
    rng = np.random.default_rng(3)
    data = np.concatenate([rng.normal(size=50), [np.nan, np.inf]])
    assert stats.bootstrap_ci(data) == legacy.bootstrap_ci(data)
    assert stats.BOOTSTRAP_ITERS == 2000 and stats.BOOTSTRAP_SEED == 42
    p = rng.uniform(size=42) * 0.1
    np.testing.assert_array_equal(stats.holm_bonferroni(p), legacy.holm_bonferroni(p))


def test_holm_bonferroni_known_values():
    np.testing.assert_allclose(stats.holm_bonferroni([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def _per_bscan_df(configs, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in configs:
        for f in ["a.npy", "b.npy"]:
            for i in range(15):
                rows.append(dict(config=c, checkpoint_type="G_best", tomogram_file=f, bscan_index=i,
                                 **{m: rng.normal() for m in metrics.METRIC_COLUMNS}))
    return pd.DataFrame(rows)


def test_wilcoxon_vs_reference_matches_legacy(tmp_path):
    configs = ["M2_ComplexCGAN_NoPhase", "M3_PCCGAN_Reference", "M4_NoAmplitudeWeighting",
               "M5_PhaseWeight_Low", "M6_PhaseWeight_MidHigh", "M7_PhaseWeight_High",
               "M8_PhaseGradientConstraint"]
    df = _per_bscan_df(configs)
    ns = load_legacy(EVAL, ["build_wilcoxon_tests", "holm_bonferroni", "log_error", "CONFIG_NAMES",
                            "REFERENCE_CONFIG", "METRIC_COLUMNS"],
                     extra_globals={"EVALUATION_DIR": str(tmp_path),
                                    "ERROR_LOG_PATH": str(tmp_path / "err.log")})
    ns.build_wilcoxon_tests(df, {"M3_PCCGAN_Reference": "G_best"})
    old = pd.read_csv(tmp_path / "wilcoxon_tests.csv")

    new = stats.wilcoxon_vs_reference(df, configs)
    assert len(new) == 42
    new_path = tmp_path / "new.csv"
    new.to_csv(new_path, index=False)
    pd.testing.assert_frame_equal(pd.read_csv(new_path), old)
