"""Evaluate the trained PC-CGAN ablations (M2-M8) on the independent validation set.

Nothing is trained. For every configuration folder in ``--ablation-root`` the
``weight/G_best.pth`` generator is evaluated on up to 40 evenly spaced B-scans per
validation volume; per-B-scan metrics, summary statistics with bootstrap CIs, paired
Wilcoxon tests against M3 (Holm-Bonferroni over all 42 tests), unified training curves
and metric box plots are written to ``--out-dir``.

Example:
    python scripts/evaluate_pccgan.py --ablation-root /path/to/AblationPCCGAN \
        --validation-dir /path/to/validationCxPaper --out-dir outputs/evaluation
"""
from __future__ import annotations

import argparse
import os
import traceback
from glob import glob
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch

from octmirror.inference import load_pccgan, run_pccgan
from octmirror.metrics import DESPECKLE_FILTER_NAME, METRIC_COLUMNS, compute_bscan_metrics
from octmirror.preprocessing import (
    inverse_log_scale,
    load_tomogram,
    log_scale,
    mirror_artifact,
    select_bscan_indices,
)
from octmirror.stats import REFERENCE_CONFIG, bootstrap_ci, wilcoxon_vs_reference
from octmirror.visualization import build_metric_boxplots, build_training_curve_plots

CONFIG_NAMES = [
    "M2_ComplexCGAN_NoPhase",
    "M3_PCCGAN_Reference",
    "M4_NoAmplitudeWeighting",
    "M5_PhaseWeight_Low",
    "M6_PhaseWeight_MidHigh",
    "M7_PhaseWeight_High",
    "M8_PhaseGradientConstraint",
]
CHECKPOINT_TYPE = 'G_best'
N_BSCANS_PER_VOLUME = 40


def make_error_logger(path: str) -> Callable[[str], None]:
    def log_error(message: str) -> None:
        with open(path, 'a') as f:
            f.write(message + "\n")
        print(message)
    return log_error


def build_validation_samples(validation_dir: str, n_bscans: int = N_BSCANS_PER_VOLUME) -> List[Dict]:
    """List of dicts ``{tomogram_file, bscan_index, input_cx, target_cx}``.

    The mirror artifact is simulated on each full clean volume (FFT along Z); up to
    ``n_bscans`` evenly spaced B-scans per volume are kept. Sorted by
    ``(tomogram_file, bscan_index)`` for reproducible pairing.
    """
    samples = []
    files = sorted(glob(os.path.join(validation_dir, '*.npy')))
    for filepath in files:
        fname = os.path.basename(filepath)
        tom = load_tomogram(filepath)  # (Z, X, Y) clean, complex
        tomcc = mirror_artifact(tom)   # (Z, X, Y) with induced mirror artifact
        num_bscans = tom.shape[2]
        y_indices = select_bscan_indices(num_bscans, n_bscans)
        for y_idx in y_indices:
            samples.append(dict(
                tomogram_file=fname,
                bscan_index=int(y_idx),
                input_cx=tomcc[:, :, y_idx],
                target_cx=tom[:, :, y_idx],
            ))
    samples.sort(key=lambda s: (s['tomogram_file'], s['bscan_index']))
    return samples


def evaluate_single_bscan(G, input_cx: np.ndarray, target_cx: np.ndarray, device) -> Dict[str, float]:
    """Run one (input, target) pair through the generator and compute the metrics.

    Prediction and ground truth are both de-normalized with the target's limits.
    """
    logslices_tgt, smax_tgt, smin_tgt, _, _ = log_scale(target_cx[np.newaxis, ...])
    pred_cx = run_pccgan(G, input_cx, smax_tgt, smin_tgt, device)
    gt_rec = inverse_log_scale(logslices_tgt, smax_tgt, smin_tgt, max_log_amp=None)
    gt_cx = gt_rec[0, :, :, 0] + 1j * gt_rec[0, :, :, 1]
    return compute_bscan_metrics(pred_cx, gt_cx)


def build_summary_metrics(per_bscan_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Mean, std and 95% bootstrap CI of every metric per configuration."""
    rows = []
    grouped = per_bscan_df.groupby(['config', 'checkpoint_type'])
    for (config_name, checkpoint_type), group in grouped:
        row = dict(config=config_name, checkpoint_type=checkpoint_type)
        for metric in METRIC_COLUMNS:
            values = group[metric].values
            row[f'{metric}_mean'] = float(np.nanmean(values))
            row[f'{metric}_std'] = float(np.nanstd(values))
            ci_lower, ci_upper = bootstrap_ci(values)
            row[f'{metric}_ci95_lower'] = ci_lower
            row[f'{metric}_ci95_upper'] = ci_upper
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    return summary_df


def build_wilcoxon_tests(per_bscan_df: pd.DataFrame, out_dir: str,
                         log_error: Callable[[str], None] = print) -> pd.DataFrame:
    """Paired Wilcoxon tests of every configuration vs M3 (Holm-Bonferroni, one family)."""
    print(f"Wilcoxon tests use checkpoint_type='{CHECKPOINT_TYPE}' for all configs.")
    wilcoxon_df = wilcoxon_vs_reference(per_bscan_df, CONFIG_NAMES, REFERENCE_CONFIG,
                                        checkpoint_type=CHECKPOINT_TYPE, log_error=log_error)
    wilcoxon_path = os.path.join(out_dir, 'wilcoxon_tests.csv')
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"Saved {wilcoxon_path}")
    return wilcoxon_df


def run_full_evaluation(ablation_root: str, validation_dir: str, out_dir: str, device) -> None:
    training_curves_dir = os.path.join(out_dir, 'plots', 'training_curves_unified')
    boxplots_dir = os.path.join(out_dir, 'plots', 'metric_boxplots')
    os.makedirs(training_curves_dir, exist_ok=True)
    os.makedirs(boxplots_dir, exist_ok=True)
    error_log_path = os.path.join(out_dir, 'evaluation_errors.log')
    open(error_log_path, 'a').close()
    log_error = make_error_logger(error_log_path)

    print("Building validation samples from independent validation set...")
    samples = build_validation_samples(validation_dir)
    print(f"Total B-scan samples per checkpoint evaluation: {len(samples)}")

    per_bscan_rows = []
    for config_name in CONFIG_NAMES:
        try:
            weight_path = os.path.join(ablation_root, config_name, 'weight', f'{CHECKPOINT_TYPE}.pth')
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
            G = load_pccgan(weight_path, device)

            for sample in samples:
                try:
                    with torch.no_grad():
                        metrics = evaluate_single_bscan(G, sample['input_cx'], sample['target_cx'], device)
                    per_bscan_rows.append(dict(
                        config=config_name,
                        checkpoint_type=CHECKPOINT_TYPE,
                        tomogram_file=sample['tomogram_file'],
                        bscan_index=sample['bscan_index'],
                        despeckle_filter=DESPECKLE_FILTER_NAME,
                        **metrics
                    ))
                except Exception as ex_sample:
                    log_error(
                        f"[{config_name}/{CHECKPOINT_TYPE}] Failed on "
                        f"{sample['tomogram_file']} bscan {sample['bscan_index']}: {ex_sample}\n"
                        f"{traceback.format_exc(limit=5)}"
                    )
            print(f"Finished evaluation: {config_name} / {CHECKPOINT_TYPE}")
        except Exception as ex_ckpt:
            log_error(
                f"[{config_name}/{CHECKPOINT_TYPE}] Checkpoint evaluation failed: {ex_ckpt}\n"
                f"{traceback.format_exc(limit=5)}"
            )

    per_bscan_df = pd.DataFrame(per_bscan_rows, columns=[
        'config', 'checkpoint_type', 'tomogram_file', 'bscan_index', 'despeckle_filter',
        'ssim', 'psnr', 'mse', 'hist_cosine_similarity', 'wpc', 'ccc',
        'pgssim', 'pgssim_axial', 'pgssim_lateral'
    ])
    per_bscan_path = os.path.join(out_dir, 'per_bscan_metrics.csv')
    per_bscan_df.to_csv(per_bscan_path, index=False)
    print(f"Saved {per_bscan_path} ({len(per_bscan_df)} rows)")

    build_summary_metrics(per_bscan_df, out_dir)
    build_wilcoxon_tests(per_bscan_df, out_dir, log_error=log_error)
    build_training_curve_plots(ablation_root, CONFIG_NAMES, training_curves_dir, log_error=log_error)
    build_metric_boxplots(per_bscan_df, CONFIG_NAMES, boxplots_dir, METRIC_COLUMNS,
                          checkpoint_type=CHECKPOINT_TYPE, log_error=log_error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the PC-CGAN ablations (M2-M8).")
    parser.add_argument('--ablation-root', required=True,
                        help='Folder with one sub-folder per configuration (weight/G_best.pth, metrics_epoch.csv).')
    parser.add_argument('--validation-dir', required=True, help='Folder with the clean validation .npy volumes.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default=None, help='Torch device (default: cuda:0 if available, else cpu).')
    args = parser.parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    run_full_evaluation(args.ablation_root, args.validation_dir, args.out_dir, device)


if __name__ == '__main__':
    main()
