"""Reconstruct complex B-scans of a tomogram with a trained PC-CGAN or diffusion model.

Example:
    python scripts/infer.py --model pccgan --weights weights/pccgan_M3_G_best.pth \
        --input data/sample_volumes/optic_nerve.npy --out-dir outputs/pccgan --compute-metrics
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from octmirror.inference import (
    DIFFUSION_MAX_LOG_AMP,
    load_diffusion,
    load_pccgan,
    run_diffusion,
    run_pccgan,
    scale_limits,
)
from octmirror.metrics import DESPECKLE_FILTER_NAME, compute_bscan_metrics
from octmirror.preprocessing import (
    extract_center_roi,
    inverse_log_scale,
    load_tomogram,
    log_scale,
    mirror_artifact,
)
from octmirror.visualization import save_amplitude_phase_png

DIFFUSION_ROI_SIZE = 256
DIFFUSION_TIME_STEPS = 1000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mirror-artifact suppression on a complex OCT tomogram (Z, X, Y).")
    p.add_argument("--model", required=True, choices=["pccgan", "diffusion"])
    p.add_argument("--weights", required=True,
                   help="PC-CGAN generator state_dict, or diffusion checkpoint.pth.")
    p.add_argument("--input", required=True,
                   help=".npy tomogram (Z, X, Y) complex, or (Z, X, Y, 2) real/imag.")
    p.add_argument("--bscan-indices", type=int, nargs="+", default=None,
                   help="B-scan indices along Y (default: all).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available, else cpu).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for diffusion sampling (reset before every B-scan).")
    p.add_argument("--simulate-mirror", action=argparse.BooleanOptionalAction, default=True,
                   help="Simulate the mirror artifact from an artifact-free input (default). "
                        "Use --no-simulate-mirror if the input already contains the artifact.")
    p.add_argument("--scale-from", choices=["target", "input"], default=None,
                   help="Log-amplitude limits used to de-normalize the output. 'target' (clean "
                        "volume, paper protocol) requires --simulate-mirror; 'input' is the only "
                        "option for data that already contain the artifact. Default: 'target' "
                        "with --simulate-mirror, 'input' otherwise.")
    p.add_argument("--compute-metrics", action="store_true",
                   help="Write metrics.csv against the clean volume (requires --simulate-mirror).")
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=None,
                   help="Diffusion only: sample with the EMA weights (default) or the raw weights.")
    return p


def resolve_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.scale_from is None:
        args.scale_from = "target" if args.simulate_mirror else "input"
    if args.scale_from == "target" and not args.simulate_mirror:
        parser.error("--scale-from target requires --simulate-mirror (no clean target is available).")
    if args.compute_metrics and not args.simulate_mirror:
        parser.error("--compute-metrics requires --simulate-mirror and an artifact-free input volume.")
    if args.use_ema is not None and args.model != "diffusion":
        parser.error("--use-ema / --no-use-ema only apply to --model diffusion.")
    if args.use_ema is None:
        args.use_ema = True
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main() -> None:
    parser = build_parser()
    args = resolve_args(parser, parser.parse_args())
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    tom = load_tomogram(args.input)  # (Z, X, Y)
    n_y = tom.shape[2]
    indices = list(range(n_y)) if args.bscan_indices is None else args.bscan_indices
    bad = [i for i in indices if not 0 <= i < n_y]
    if bad:
        parser.error(f"B-scan indices out of range [0, {n_y - 1}]: {bad}")

    if args.model == "pccgan":
        model = load_pccgan(args.weights, device)
        # PC-CGAN evaluation protocol: artifact simulated on the full volume along Z.
        tomcc = mirror_artifact(tom) if args.simulate_mirror else tom
    else:
        model = load_diffusion(args.weights, device, use_ema=args.use_ema)
        tomcc = None

    stem = os.path.splitext(os.path.basename(args.input))[0]
    recon = []
    rows = []
    for k, y in enumerate(indices):
        print(f"B-scan {y} ({k + 1}/{len(indices)})")
        if args.model == "pccgan":
            input_cx = tomcc[:, :, y]
            target_cx = tom[:, :, y] if args.simulate_mirror else None
            max_log_amp = None
        else:
            if args.simulate_mirror:
                # Diffusion training protocol: artifact simulated on the central ROI.
                target_cx = extract_center_roi(tom[:, :, y], roi_size=DIFFUSION_ROI_SIZE)
                input_cx = mirror_artifact(target_cx)
            else:
                target_cx = None
                input_cx = extract_center_roi(tom[:, :, y], roi_size=DIFFUSION_ROI_SIZE)
            max_log_amp = DIFFUSION_MAX_LOG_AMP

        smax, smin = scale_limits(target_cx if args.scale_from == "target" else input_cx)
        if args.model == "pccgan":
            pred_cx = run_pccgan(model, input_cx, smax, smin, device)
        else:
            pred_cx = run_diffusion(model, input_cx, smax, smin, device,
                                    num_time_steps=DIFFUSION_TIME_STEPS, seed=args.seed)
        recon.append(pred_cx)

        save_amplitude_phase_png(
            pred_cx,
            os.path.join(png_dir, f"{stem}_y{y:04d}_amplitude_db.png"),
            os.path.join(png_dir, f"{stem}_y{y:04d}_phase.png"),
            title=f"{args.model} B-scan {y}",
        )

        if args.compute_metrics:
            # Reference: the clean B-scan passed through the normalization with its own limits.
            log_t, smax_t, smin_t, _, _ = log_scale(target_cx[np.newaxis, ...])
            gt_rec = inverse_log_scale(log_t, smax_t, smin_t, max_log_amp=max_log_amp)
            gt_cx = gt_rec[0, :, :, 0] + 1j * gt_rec[0, :, :, 1]
            rows.append(dict(tomogram_file=os.path.basename(args.input), bscan_index=int(y),
                             model=args.model, despeckle_filter=DESPECKLE_FILTER_NAME,
                             **compute_bscan_metrics(pred_cx, gt_cx)))

    recon_path = os.path.join(args.out_dir, f"{stem}_{args.model}_reconstruction.npy")
    np.save(recon_path, np.stack(recon, axis=-1))  # (Z, X, n_bscans) complex
    print(f"Saved {recon_path}")

    info = dict(model=args.model, weights=args.weights, input=args.input, bscan_indices=indices,
                simulate_mirror=args.simulate_mirror, scale_from=args.scale_from,
                use_ema=args.use_ema if args.model == "diffusion" else None,
                seed=args.seed if args.model == "diffusion" else None,
                roi_size=DIFFUSION_ROI_SIZE if args.model == "diffusion" else None,
                num_time_steps=DIFFUSION_TIME_STEPS if args.model == "diffusion" else None,
                device=str(device))
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    if args.compute_metrics:
        metrics_path = os.path.join(args.out_dir, "metrics.csv")
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
