# Reproducibility notes

This page lists the seeds, data splits and evaluation protocol used by the code, and what can and cannot be reproduced exactly. See `docs/KNOWN_ISSUES.md` for the issues of the original implementation.

## Environment

- Python >= 3.9, PyTorch >= 2.3, Matplotlib >= 3.9. Install with `pip install -e .` and `pip install -r requirements.txt`.
- The paper's models were trained on CUDA GPUs. The diffusion model used mixed precision (AMP) and, with several GPUs, `nn.DataParallel`.
- `set_seed` sets `torch.backends.cudnn.deterministic = True` and `benchmark = False`, but does not call `torch.use_deterministic_algorithms(True)`. With AMP, `DataParallel` and fused attention kernels, GPU runs are not bitwise reproducible, and results may also vary across PyTorch versions and hardware.

## Seeds

| Where | Value | What it controls |
|---|---|---|
| PC-CGAN `seed` (YAML) | 0 | Python, NumPy and PyTorch RNGs, set right before building the generator and discriminator: network initialization, DataLoader shuffling, dropout. Same as the original script, which reseeded before each configuration. |
| PC-CGAN `split_seed` (YAML) | 0 | Permutation of the 90/10 train/validation split. The original script shuffled before seeding, so the original split is not reproducible; see `KNOWN_ISSUES.md` 1.3. `null` restores the unseeded behaviour. |
| Diffusion `seed` (YAML) | 0 | Set at the start of `train_diffusion`: network initialization, DataLoader shuffling, time steps and noise. |
| Diffusion sampling `--seed` (`infer.py`) | 0 by default | `torch.manual_seed` before the initial noise of every B-scan (the same seed is reused for each B-scan). |
| Bootstrap confidence intervals | 2000 resamples, seed 42 | `octmirror.stats.bootstrap_ci`. |

## Data and splits

- Tomograms are `.npy` arrays with axes (Z, X, Y): depth, fast-scan, slow-scan. A 4D array (Z, X, Y, 2) holds real and imaginary parts.
- Mirror artifacts are simulated numerically from artifact-free volumes (`octmirror.preprocessing.mirror_artifact`): FFT along Z, real part of the fringes, FFT back.
- Each B-scan is normalized independently (`log_scale`): log10 amplitude min-max scaled to [0, 1] with the original phase, then real and imaginary parts mapped to [0, 1].

### PC-CGAN

- Training data: the 9 files listed in `configs/pccgan/*.yaml`, from the folder `phase/` under `--data-root`.
- The artifact is simulated on each full volume; all B-scans are used (512 x 512).
- B-scans of all volumes are pooled and split at random **per B-scan** (not per volume) into 90% training and 10% internal validation. Neighbouring B-scans of one volume can fall on both sides. The internal validation set is only used to select `G_best.pth` and to drive the learning-rate scheduler.
- Volumes are read in `os.listdir` order, which can differ between machines and changes the split even with a fixed `split_seed`.

### Diffusion

- Training data: every `.npy` file in `noPhase/`, `phase/` and `synthetic/` under `--data-root`.
- Up to 40 evenly spaced B-scans per volume (`np.linspace`; all B-scans if a volume has 40 or fewer).
- The central 256 x 256 ROI is cropped at native resolution (no interpolation) and the artifact is simulated on the ROI.
- There is no validation split; the final checkpoint (EMA weights) is used.

## PC-CGAN evaluation protocol (`scripts/evaluate_pccgan.py`)

1. Validation volumes: every `.npy` in `--validation-dir`, sorted by file name. This set is independent of the training data.
2. For each volume, the artifact is simulated on the full volume and up to 40 evenly spaced B-scans are kept (all if 40 or fewer).
3. For each configuration M2-M8, `weight/G_best.pth` is loaded (the only checkpoint evaluated).
4. Each corrupted B-scan is normalized with its own limits and passed through the generator. The prediction and the clean B-scan are both de-normalized with the **clean target's** log-amplitude limits. This uses information from the target and is only possible with simulated artifacts; `infer.py --scale-from input` is the alternative for real corrupted data.
5. Intensity metrics: amplitude in dB (`20*log10|.|`), min-max normalized with the target's range, clipped to [0, 1], despeckled with non-local means (`h = 0.1`, `patch_size = 5`, `patch_distance = 6`, `fast_mode = True`); then SSIM, PSNR, MSE and histogram cosine similarity.
6. Phase metrics on the raw complex fields, without despeckling: WPC, CCC (pixels above the 50th amplitude percentile of the target, after removing the global phase piston) and PG-SSIM (mean of axial and lateral SSIM of the wrapped phase gradients inside the same mask).
7. Summary: mean, standard deviation and 95% bootstrap CI (2000 resamples, seed 42) per metric and configuration.
8. Paired Wilcoxon signed-rank tests of each configuration against M3, pairing B-scans by `(tomogram_file, bscan_index)`. Holm-Bonferroni correction across all 42 tests (7 metrics x 6 comparisons) as one family.

Outputs in `--out-dir`: `per_bscan_metrics.csv`, `summary_metrics.csv`, `wilcoxon_tests.csv`, `evaluation_errors.log`, `plots/training_curves_unified/` and `plots/metric_boxplots/`.

**Statistical unit.** Tests and confidence intervals treat B-scans as samples. Several B-scans come from the same volume, so they are not independent, and the statistics do not support subject-level conclusions.

## What reproduces exactly

The tests in `tests/` compare the refactored code against the original scripts in `legacy/` (golden tests) with fixed random inputs on CPU:

- preprocessing, datasets, architectures, losses and metrics: identical outputs;
- one PC-CGAN training epoch (`train_fn`): identical losses and weights;
- one diffusion training epoch (`train_diffusion` vs. the original `train`): identical checkpoint and EMA;
- the DDPM sampler: identical samples for the same RNG state;
- the full PC-CGAN evaluation: identical `per_bscan_metrics.csv` (except the corrected `despeckle_filter` label), `summary_metrics.csv` and `wilcoxon_tests.csv`.

Run them with `pytest`.

## What does not reproduce exactly

- The PC-CGAN train/validation partition of the original runs (unseeded shuffle).
- Bitwise GPU results (non-deterministic kernels, AMP, PyTorch version).
- Diffusion samples of the original runs: the original sampler was not seeded.
