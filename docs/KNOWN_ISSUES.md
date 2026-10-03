# Known issues

This repository is a refactoring of the scripts used for the paper (kept unchanged in `legacy/`). The numerical behaviour of the original code was preserved on purpose, including the issues below. Golden tests in `tests/` check the refactored functions against the originals.

Each entry gives the location in this package, the location in the original scripts, a description, and the impact.

Original scripts:
- `legacy/torchPCCGAN_AblationM2_M8.py`: PC-CGAN training (M2-M8).
- `legacy/evaluate_ablation_unified.py`: PC-CGAN evaluation.
- `legacy/diffComplexField_ROI256_D1_D4.py`: diffusion training and sampling (D1-D4).
- `legacy/complex_field_utils.py`: shared preprocessing.

## 1. Issues in the original code (kept as is)

### 1.1 Phase losses and validation metrics use the shifted [0, 1] representation

- **Files / functions:** `octmirror/losses.py` (`phase_circular_loss`, `phase_gradient_loss`), `octmirror/training.py` (`validate_fn`, phase term in `train_diffusion`). Originals: same names in `torchPCCGAN_AblationM2_M8.py` and `complex_field_utils.py`; `train()` in `diffComplexField_ROI256_D1_D4.py`.
- **Description:** `log_scale` maps the real and imaginary parts from [-1, 1] to [0, 1] (`(x + 1) / 2`). The phase losses compute `atan2(imag, real)` and the weights `sqrt(real**2 + imag**2)` directly on these [0, 1] channels, without mapping them back to [-1, 1]. The measured angle is therefore the angle of the point `((Re+1)/2, (Im+1)/2)` around the origin, which lies in [0, pi/2] and depends on both amplitude and phase of the normalized field. It is not the phase of the field. The amplitude loss in `train_fn` does map back to [-1, 1] first (`_compute_amp`), so the two terms use different conventions. The same applies to `amp_mae` and `phase_mae` in `validate_fn`, and to the phase loss on the predicted x0 in the diffusion training (D2-D4).
- **Impact:** the "phase" term constrains a quantity that mixes amplitude and phase. The M2-M8 and D1-D4 ablations measure the effect of this term as implemented. `val_amp_mae` (used to select `G_best.pth` and to drive the learning-rate scheduler) is the MAE of the modulus of the shifted vector, not of the normalized amplitude. Check that the description of the loss in the paper matches this implementation.

### 1.2 PC-CGAN EMA: updated once per epoch, parameters only, and it drives model selection

- **Files / functions:** `octmirror/training.py` (`update_ema`, `train_loop`, `validate_fn`). Originals: same names in `torchPCCGAN_AblationM2_M8.py`.
- **Description:**
  - `update_ema` is called once per epoch, not once per iteration. With `decay = 0.999` and 300 epochs, the weight left on the initial (random) parameters at the end of training is `0.999**300 = 0.74`. The EMA generator stays close to its random initialization for the whole run.
  - Only parameters are averaged. BatchNorm buffers (`running_mean`, `running_var`) of the EMA copy keep the values they had when it was created (before training), and the EMA generator is evaluated in `eval()` mode with those statistics.
  - With `use_ema=True` (all configurations), `validate_fn` evaluates the EMA generator. Its `val_amp_mae` decides when `weight/G_best.pth` is written and is the quantity monitored by `ReduceLROnPlateau`.
  - `G_best.pth` and `D_best.pth` contain the raw (non-EMA) networks at the epoch where the EMA generator had its lowest `val_amp_mae`.
- **Impact:** the epoch of `G_best.pth` and the learning-rate reductions are driven by a model that is mostly the random initialization. The reported PC-CGAN results use `G_best.pth`, which is a raw generator and does not contain the EMA weights. The original evaluation script already dropped `G_ema_best` for this reason ("updated only once per epoch, so it never converged"). The refactored code no longer writes `G_ema_best.pth`; periodic `G_ema{epoch}.pth` and `G_ema_final.pth` are still written.

### 1.3 Train/validation split before seeding, and per B-scan

- **Files / functions:** `octmirror/datasets.py` (`build_pccgan_datasets`), `scripts/train_pccgan.py`. Original: module-level code in `torchPCCGAN_AblationM2_M8.py`.
- **Description:** the original script calls `random.shuffle(indices)` before any seed is set (the seed is set later, per configuration), so the 90/10 split of the original runs cannot be reproduced. The refactored code seeds the split with `split_seed` (0 in all configs), which produces a different partition from the original runs. `split_seed: null` restores the unseeded behaviour. The split is per B-scan: B-scans of all volumes are pooled before splitting, so neighbouring B-scans of the same volume appear in both training and validation.
- **Impact:** retraining with this code gives a different internal validation set than the original runs. The internal validation set is optimistic (correlated with training data). It only affects model selection and the learning-rate schedule; the results in the paper are computed on a separate, independent validation set.

### 1.4 File order depends on the file system

- **Files / functions:** `octmirror/datasets.py` (`build_pccgan_datasets`, `build_diffusion_dataset`). Originals: module-level code in the two training scripts.
- **Description:** volumes are read in `os.listdir` order, which is not guaranteed to be the same across file systems or machines. The order determines the concatenation of B-scans and therefore which B-scans the seeded split assigns to training and validation (PC-CGAN).
- **Impact:** the same `split_seed` can give different partitions on different machines. The diffusion dataset is shuffled by the DataLoader, so only the batch composition changes.

### 1.5 Phase of the minimum-amplitude pixel is lost in `log_scale`

- **Files / functions:** `octmirror/preprocessing.py` (`log_scale`, `inverse_log_scale`). Originals: `logScale`, `inverseLogScale` in all scripts.
- **Description:** the minimum log amplitude of each slice is normalized to 0, so that pixel becomes `0 + 0j` and its phase is lost; after `inverse_log_scale` it gets amplitude `10**smin` and phase 0. A pixel of exactly zero amplitude gives `log10(0) = -inf` and NaNs.
- **Impact:** one pixel per B-scan in the reference fields used for metrics; negligible on the metrics. The round-trip test excludes that pixel.

### 1.6 Inverse normalization: clipped for diffusion, not clipped for PC-CGAN

- **Files / functions:** `octmirror/preprocessing.py` (`inverse_log_scale(..., max_log_amp)`). Originals: `inverseLogScale` in `complex_field_utils.py` (clip at 30.0) and in the PC-CGAN scripts (no clip).
- **Description:** both variants are kept behind the `max_log_amp` argument: `30.0` for diffusion, `None` for PC-CGAN.
- **Impact:** none for normal amplitudes; the clip only prevents overflow when a diffusion sample has a very large normalized amplitude.

### 1.7 Different mirror-artifact simulation for PC-CGAN and diffusion

- **Files / functions:** `octmirror/datasets.py`, `scripts/infer.py`, `scripts/evaluate_pccgan.py`. Originals: data loading in both training scripts, `build_validation_samples` in the evaluation script.
- **Description:** PC-CGAN training and evaluation simulate the artifact on the full volume (FFT over the full depth, 512 samples) and then take B-scans. Diffusion training simulates the artifact on the 256 x 256 central ROI (FFT over 256 depth samples). These are not equivalent: the artifact of the ROI is not the ROI of the full-depth artifact. `infer.py` follows the protocol of each model. With `--no-simulate-mirror`, the diffusion model receives the central crop of an artifact produced over the full depth, which differs from its training distribution.
- **Impact:** diffusion results on real corrupted data may differ from those obtained with the simulated protocol.

### 1.8 Diffusion training uses every file and has no validation split

- **Files / functions:** `octmirror/datasets.py` (`build_diffusion_dataset`). Original: `__main__` block of `diffComplexField_ROI256_D1_D4.py`.
- **Description:** the variable `filestrain` was defined but never used: every file in `noPhase/`, `phase/` and `synthetic/` is loaded. The refactored code keeps this and only adds a `.npy` extension filter (the original called `np.load` on every file and would fail on any other file). There is no validation split and no model selection: the final checkpoint is used.
- **Impact:** the diffusion training set is larger than the PC-CGAN training set and includes the folders `noPhase/` and `synthetic/`. Check that none of these files belong to the independent validation set.

### 1.9 The original diffusion sampler cannot load a `DataParallel` EMA

- **Files / functions:** `octmirror/inference.py` (`load_diffusion`). Original: `inference()` in `diffComplexField_ROI256_D1_D4.py`.
- **Description:** with more than one GPU, training wraps the model in `nn.DataParallel` before creating `ModelEmaV3`, so the EMA keys are `module.module.<name>`. The original `inference()` removed `module.` from `weights` only, then built the EMA around a plain `UNET`, and `ema.load_state_dict(checkpoint['ema'])` fails (`tests/test_inference.py::test_legacy_inference_cannot_load_data_parallel_ema`). `load_diffusion` removes the EMA wrapper prefix and the `DataParallel` prefix and loads the weights directly into a `UNET`.
- **Impact:** none on the numbers. Sampling with the EMA weights is the same as in the original (`ema.module`).

### 1.10 Despeckling filter label

- **Files / functions:** `scripts/evaluate_pccgan.py`, `octmirror/metrics.py` (`DESPECKLE_FILTER_NAME`). Original: `DESPECKLE_FILTER_NAME = 'nlm_h020'` in `evaluate_ablation_unified.py`.
- **Description:** the original evaluation wrote `despeckle_filter = 'nlm_h020'` in `per_bscan_metrics.csv`, but the filter was applied with `h = 0.1` (default of `despeckle_nlm`). The label was wrong; the values were computed with `h = 0.1`. The refactored evaluation writes `nlm_h010`. `h = 0.1` is used everywhere in this repository.
- **Impact:** only the label in the CSV. Metric values are unchanged (golden test `tests/test_scripts.py::test_evaluate_pccgan_reproduces_legacy_outputs`).

## 2. Changes made during the refactoring

These changes do not alter the numerical results of the configurations used in the paper. Each one is covered by a golden test unless stated otherwise.

| Change | Location | Original | Notes |
|---|---|---|---|
| `random.shuffle` of the split is seeded (`split_seed`) | `datasets.build_pccgan_datasets` | unseeded | Changes the partition; see 1.3. |
| Unsupported phase weighting modes raise `ValueError` | `losses.phase_circular_loss`, `losses.phase_gradient_loss` | `'real'`, `'fake'` existed; any other string silently used uniform weights | Only `'geo'` and `'none'` were used in the paper. |
| `verbose=True` removed from `ReduceLROnPlateau` | `training.train_loop` | `verbose=True` | The argument was removed from recent PyTorch (it raises `TypeError` on PyTorch 2.14). Learning-rate changes are no longer printed. |
| `torch.backends.cuda.sdp_kernel(...)` replaced by `torch.nn.attention.sdpa_kernel([...])` | `models/diffusion.py` (`Attention`) | deprecated context manager | Same enabled backends (flash, memory-efficient, math, cuDNN) and identical output on CPU (`tests/test_models.py`). In recent PyTorch the old API calls the new one with exactly this list. The kernel actually picked on a GPU depends on the PyTorch version and hardware, so bitwise GPU equality with the original runs is not guaranteed. |
| Discriminator call hack removed | `training.train_fn` | `D(...) if D.forward.__code__.co_argcount == 3 else ...` | Both discriminators use `forward(real, fake)`; the call is `D(real=input, fake=...)`, the branch that was always taken. |
| Device, seeds and paths passed as arguments | everywhere | module-level globals (`device`, hard-coded `/home/...` paths) | `train_fn` takes `device`; diffusion training uses `device` instead of `.cuda()`. |
| Mixed precision enabled when the device is CUDA | `training.train_diffusion` | enabled when `torch.cuda.is_available()` | Same on a GPU run. |
| GPU diagnostics, checkpoint resume, interpolation mode and the post-training reverse-trajectory figure removed | diffusion training | `diagnose_gpu_and_model`, `diagnose_batch_forward`, `checkpoint_path`, `interpolate`, `inference()` + `display_reverse` | `reverse.png` is no longer written after training. |
| Unused code removed | PC-CGAN training and evaluation | `Generator`, `Discriminator`, `show_img_sample`, `show_losses`, BCE branch (`use_hinge=False`), `criterion_bce`, `LAMBDA`, `G_ema_best`, `build_checkpoint_comparison` | With `G_best` as the only checkpoint, `build_checkpoint_comparison` always selected `G_best`; `checkpoint_comparison.csv` is no longer written. |
| One run per configuration | `scripts/train_pccgan.py` | loop over the 7 configurations in one process | Each configuration seeds with `seed: 0` before building the networks, as in the original loop. `ablation_summary.json` is updated per run. |
| `ax.boxplot(labels=...)` replaced by `tick_labels=` | `visualization.build_metric_boxplots` | `labels=` | `labels` was removed in Matplotlib 3.11 (the original raises `TypeError`, which it logged). Requires `matplotlib>=3.9`. |
| Diffusion sampler seeded explicitly | `inference.sample_ddpm` | unseeded | `torch.manual_seed(seed)` is called right before the initial noise. The original built a `UNET()` between any seed and the noise, so samples do not match an original run with the same seed; the update equations are identical (`tests/test_inference.py`). `infer.py` uses the same seed for every B-scan. |
| Diffusion checkpoint loading strips `module.` prefixes | `inference.load_diffusion` | see 1.9 | |

## 3. Checked and found correct

- **Discriminator step and `torch.no_grad()`:** in `train_fn`, only `fake_detached = fake_img.detach()` is inside `torch.no_grad()`. The calls `D(input, target)` and `D(input, fake_detached)` are outside it, so `loss_d.backward()` works and the discriminator receives gradients. No change was needed.
- **M2 (`lambda_phase = 0`):** the phase loss is still computed and multiplied by zero (`use_phase=True` in all configurations). This has no effect on the gradients.

## 4. Not included in this release

- Evaluation of the diffusion ablations: `octmirror.inference.evaluate_diffusion` is a stub that raises `NotImplementedError` (TODO).
- Hilbert-transform analytic-signal baseline: `octmirror.baselines.hilbert_analytic_signal` is a stub that raises `NotImplementedError` (TODO).
- `infer.py --compute-metrics` with `--model diffusion` applies the PC-CGAN evaluation protocol (metrics on the 256 x 256 ROI). It is not the protocol used for the diffusion results in the paper.
