# Trained weights

Weights are not stored in this repository. They are available upon request from the corresponding author (see the main `README.md`).

Expected file names:

| Model | Paper name | File | Format |
|---|---|---|---|
| PC-CGAN reference | M3 | `pccgan_M3_G_best.pth` | `state_dict` of `octmirror.models.pccgan.UNetGenerator(in_ch=2, out_ch=2)` (the `weight/G_best.pth` file written by `scripts/train_pccgan.py`). |
| Conditional diffusion (no phase loss) | D1 | `diffusion_D1_checkpoint.pth` | Dictionary with keys `weights`, `optimizer`, `ema` (the `checkpoint.pth` written by `scripts/train_diffusion.py`). Keys may carry a `module.` prefix if trained with `DataParallel`; `octmirror.inference.load_diffusion` removes it. |

`*.pth` and `*.pt` files are ignored by git.
