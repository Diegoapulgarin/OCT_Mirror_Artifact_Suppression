#%%
"""
Evaluation-only script for the PCCGAN ablation study (M2-M8).

This script does NOT train anything. It walks the results already produced by
torchPix2Pix_AblationM2_M8.py (checkpoints + metrics_epoch.csv + ablation_summary.json)
and produces the unified evaluation CSVs/plots requested for the rebuttal.

The GeneratorTf / DiscriminatorTf classes and mirrorArtifact / logScale / inverseLogScale
functions are copied verbatim from torchPix2Pix_AblationM2_M8.py (that script is not
structured as an importable module, running top-level training code on import). 
"""
import os
import json
import traceback
from glob import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from numpy.fft import fft, ifft, fftshift, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means
import cv2
from scipy.stats import wilcoxon

# ===================== COPIED VERBATIM FROM torchPix2Pix_AblationM2_M8.py ===================== #

def mirrorArtifact(array):
    '''
    Introduce mirror artifacts in the 3D input array along axis = 0.
    using numpy fft operations.
    '''
    fringes = ifftshift(ifft(ifftshift(array, axes=0), axis=0), axes=0)
    tomcc = fftshift(fft(fftshift(fringes.real, axes=0), axis=0), axes=0)
    return tomcc

def logScale(slices):

    logslices = np.copy(slices)
    nSlices = slices.shape[0]
    if len(slices.shape) == 4:
        logslicesAmp = abs(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
        logslicesPhase = np.angle(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    else:
        logslicesAmp = abs(logslices)
        logslicesPhase = np.angle(logslices)
    # and retrieve the phase
    # reescale amplitude
    logslicesAmp = np.log10(logslicesAmp)
    slicesMax = np.reshape(logslicesAmp.max(axis=(1, 2)), (nSlices, 1, 1))
    slicesMin = np.reshape(logslicesAmp.min(axis=(1, 2)), (nSlices, 1, 1))
    logslicesAmp = (logslicesAmp - slicesMin) / (slicesMax - slicesMin)
    # --- here, we could even normalize each slice to 0-1, keeping the original
    # --- limits to rescale after the network processes
    # and redefine the real and imaginary components with the new amplitude and
    # same phase
    logslicesReal = (np.real(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    logslicesImag = (np.imag(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    logslices = np.stack((logslicesReal, logslicesImag), axis=-1)
    return logslices, slicesMax, slicesMin, logslicesAmp, logslicesPhase

def inverseLogScale(oldslices, slicesMax, slicesMin):

    slices = np.copy(oldslices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesPhase = np.angle(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesAmp = slicesAmp * (slicesMax - slicesMin) + slicesMin
    slicesAmp = 10**(slicesAmp)
    slices[:, :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices

class GeneratorTf(nn.Module):
    """
    U-Net pix2pix replicando tu versión TensorFlow:
    Encoder: 6 bloques stride2 + bottleneck
    Decoder: 6 bloques con skip + salida
    """
    def __init__(self, in_ch=2, out_ch=2, use_dropout=True):
        super().__init__()
        self.use_dropout = use_dropout

        def enc_block(in_c, out_c, apply_bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not apply_bn)]
            if apply_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def dec_block(in_c, out_c, apply_dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder (e1 sin BN)
        self.e1 = enc_block(in_ch, 64, apply_bn=False)   # -> 64 -> 512 x 512 x 64
        self.e2 = enc_block(64, 128)                     # -> 128 -> 256 x 256 x 128
        self.e3 = enc_block(128, 256)                    # -> 256 -> 128 x 128 x 256
        self.e4 = enc_block(256, 512)                    # -> 512 -> 64 x 64 x 512
        self.e5 = enc_block(512, 512)                    # -> 512 -> 32 x 32 x 512
        self.e6 = enc_block(512, 512)                    # -> 512 -> 16 x 16 x 512
        # Bottleneck (conv stride2 + ReLU, sin BN)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True), # -> 512 -> 8 x 8 x 512
            nn.ReLU(inplace=True)
        )

        # Decoder (con cat skip)
        self.d1 = dec_block(512, 512, apply_dropout=use_dropout)      # cat con e6 -> 16 x 16 x 512
        self.d2 = dec_block(512+512, 512, apply_dropout=use_dropout)  # cat con e5 -> 32 x 32 x 512
        self.d3 = dec_block(512+512, 512, apply_dropout=use_dropout)  # cat con e4 -> 64 x 64 x 512
        self.d4 = dec_block(512+512, 256, apply_dropout=False)        # cat con e3 -> 128 x 128 x 256
        self.d5 = dec_block(256+256, 128, apply_dropout=False)        # cat con e2 -> 256 x 256 x 128
        self.d6 = dec_block(128+128, 64, apply_dropout=False)         # cat con e1 -> 512 x 512 x 64

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(64+64, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()   # Output in [0,1] to match normalized log-scale representation
        )

    def forward(self, x):
        e1 = self.e1(x)   # 64
        e2 = self.e2(e1)  # 128
        e3 = self.e3(e2)  # 256
        e4 = self.e4(e3)  # 512
        e5 = self.e5(e4)  # 512
        e6 = self.e6(e5)  # 512
        b  = self.bottleneck(e6)  # 512 (más pequeño)

        d1 = self.d1(b)
        d1 = torch.cat([d1, e6], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e5], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e4], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e3], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e2], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e1], dim=1)
        out = self.out_conv(d6)
        return out

class DiscriminatorTf(nn.Module):
    """
    PatchGAN alineado con tu versión TF:
    Conv(64,s2)->Conv(128,s2)->Conv(256,s2)->Conv(512,s2)->Conv(512,s1)->Conv(1,s1)
    Activaciones ReLU (como tu TF), BN en todas menos la primera.
    """
    def __init__(self, in_ch=2):
        super().__init__()
        ch_in = in_ch * 2  # concat real + fake
        def disc_block(in_c, out_c, stride, apply_bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=not apply_bn)]
            if apply_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))  # Usas ReLU en TF (original pix2pix usa LeakyReLU)
            return nn.Sequential(*layers)

        self.c1 = disc_block(ch_in, 64, stride=2, apply_bn=False)
        self.c2 = disc_block(64, 128, stride=2)
        self.c3 = disc_block(128, 256, stride=2)
        self.c4 = disc_block(256, 512, stride=2)   
        self.c5 = disc_block(512, 512, stride=1)
        self.out = nn.Conv2d(512, 1, 4, stride=1, padding=1)

    def forward(self, real, fake):
        x = torch.cat([real, fake], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        return self.out(x)

# ===================== END COPIED CODE ===================== #


# ===================== METRIC FUNCTIONS (exact, do not modify) ===================== #

def simcos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def calculate_ssim(image1, image2):
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño."
    ssim_value, _ = ssim(image1, image2, full=True, data_range=1)
    return ssim_value

def calculate_mse(image_original, image_reconstructed):
    assert image_original.shape == image_reconstructed.shape, "Las imágenes deben tener el mismo tamaño."
    mse = np.mean((image_original - image_reconstructed) ** 2)
    return mse

def calculate_psnr(image1, image2, max_val=1.0):
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño."
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(max_val**2 / mse)
    return psnr

def histogram_difference(image1, image2, method="cosine-similarity"):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 1])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 1])
    if method == "kullback-leibler":
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
    if method == "chi-squared":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    elif method == "kullback-leibler":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
    elif method == 'cosine-similarity':
        return simcos(np.ravel(hist1), np.ravel(hist2))
    else:
        raise ValueError("Método no reconocido")

def weighted_phase_coherence(pred_cx, gt_cx):
    w    = np.abs(pred_cx) * np.abs(gt_cx)
    diff = np.angle(pred_cx) - np.angle(gt_cx)
    return float(np.sum(w * np.cos(diff)) / (np.sum(w) + 1e-12))

def masked_complex_coherence(pred_cx, gt_cx, percentile=50):
    mask   = np.abs(gt_cx) > np.percentile(np.abs(gt_cx), percentile)
    d1, d2 = pred_cx[mask], gt_cx[mask]
    piston = np.angle(np.sum(d1 * np.conj(d2)))
    d1c    = d1 * np.exp(-1j * piston)
    num    = np.abs(np.sum(d1c * np.conj(d2)))
    den    = np.sqrt(np.sum(np.abs(d1c)**2) * np.sum(np.abs(d2)**2))
    return float(num / (den + 1e-12))

def phase_gradient_ssim(pred_cx, gt_cx, percentile=50):
    mask     = np.abs(gt_cx) > np.percentile(np.abs(gt_cx), percentile)
    phi_pred = np.angle(pred_cx)
    phi_gt   = np.angle(gt_cx)
    dpz_pred = np.angle(np.exp(1j * np.diff(phi_pred, axis=0, append=phi_pred[-1:])))
    dpx_pred = np.angle(np.exp(1j * np.diff(phi_pred, axis=1, append=phi_pred[:, -1:])))
    dpz_gt   = np.angle(np.exp(1j * np.diff(phi_gt,   axis=0, append=phi_gt[-1:])))
    dpx_gt   = np.angle(np.exp(1j * np.diff(phi_gt,   axis=1, append=phi_gt[:, -1:])))
    for arr in [dpz_pred, dpx_pred, dpz_gt, dpx_gt]:
        arr[~mask] = 0.0
    sz = ssim(dpz_pred, dpz_gt, data_range=2*np.pi)
    sx = ssim(dpx_pred, dpx_gt, data_range=2*np.pi)
    return float((sz + sx) / 2.0), float(sz), float(sx)

# ===================== END METRIC FUNCTIONS ===================== #


# ===================== CONFIGURATION ===================== #

ABLATION_ROOT = '/home/rsg-dapulgaris/Models/AblationPCCGAN'
VALIDATION_DATA_DIR = '/home/rsg-dapulgaris/Data/validationCxPaper'
EVALUATION_DIR = os.path.join(ABLATION_ROOT, 'evaluation')
PLOTS_DIR = os.path.join(EVALUATION_DIR, 'plots')
TRAINING_CURVES_PLOTS_DIR = os.path.join(PLOTS_DIR, 'training_curves_unified')
BOXPLOTS_DIR = os.path.join(PLOTS_DIR, 'metric_boxplots')
ERROR_LOG_PATH = os.path.join(EVALUATION_DIR, 'evaluation_errors.log')

CONFIG_NAMES = [
    "M2_ComplexCGAN_NoPhase",
    "M3_PCCGAN_Reference",
    "M4_NoAmplitudeWeighting",
    "M5_PhaseWeight_Low",
    "M6_PhaseWeight_MidHigh",
    "M7_PhaseWeight_High",
    "M8_PhaseGradientConstraint",
]
REFERENCE_CONFIG = "M3_PCCGAN_Reference"

# G_ema_best dropped: a confirmed EMA bug updated it only once per epoch, so it never converged.
CHECKPOINT_TYPES = ['G_best']
N_BSCANS_PER_VOLUME = 40
DESPECKLE_FILTER_NAME = 'nlm_h020'
METRIC_COLUMNS = ['ssim', 'psnr', 'mse', 'hist_cosine_similarity', 'wpc', 'ccc', 'pgssim']
TRAINING_CURVE_METRICS = ['loss_g', 'val_amp_mae', 'val_phase_mae', 'train_phase', 'train_phase_grad', 'train_amp']

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 42

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def log_error(message):
    with open(ERROR_LOG_PATH, 'a') as f:
        f.write(message + "\n")
    print(message) 


def load_tomogram_complex(filepath):
    """
    Fixed axis convention (no runtime detection): tomograms are stored as (Z, X, Y).
    3D array -> already complex (Z, X, Y).
    4D array -> (Z, X, Y, 2) real/imag channels, combined into complex (Z, X, Y).
    """
    raw = np.load(filepath)
    original_shape = raw.shape
    if raw.ndim == 4:
        tom = raw[..., 0] + 1j * raw[..., 1]
    else:
        tom = raw
    print(f"  Loaded {os.path.basename(filepath)}: original shape {original_shape} -> complex shape {tom.shape} (Z, X, Y)")
    return tom


def select_bscan_indices(num_bscans):
    if num_bscans <= N_BSCANS_PER_VOLUME:
        return np.arange(num_bscans)
    idx = np.linspace(0, num_bscans - 1, N_BSCANS_PER_VOLUME, dtype=int)
    return np.unique(idx)


def build_validation_samples():
    """
    Returns a list of dicts: {tomogram_file, bscan_index, input_cx (Z,X) w/ mirror artifact,
    target_cx (Z,X) clean}, sorted by (tomogram_file, bscan_index) for reproducible pairing.
    """
    samples = []
    files = sorted(glob(os.path.join(VALIDATION_DATA_DIR, '*.npy')))
    for filepath in files:
        fname = os.path.basename(filepath)
        tom = load_tomogram_complex(filepath)  # (Z, X, Y) clean, complex
        tomcc = mirrorArtifact(tom)            # (Z, X, Y) with induced mirror artifact, FFT along axis 0 (Z)
        num_bscans = tom.shape[2]
        y_indices = select_bscan_indices(num_bscans)
        for y_idx in y_indices:
            bscan_clean = tom[:, :, y_idx]     # (Z, X)
            bscan_mirror = tomcc[:, :, y_idx]   # (Z, X)
            samples.append(dict(
                tomogram_file=fname,
                bscan_index=int(y_idx),
                input_cx=bscan_mirror,
                target_cx=bscan_clean,
            ))
    samples.sort(key=lambda s: (s['tomogram_file'], s['bscan_index']))
    return samples


def evaluate_single_bscan(G, input_cx, target_cx):
    """
    Runs one (input, target) B-scan pair through the generator and computes the 7 Table-1 metrics
    plus the axial/lateral phase-gradient SSIM components.
    """
    logslices_in, _, _, _, _ = logScale(input_cx[np.newaxis, ...])
    logslices_tgt, smax_tgt, smin_tgt, _, _ = logScale(target_cx[np.newaxis, ...])

    input_tensor = torch.tensor(
        logslices_in[0].transpose(2, 0, 1), dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_img = G(input_tensor)  # (1, 2, Z, X)

    fake_np = fake_img[0].detach().cpu().numpy().transpose(1, 2, 0)[np.newaxis, ...]  # (1, Z, X, 2)

    # Reconstruct both prediction and ground truth using the TARGET's smax/smin
    pred_rec = inverseLogScale(fake_np, smax_tgt, smin_tgt)
    gt_rec = inverseLogScale(logslices_tgt, smax_tgt, smin_tgt)

    pred_cx = pred_rec[0, :, :, 0] + 1j * pred_rec[0, :, :, 1]
    gt_cx = gt_rec[0, :, :, 0] + 1j * gt_rec[0, :, :, 1]

    intensity_pred = 20 * np.log10(np.abs(pred_cx) + 1e-12)
    intensity_gt = 20 * np.log10(np.abs(gt_cx) + 1e-12)

    gt_min = intensity_gt.min()
    gt_max = intensity_gt.max()
    norm_pred = np.clip((intensity_pred - gt_min) / (gt_max - gt_min), 0, 1)
    norm_gt = np.clip((intensity_gt - gt_min) / (gt_max - gt_min), 0, 1)

    # Despeckling applies only to intensity metrics; phase metrics below use raw pred_cx/gt_cx.
    norm_pred_filtered = despeckle_nlm(norm_pred)
    norm_gt_filtered = despeckle_nlm(norm_gt)

    ssim_val = calculate_ssim(norm_pred_filtered, norm_gt_filtered)
    psnr_val = calculate_psnr(norm_pred_filtered, norm_gt_filtered)
    mse_val = calculate_mse(norm_pred_filtered, norm_gt_filtered)
    hist_val = histogram_difference(norm_pred_filtered, norm_gt_filtered, method='cosine-similarity')
    wpc_val = weighted_phase_coherence(pred_cx, gt_cx)
    ccc_val = masked_complex_coherence(pred_cx, gt_cx)
    pgssim_val, pgssim_axial, pgssim_lateral = phase_gradient_ssim(pred_cx, gt_cx)

    return dict(
        ssim=ssim_val,
        psnr=psnr_val,
        mse=mse_val,
        hist_cosine_similarity=hist_val,
        wpc=wpc_val,
        ccc=ccc_val,
        pgssim=pgssim_val,
        pgssim_axial=pgssim_axial,
        pgssim_lateral=pgssim_lateral,
    )



def despeckle_nlm(image, h=0.1, patch_size=5, patch_distance=6):
    return denoise_nl_means(image, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)


def bootstrap_ci(data, n_boot=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.nan, np.nan
    n = len(data)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)
    return float(lower), float(upper)


def holm_bonferroni(pvalues):
    """
    Manual Holm-Bonferroni step-down correction.
    Sort p-values ascending, adjust each by (n - rank), enforce monotonicity via running max.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvalues[idx] * (n - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


# ===================== MAIN EVALUATION ===================== #

def run_full_evaluation():
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    os.makedirs(TRAINING_CURVES_PLOTS_DIR, exist_ok=True)
    os.makedirs(BOXPLOTS_DIR, exist_ok=True)
    open(ERROR_LOG_PATH, 'a').close()

    print("Building validation samples from independent validation set...")
    samples = build_validation_samples()
    print(f"Total B-scan samples per checkpoint evaluation: {len(samples)}")

    per_bscan_rows = []

    for config_name in CONFIG_NAMES:
        config_dir = os.path.join(ABLATION_ROOT, config_name)
        for checkpoint_type in CHECKPOINT_TYPES:
            try:
                weight_path = os.path.join(config_dir, 'weight', f'{checkpoint_type}.pth')
                if not os.path.exists(weight_path):
                    raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

                G = GeneratorTf(in_ch=2, out_ch=2)
                state_dict = torch.load(weight_path, map_location=device)
                G.load_state_dict(state_dict)
                G.to(device)
                G.eval()

                for sample in samples:
                    try:
                        with torch.no_grad():
                            metrics = evaluate_single_bscan(G, sample['input_cx'], sample['target_cx'])
                        per_bscan_rows.append(dict(
                            config=config_name,
                            checkpoint_type=checkpoint_type,
                            tomogram_file=sample['tomogram_file'],
                            bscan_index=sample['bscan_index'],
                            despeckle_filter=DESPECKLE_FILTER_NAME,
                            **metrics
                        ))
                    except Exception as ex_sample:
                        log_error(
                            f"[{config_name}/{checkpoint_type}] Failed on "
                            f"{sample['tomogram_file']} bscan {sample['bscan_index']}: {ex_sample}\n"
                            f"{traceback.format_exc(limit=5)}"
                        )
                print(f"Finished evaluation: {config_name} / {checkpoint_type}")

            except Exception as ex_ckpt:
                log_error(
                    f"[{config_name}/{checkpoint_type}] Checkpoint evaluation failed: {ex_ckpt}\n"
                    f"{traceback.format_exc(limit=5)}"
                )

    per_bscan_df = pd.DataFrame(per_bscan_rows, columns=[
        'config', 'checkpoint_type', 'tomogram_file', 'bscan_index', 'despeckle_filter',
        'ssim', 'psnr', 'mse', 'hist_cosine_similarity', 'wpc', 'ccc',
        'pgssim', 'pgssim_axial', 'pgssim_lateral'
    ])
    per_bscan_path = os.path.join(EVALUATION_DIR, 'per_bscan_metrics.csv')
    per_bscan_df.to_csv(per_bscan_path, index=False)
    print(f"Saved {per_bscan_path} ({len(per_bscan_df)} rows)")

    build_summary_metrics(per_bscan_df)
    recommended_checkpoints = build_checkpoint_comparison(per_bscan_df)
    build_wilcoxon_tests(per_bscan_df, recommended_checkpoints)
    build_training_curve_plots()
    build_metric_boxplots(per_bscan_df, recommended_checkpoints)


def build_summary_metrics(per_bscan_df):
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
    summary_path = os.path.join(EVALUATION_DIR, 'summary_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


def build_checkpoint_comparison(per_bscan_df):
    """
    Compares G_best vs G_ema_best per config. Returns dict config_name -> recommended checkpoint
    (falls back to 'G_ema_best' when SSIM/WPC disagree, flagged as 'revisar manualmente').
    """
    rows = []
    recommended = {}
    for config_name in CONFIG_NAMES:
        try:
            row = dict(config=config_name)
            means = {}
            for checkpoint_type in CHECKPOINT_TYPES:
                subset = per_bscan_df[
                    (per_bscan_df['config'] == config_name) &
                    (per_bscan_df['checkpoint_type'] == checkpoint_type)
                ]
                means[checkpoint_type] = {m: float(np.nanmean(subset[m].values)) if len(subset) else np.nan
                                           for m in METRIC_COLUMNS}
                for metric in METRIC_COLUMNS:
                    row[f'{metric}_{checkpoint_type}'] = means[checkpoint_type][metric]

            best_ssim_ckpt = max(CHECKPOINT_TYPES, key=lambda c: means[c]['ssim']) \
                if all(np.isfinite(means[c]['ssim']) for c in CHECKPOINT_TYPES) else None
            best_wpc_ckpt = max(CHECKPOINT_TYPES, key=lambda c: means[c]['wpc']) \
                if all(np.isfinite(means[c]['wpc']) for c in CHECKPOINT_TYPES) else None

            if best_ssim_ckpt is not None and best_ssim_ckpt == best_wpc_ckpt:
                recommendation = best_ssim_ckpt
            else:
                recommendation = "revisar manualmente"

            row['recommended_checkpoint'] = recommendation
            recommended[config_name] = recommendation if recommendation != "revisar manualmente" else 'G_ema_best'
            rows.append(row)
        except Exception as ex:
            log_error(f"[{config_name}] checkpoint_comparison failed: {ex}\n{traceback.format_exc(limit=5)}")
            recommended[config_name] = 'G_ema_best'

    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(EVALUATION_DIR, 'checkpoint_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved {comparison_path}")
    return recommended


def build_wilcoxon_tests(per_bscan_df, recommended_checkpoints):
    """
    Wilcoxon paired tests between M3_PCCGAN_Reference and each other config, per metric.
    Uses the checkpoint recommended for the reference config (M3) applied uniformly to all
    comparisons so both vectors in each paired test come from the same checkpoint type.
    Holm-Bonferroni correction is applied across ALL resulting p-values in this table (7 metrics x
    6 comparisons = 42 tests) as a single family.
    """
    primary_checkpoint = recommended_checkpoints.get(REFERENCE_CONFIG, 'G_ema_best')
    print(f"Wilcoxon tests use checkpoint_type='{primary_checkpoint}' for all configs (recommended for {REFERENCE_CONFIG}).")

    ref_df = per_bscan_df[
        (per_bscan_df['config'] == REFERENCE_CONFIG) &
        (per_bscan_df['checkpoint_type'] == primary_checkpoint)
    ][['tomogram_file', 'bscan_index'] + METRIC_COLUMNS]

    rows = []
    raw_pvalues = []
    row_refs = []

    for config_name in CONFIG_NAMES:
        if config_name == REFERENCE_CONFIG:
            continue
        try:
            other_df = per_bscan_df[
                (per_bscan_df['config'] == config_name) &
                (per_bscan_df['checkpoint_type'] == primary_checkpoint)
            ][['tomogram_file', 'bscan_index'] + METRIC_COLUMNS]

            merged = pd.merge(
                ref_df, other_df, on=['tomogram_file', 'bscan_index'],
                suffixes=('_ref', '_other')
            )

            for metric in METRIC_COLUMNS:
                x = merged[f'{metric}_ref'].values
                y = merged[f'{metric}_other'].values
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                try:
                    statistic, p_value = wilcoxon(x, y)
                except Exception as ex_test:
                    statistic, p_value = np.nan, np.nan
                    log_error(f"Wilcoxon failed for metric={metric}, config={config_name}: {ex_test}")

                row = dict(metric=metric, config_comparado=config_name,
                           statistic=statistic, p_value=p_value)
                rows.append(row)
                raw_pvalues.append(p_value if np.isfinite(p_value) else 1.0)
                row_refs.append(row)
        except Exception as ex:
            log_error(f"[{config_name}] wilcoxon comparison failed: {ex}\n{traceback.format_exc(limit=5)}")

    adjusted = holm_bonferroni(raw_pvalues) if raw_pvalues else []
    for row, adj_p in zip(row_refs, adjusted):
        row['p_value_holm_corrected'] = adj_p
        row['significant_0.05'] = bool(adj_p < 0.05)

    wilcoxon_df = pd.DataFrame(rows, columns=[
        'metric', 'config_comparado', 'statistic', 'p_value',
        'p_value_holm_corrected', 'significant_0.05'
    ])
    wilcoxon_path = os.path.join(EVALUATION_DIR, 'wilcoxon_tests.csv')
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"Saved {wilcoxon_path}")


def build_training_curve_plots():
    for metric in TRAINING_CURVE_METRICS:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            for config_name in CONFIG_NAMES:
                csv_path = os.path.join(ABLATION_ROOT, config_name, 'metrics_epoch.csv')
                if not os.path.exists(csv_path):
                    log_error(f"Missing metrics_epoch.csv for {config_name}, skipping in {metric} plot.")
                    continue
                df = pd.read_csv(csv_path)
                if metric not in df.columns:
                    log_error(f"Column {metric} missing in {csv_path}, skipping.")
                    continue
                ax.plot(df['epoch'], df[metric], label=config_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'Unified training curve: {metric}')
            ax.legend(fontsize=8)
            ax.grid(True)
            out_path = os.path.join(TRAINING_CURVES_PLOTS_DIR, f'unified_{metric}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
        except Exception as ex:
            log_error(f"Failed to build unified training curve plot for {metric}: {ex}\n{traceback.format_exc(limit=5)}")


def build_metric_boxplots(per_bscan_df, recommended_checkpoints):
    for metric in METRIC_COLUMNS:
        try:
            data_per_config = []
            labels = []
            for config_name in CONFIG_NAMES:
                checkpoint_type = recommended_checkpoints.get(config_name, 'G_ema_best')
                subset = per_bscan_df[
                    (per_bscan_df['config'] == config_name) &
                    (per_bscan_df['checkpoint_type'] == checkpoint_type)
                ]
                values = subset[metric].dropna().values
                data_per_config.append(values)
                labels.append(config_name)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data_per_config, labels=labels, showmeans=True)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} across configurations (recommended checkpoint per config)')
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            out_path = os.path.join(BOXPLOTS_DIR, f'boxplot_{metric}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
        except Exception as ex:
            log_error(f"Failed to build boxplot for {metric}: {ex}\n{traceback.format_exc(limit=5)}")


if __name__ == '__main__':
    run_full_evaluation()
