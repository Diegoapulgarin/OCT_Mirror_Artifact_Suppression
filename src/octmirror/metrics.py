"""Image and phase metrics used in the paper (exact implementations).

Intensity metrics (SSIM, PSNR, MSE, histogram cosine similarity) are computed on
normalized amplitude in dB after non-local-means despeckling (``h = 0.1``). Phase
metrics (WPC, CCC, PG-SSIM) are computed on the raw complex field.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means

NLM_H = 0.1
DESPECKLE_FILTER_NAME = 'nlm_h010'
METRIC_COLUMNS = ['ssim', 'psnr', 'mse', 'hist_cosine_similarity', 'wpc', 'ccc', 'pgssim']


def simcos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """SSIM with ``data_range=1``."""
    assert image1.shape == image2.shape, "Images must have the same shape."
    ssim_value, _ = ssim(image1, image2, full=True, data_range=1)
    return ssim_value


def calculate_mse(image_original: np.ndarray, image_reconstructed: np.ndarray) -> float:
    """Mean squared error."""
    assert image_original.shape == image_reconstructed.shape, "Images must have the same shape."
    mse = np.mean((image_original - image_reconstructed) ** 2)
    return mse


def calculate_psnr(image1: np.ndarray, image2: np.ndarray, max_val: float = 1.0) -> float:
    """PSNR in dB (``inf`` for identical images)."""
    assert image1.shape == image2.shape, "Images must have the same shape."
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(max_val**2 / mse)
    return psnr


def histogram_difference(image1: np.ndarray, image2: np.ndarray, method: str = "cosine-similarity") -> float:
    """Compare 256-bin histograms on [0, 1].

    Args:
        method: ``'cosine-similarity'`` (used in the paper), ``'chi-squared'`` or
            ``'kullback-leibler'``.
    """
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
        raise ValueError("Unrecognized method")


def weighted_phase_coherence(pred_cx: np.ndarray, gt_cx: np.ndarray) -> float:
    """Weighted phase coherence (WPC): amplitude-weighted mean of cos(phase difference)."""
    w    = np.abs(pred_cx) * np.abs(gt_cx)
    diff = np.angle(pred_cx) - np.angle(gt_cx)
    return float(np.sum(w * np.cos(diff)) / (np.sum(w) + 1e-12))


def masked_complex_coherence(pred_cx: np.ndarray, gt_cx: np.ndarray, percentile: float = 50) -> float:
    """Masked complex correlation coefficient (CCC) after removing the global phase piston.

    Only pixels whose target amplitude exceeds the given percentile are used.
    """
    mask   = np.abs(gt_cx) > np.percentile(np.abs(gt_cx), percentile)
    d1, d2 = pred_cx[mask], gt_cx[mask]
    piston = np.angle(np.sum(d1 * np.conj(d2)))
    d1c    = d1 * np.exp(-1j * piston)
    num    = np.abs(np.sum(d1c * np.conj(d2)))
    den    = np.sqrt(np.sum(np.abs(d1c)**2) * np.sum(np.abs(d2)**2))
    return float(num / (den + 1e-12))


def phase_gradient_ssim(pred_cx: np.ndarray, gt_cx: np.ndarray,
                        percentile: float = 50) -> Tuple[float, float, float]:
    """Phase-gradient SSIM (PG-SSIM).

    SSIM of the wrapped axial and lateral phase gradients inside the mask of target
    amplitudes above the given percentile.

    Returns:
        ``(mean, axial, lateral)``.
    """
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


def despeckle_nlm(image: np.ndarray, h: float = NLM_H, patch_size: int = 5,
                  patch_distance: int = 6) -> np.ndarray:
    """Non-local-means despeckling (``fast_mode=True``) applied before intensity metrics."""
    return denoise_nl_means(image, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)


def compute_bscan_metrics(pred_cx: np.ndarray, gt_cx: np.ndarray) -> Dict[str, float]:
    """The 7 metrics of the paper (plus the PG-SSIM components) for one complex B-scan.

    Both fields are converted to amplitude in dB, min-max normalized with the target's
    range and clipped to [0, 1]; the normalized images are despeckled with
    :func:`despeckle_nlm` before the intensity metrics. Phase metrics use the raw
    complex fields.

    Args:
        pred_cx: Reconstructed complex B-scan (Z, X).
        gt_cx: Clean complex B-scan (Z, X).
    """
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
