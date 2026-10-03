"""Preprocessing of complex OCT tomograms.

Mirror-artifact simulation, per-slice log-amplitude normalization and its inverse,
central ROI extraction, tomogram loading and B-scan subsampling.

Axis convention: tomograms are stored as (Z, X, Y) = (depth, fast-scan, slow-scan).
A B-scan is a (Z, X) slice. Normalized slices are real arrays whose last axis holds
the (real, imag) channels.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift


def mirror_artifact(array: np.ndarray) -> np.ndarray:
    """Introduce mirror artifacts in the input array along axis 0 (Z).

    The fringes are recovered with an inverse FFT along depth, only their real part
    is kept, and the result is transformed back to the depth domain.

    Args:
        array: Complex array whose axis 0 is depth, e.g. a tomogram (Z, X, Y) or a
            B-scan / ROI (Z, X).

    Returns:
        Complex array of the same shape containing the complex-conjugate artifact.
    """
    fringes = ifftshift(ifft(ifftshift(array, axes=0), axis=0), axes=0)
    tomcc = fftshift(fft(fftshift(fringes.real, axes=0), axis=0), axes=0)
    return tomcc


def log_scale(
    slices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-slice log-amplitude normalization of complex slices.

    The log10 amplitude of every slice is min-max normalized to [0, 1] and recombined
    with the original phase; real and imaginary parts are then mapped from [-1, 1] to
    [0, 1].

    Args:
        slices: Complex array (N, H, W), or real array (N, H, W, 2) with (real, imag)
            in the last axis.

    Returns:
        Tuple ``(logslices, slices_max, slices_min, amp, phase)`` where ``logslices``
        has shape (N, H, W, 2), ``slices_max`` / ``slices_min`` have shape (N, 1, 1)
        and hold the per-slice log10-amplitude limits, and ``amp`` / ``phase`` are the
        normalized log amplitude and the phase, both (N, H, W).
    """
    logslices = np.copy(slices)
    nSlices = slices.shape[0]
    if len(slices.shape) == 4:
        logslicesAmp = abs(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
        logslicesPhase = np.angle(logslices[:, :, :, 0] + 1j*logslices[:, :, :, 1])
    else:
        logslicesAmp = abs(logslices)
        logslicesPhase = np.angle(logslices)
    # rescale amplitude
    logslicesAmp = np.log10(logslicesAmp)
    slicesMax = np.reshape(logslicesAmp.max(axis=(1, 2)), (nSlices, 1, 1))
    slicesMin = np.reshape(logslicesAmp.min(axis=(1, 2)), (nSlices, 1, 1))
    logslicesAmp = (logslicesAmp - slicesMin) / (slicesMax - slicesMin)
    # redefine the real and imaginary components with the new amplitude and same phase
    logslicesReal = (np.real(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    logslicesImag = (np.imag(logslicesAmp * np.exp(1j*logslicesPhase)) + 1)/2
    logslices = np.stack((logslicesReal, logslicesImag), axis=-1)
    return logslices, slicesMax, slicesMin, logslicesAmp, logslicesPhase


def inverse_log_scale(
    slices: np.ndarray,
    smax: np.ndarray,
    smin: np.ndarray,
    max_log_amp: Optional[float] = None,
) -> np.ndarray:
    """Invert :func:`log_scale` given the per-slice log10-amplitude limits.

    Args:
        slices: Normalized slices (N, H, W, 2) with (real, imag) in [0, 1].
        smax: Per-slice maximum log10 amplitude, broadcastable to (N, H, W).
        smin: Per-slice minimum log10 amplitude, broadcastable to (N, H, W).
        max_log_amp: If not None, the de-normalized log10 amplitude is clipped to this
            upper bound before exponentiation. The diffusion pipeline uses 30.0 (keeps
            ``10**x`` far below float64 overflow while staying far above any physically
            expected OCT amplitude); the PC-CGAN pipeline uses None (no clipping).

    Returns:
        Array (N, H, W, 2) with the real and imaginary parts of the reconstructed field.
    """
    slices = np.copy(slices)
    slices = (slices * 2) - 1
    slicesAmp = abs(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesPhase = np.angle(slices[:, :, :, 0] + 1j*slices[:, :, :, 1])
    slicesAmp = slicesAmp * (smax - smin) + smin
    if max_log_amp is not None:
        slicesAmp = np.clip(slicesAmp, a_min=None, a_max=max_log_amp)
    slicesAmp = 10**(slicesAmp)
    slices[:, :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices


def extract_center_roi(bscan_complex: np.ndarray, roi_size: int = 128) -> np.ndarray:
    """Extract a native, centered ROI from a single B-scan (no interpolation/resize).

    Args:
        bscan_complex: Complex array with shape (Z, X), e.g. (512, 512).
        roi_size: Side of the square ROI to extract.

    Returns:
        Complex ROI with shape (roi_size, roi_size), a direct centered crop.

    Raises:
        ValueError: If the B-scan is smaller than the ROI.
    """
    z, x = bscan_complex.shape
    if z < roi_size or x < roi_size:
        raise ValueError(f"B-scan {bscan_complex.shape} smaller than ROI {roi_size}")
    z0 = (z - roi_size) // 2
    x0 = (x - roi_size) // 2
    return bscan_complex[z0:z0+roi_size, x0:x0+roi_size]


def load_tomogram(filepath: str, verbose: bool = True) -> np.ndarray:
    """Load a tomogram stored with the fixed axis convention (Z, X, Y).

    A 3D array is returned unchanged (already complex). A 4D array (Z, X, Y, 2) is
    interpreted as real/imaginary channels and combined into a complex (Z, X, Y) array.

    Args:
        filepath: Path to a ``.npy`` file.
        verbose: Print the original and resulting shapes.

    Returns:
        Complex tomogram (Z, X, Y).
    """
    raw = np.load(filepath)
    original_shape = raw.shape
    if raw.ndim == 4:
        tom = raw[..., 0] + 1j * raw[..., 1]
    else:
        tom = raw
    if verbose:
        print(f"  Loaded {os.path.basename(filepath)}: original shape {original_shape} "
              f"-> complex shape {tom.shape} (Z, X, Y)")
    return tom


def select_bscan_indices(num_bscans: int, n_bscans: int = 40) -> np.ndarray:
    """Select up to ``n_bscans`` evenly spaced B-scan indices along the slow axis.

    Args:
        num_bscans: Number of B-scans in the volume (size of the Y axis).
        n_bscans: Maximum number of B-scans to keep.

    Returns:
        All indices if ``num_bscans <= n_bscans``; otherwise ``n_bscans`` indices from
        ``np.linspace(0, num_bscans - 1, n_bscans)`` truncated to int.
    """
    if num_bscans <= n_bscans:
        return np.arange(num_bscans)
    idx = np.linspace(0, num_bscans - 1, n_bscans, dtype=int)
    return np.unique(idx)
