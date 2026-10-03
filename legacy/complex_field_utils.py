import numpy as np
import torch
import math
from numpy.fft import fft, ifft, fftshift, ifftshift

def mirrorArtifact(array):
    '''
    Introduce mirror artifacts in the 3D input array along axis = 0.
    using numpy fft operations.
    '''
    fringes = ifftshift(ifft(ifftshift(array, axes=0), axis=0), axes=0)
    tomcc = fftshift(fft(fftshift(fringes.real,axes=0),axis=0),axes=0)
    return tomcc

def phase_circular_loss(fake_img, real_img, weight_mode='geo'):
    """
    fake_img / real_img: (B, 2, H, W)  (channels: real, imag)
    L = weighted_mean (1 - cos(Δφ))  with Δφ wrapped to [-π,π]
    """
    fr, fi = fake_img[:,0], fake_img[:,1]
    rr, ri = real_img[:,0], real_img[:,1]
    phase_fake = torch.atan2(fi, fr)
    phase_real = torch.atan2(ri, rr)
    delta = phase_fake - phase_real
    # wrap to [-pi, pi]
    delta = torch.remainder(delta + math.pi, 2*math.pi) - math.pi
    amp_f = torch.sqrt(fr**2 + fi**2 + 1e-8)
    amp_r = torch.sqrt(rr**2 + ri**2 + 1e-8)
    if weight_mode == 'geo':
        w = torch.sqrt(amp_f * amp_r + 1e-8)
    elif weight_mode == 'real':
        w = amp_r
    elif weight_mode == 'fake':
        w = amp_f
    else:
        w = torch.ones_like(delta)
    loss = (w * (1 - torch.cos(delta))).sum() / (w.sum() + 1e-8)
    return loss

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
    slicesMax = np.reshape(logslicesAmp.max(axis=(1, 2)), ( nSlices,1, 1))
    slicesMin = np.reshape(logslicesAmp.min(axis=(1, 2)), ( nSlices,1, 1))
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
    # 30.0 keeps 10**x far below float64 overflow while staying far above any physically
    # expected OCT reconstructed amplitude magnitude, so normal reconstructions are unaffected.
    slicesAmp = np.clip(slicesAmp, a_min=None, a_max=30.0)
    slicesAmp = 10**(slicesAmp)
    slices[:, :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices
