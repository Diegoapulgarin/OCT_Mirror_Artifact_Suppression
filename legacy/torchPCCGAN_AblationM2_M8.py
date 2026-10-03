#%%
print("Starting pix2pix robust training script...")
# import sys
# sys.path.append('/Users/diegoalejandropulgarinsuarez/Documents/GitHub/DLOCT/CxDLOCT')
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os, time, pickle, json 
import csv
from glob import glob 
from typing import List, Tuple, Dict
from statistics import mean 
from tqdm import tqdm 
import re
import copy
import torch 
import torch.nn as nn 
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import random
import traceback
from scipy.interpolate import interp1d
from numpy.fft import fft, ifft, fftshift, ifftshift
import torch.nn.functional as F
import math

def mirrorArtifact(array):
    '''
    Introduce mirror artifacts in the 3D input array along axis = 0.
    using numpy fft operations.
    '''
    fringes = ifftshift(ifft(ifftshift(array, axes=0), axis=0), axes=0)
    tomcc = fftshift(fft(fftshift(fringes.real,axes=0),axis=0),axes=0)
    return tomcc

def Correlation(slices, savename=None):
    slices = slices[:,:,0] + 1j*slices[:,:,1]
    
    correlationy = np.angle(slices[:,1:] * np.conjugate(slices[:,:-1]))
    correlationx = np.angle(slices[1:, :] * np.conjugate(slices[:-1, :]))
    # stdx = np.std(correlationx)
    # meanx = np.mean(correlationx)
    # stdy = np.std(correlationy)
    # meany = np.mean(correlationy)
    return correlationx, correlationy

# ===================== LOSS HELPERS ADVANCED ===================== #

def hinge_d_loss(pred_real, pred_fake):
    # pred_*: (B,1,H,W)
    return F.relu(1 - pred_real).mean() + F.relu(1 + pred_fake).mean()

def hinge_g_loss(pred_fake):
    return -pred_fake.mean()

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
    elif weight_mode == 'none':
        w = torch.ones_like(delta)
    else:
        w = torch.ones_like(delta)
    loss = (w * (1 - torch.cos(delta))).sum() / (w.sum() + 1e-8)
    return loss

def phase_gradient_loss(fake_img, real_img, weight_mode='geo'):
    """
    Diferenciable phase-gradient constraint (axial + lateral) in PyTorch.
    fake_img / real_img: (B, 2, H, W)  (channels: real, imag)
    """
    fr, fi = fake_img[:,0], fake_img[:,1]
    rr, ri = real_img[:,0], real_img[:,1]

    phase_fake = torch.atan2(fi, fr)
    phase_real = torch.atan2(ri, rr)

    # gradients in axial (z/H) and lateral (x/W)
    grad_ax_fake = phase_fake[:, 1:, :] - phase_fake[:, :-1, :]
    grad_ax_real = phase_real[:, 1:, :] - phase_real[:, :-1, :]
    grad_lat_fake = phase_fake[:, :, 1:] - phase_fake[:, :, :-1]
    grad_lat_real = phase_real[:, :, 1:] - phase_real[:, :, :-1]

    # wrapped gradient differences
    delta_ax = grad_ax_fake - grad_ax_real
    delta_ax = torch.remainder(delta_ax + math.pi, 2*math.pi) - math.pi
    delta_lat = grad_lat_fake - grad_lat_real
    delta_lat = torch.remainder(delta_lat + math.pi, 2*math.pi) - math.pi

    amp_f = torch.sqrt(fr**2 + fi**2 + 1e-8)
    amp_r = torch.sqrt(rr**2 + ri**2 + 1e-8)

    if weight_mode == 'geo':
        w = torch.sqrt(amp_f * amp_r + 1e-8)
    elif weight_mode == 'real':
        w = amp_r
    elif weight_mode == 'fake':
        w = amp_f
    elif weight_mode == 'none':
        w = torch.ones_like(phase_fake)
    else:
        w = torch.ones_like(phase_fake)

    w_ax = w[:, 1:, :]
    w_lat = w[:, :, 1:]

    loss_ax = (w_ax * (1 - torch.cos(delta_ax))).sum() / (w_ax.sum() + 1e-8)
    loss_lat = (w_lat * (1 - torch.cos(delta_lat))).sum() / (w_lat.sum() + 1e-8)
    return 0.5 * (loss_ax + loss_lat)

def update_ema(ema_model, model, decay=0.999):
    """Update exponential moving average of model parameters."""
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name in model_params:
            if name in ema_params:
                ema_params[name].data.mul_(decay).add_(model_params[name].data, alpha=1 - decay)

# ===================== END Helpers  ===================== #

def validate_fn(val_dl, G, device, use_ema=False, G_ema=None):
    """
    Compute validation metrics: amplitude MAE, phase circular MAE, L1.
    Returns dict with avg metrics.
    """
    model = G_ema if use_ema and G_ema is not None else G
    model.eval()
    
    total_amp_mae = 0.0
    total_phase_mae = 0.0
    total_l1 = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for input_img, real_img, smax, smin in val_dl:
            input_img = input_img.to(device)
            real_img = real_img.to(device)
            
            fake_img = model(input_img)
            
            # Amplitude MAE
            fr, fi = fake_img[:,0], fake_img[:,1]
            rr, ri = real_img[:,0], real_img[:,1]
            amp_fake = torch.sqrt(fr**2 + fi**2 + 1e-8)
            amp_real = torch.sqrt(rr**2 + ri**2 + 1e-8)
            amp_mae = (amp_fake - amp_real).abs().mean()
            
            # Phase circular MAE (wrapped difference)
            phase_fake = torch.atan2(fi, fr)
            phase_real = torch.atan2(ri, rr)
            delta = phase_fake - phase_real
            delta = torch.remainder(delta + math.pi, 2*math.pi) - math.pi
            phase_mae = delta.abs().mean()
            
            # L1
            l1 = (fake_img - real_img).abs().mean()
            
            total_amp_mae += amp_mae.item()
            total_phase_mae += phase_mae.item()
            total_l1 += l1.item()
            n_batches += 1
    
    return {
        'amp_mae': total_amp_mae / n_batches if n_batches > 0 else 0.0,
        'phase_mae': total_phase_mae / n_batches if n_batches > 0 else 0.0,
        'l1': total_l1 / n_batches if n_batches > 0 else 0.0
    }

def extract_shape_from_folder(folder_name):
    """
    Extrae el shape (tuple de ints) del nombre del folder.
    Ejemplo: '30-09-2021_21-56_64x256x256_25percent_4' -> (64, 256, 256)
    """
    match = re.search(r'(\d+)x(\d+)x(\d+)', folder_name)
    if match:
        return tuple(int(x) for x in match.groups())
    else:
        raise ValueError(f"No se encontró shape en el nombre del folder: {folder_name}")

def dbscale(darray):
    if len(np.shape(darray))==3:
        img = 10*np.log10(abs(darray[:,:,0]+1j*darray[:,:,1])**2)
    else:
        img = 10*np.log10(abs(darray[:,:])**2)
    return img

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
    slicesAmp = 10**(slicesAmp)
    slices[:, :, :, 0] = np.real(slicesAmp * np.exp(1j*slicesPhase))
    slices[:, :, :, 1] = np.imag(slicesAmp * np.exp(1j*slicesPhase))
    return slices

def show_img_sample(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    """
    Muestra las imágenes de entrada y salida esperada con sus canales real e imaginario.
    
    Args:
        input_tensor (torch.Tensor): Tensor de entrada con forma (2, 1024, 1024).
        output_tensor (torch.Tensor): Tensor de salida esperada con forma (2, 1024, 1024).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    ax = axes.ravel()

    # Canal real de la imagen de entrada
    ax[0].imshow(input_tensor[0].cpu().numpy(), cmap="gray")
    ax[0].set_title("Canal Real - Imagen de Entrada")
    ax[0].axis("off")
    
    # Canal imaginario de la imagen de entrada
    ax[1].imshow(input_tensor[1].cpu().numpy(), cmap="gray")
    ax[1].set_title("Canal Imaginario - Imagen de Entrada")
    ax[1].axis("off")
    
    # Canal real de la imagen de salida esperada
    ax[2].imshow(output_tensor[0].cpu().numpy(), cmap="gray")
    ax[2].set_title("Canal Real - Imagen de Salida Esperada")
    ax[2].axis("off")
    
    # Canal imaginario de la imagen de salida esperada
    ax[3].imshow(output_tensor[1].cpu().numpy(), cmap="gray")
    ax[3].set_title("Canal Imaginario - Imagen de Salida Esperada")
    ax[3].axis("off")
    
    plt.tight_layout()
  
class ArrayDataset(Dataset):
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]], max_min: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Dataset adaptado para manejar matrices numpy con 2 canales (real e imaginario),
        junto con los valores máximos y mínimos para revertir la normalización.
        
        Args:
            data (List[Tuple[np.ndarray, np.ndarray]]): Lista de tuplas (entrada, salida esperada).
            max_min (List[Tuple[np.ndarray, np.ndarray]]): Lista de tuplas (max, min) para cada imagen.
        """
        self.data = data
        self.max_min = max_min

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Devuelve los tensores de entrada, salida esperada y los valores max/min para el índice dado.
        
        Args:
            idx (int): Índice del dato a retornar.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]: Tensores para entrenamiento y validación,
            junto con los valores máximos y mínimos.
        """
        input_array, output_array = self.data[idx]
        max_val, min_val = self.max_min[idx]
        
        # Convertir a tensores con 2 canales (real e imaginario)
        input_tensor = torch.tensor(input_array.transpose(2, 0, 1), dtype=torch.float32)
        output_tensor = torch.tensor(output_array.transpose(2, 0, 1), dtype=torch.float32)
        
        return input_tensor, output_tensor, max_val, min_val

    def __len__(self) -> int:
        """
        Retorna la longitud del dataset.
        
        Returns:
            int: Número de elementos en el dataset.
        """
        return len(self.data)

class Generator(nn.Module):
    def __init__(self, in_ch=2, out_ch=2):
        super().__init__()
        self.enc1 = self.conv2Relu(in_ch, 32, 5)
        self.enc2 = self.conv2Relu(32, 64, pool_size=4)
        self.enc3 = self.conv2Relu(64, 128, pool_size=2)
        self.enc4 = self.conv2Relu(128, 256, pool_size=2)
        
        self.dec1 = self.deconv2Relu(256, 128, pool_size=2)
        self.dec2 = self.deconv2Relu(128+128, 64, pool_size=2)
        self.dec3 = self.deconv2Relu(64+64, 32, pool_size=4)
        self.dec4 = nn.Sequential(
            nn.Conv2d(32+32, out_ch, 5, padding=2),
            nn.ReLU()
        )
        
    def conv2Relu(self, in_c, out_c, kernel_size=3, pool_size=None):
        layer = []
        if pool_size:
            layer.append(nn.AvgPool2d(pool_size))
        layer.append(nn.Conv2d(in_c, out_c, kernel_size, padding=(kernel_size-1)//2))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        layer.append(nn.BatchNorm2d(out_c))
        layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)
    
    def deconv2Relu(self, in_c, out_c, kernel_size=3, stride=1, pool_size=None):
        layer = []
        if pool_size:
            layer.append(nn.UpsamplingNearest2d(scale_factor=pool_size))
        layer.append(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1))
        layer.append(nn.BatchNorm2d(out_c))
        layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # (b, 32, H, W)
        x2 = self.enc2(x1) # (b, 64, H/4, W/4)
        x3 = self.enc3(x2) # (b, 128, H/8, W/8)
        x4 = self.enc4(x3) # (b, 256, H/16, W/16)
        
        # Decoder con skip connections
        out = self.dec1(x4)  # (b, 128, H/8, W/8)
        out = self.dec2(torch.cat((out, x3), dim=1))  # (b, 64, H/4, W/4)
        out = self.dec3(torch.cat((out, x2), dim=1))  # (b, 32, H, W)
        out = self.dec4(torch.cat((out, x1), dim=1))  # (b, 2, H, W) (2 canales de salida) 
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Entrada: 4 canales (2 de la imagen real + 2 de la imagen falsa)
        # La primera capa del discriminador no usa Batch Normalization, según el paper de pix2pix.
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1, bias=False), # 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Capas intermedias con Batch Normalization y stride=2
        self.conv2 = self.conv_block(64, 128, kernel_size=4, stride=2, padding=1) # 64x64 -> 32x32
        self.conv3 = self.conv_block(128, 256, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        
        # Última capa antes de la salida, con stride=1 para mantener el tamaño
        self.conv4 = self.conv_block(256, 512, kernel_size=4, stride=1, padding=1) # 16x16 -> 16x16

        # Capa de salida: 1 canal de probabilidad.
        # Usa una convolución de 4x4 con stride=1 para producir un mapa de probabilidad.
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # 16x16 -> 15x15 (sin padding) o 16x16 (con padding)

    def conv_block(self, in_c, out_c, kernel_size, stride, padding):
        """Bloque de convolución con Batch Normalization y LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_real, x_fake):
        # Concatenar las imágenes reales y generadas a nivel de canales
        x = torch.cat((x_real, x_fake), dim=1) # (b, 4, 128, 128)
        
        # Pasar por las capas convolucionales
        out = self.conv1(x)  # (b, 64, 64, 64)
        out = self.conv2(out) # (b, 128, 32, 32)
        out = self.conv3(out) # (b, 256, 16, 16)
        out = self.conv4(out) # (b, 512, 16, 16)
        
        # Salida final (mapa de probabilidad)
        return self.final(out)

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
        self.c4 = disc_block(256, 512, stride=2)   # faltaba en tu versión torch
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

def train_fn(train_dl, G, D,
             criterion_l1,
             optimizer_g, optimizer_d,
             epoch=0, num_epochs=0,
             lambda_l1=100.0,
             use_hinge=True,
             use_phase=True,
             lambda_phase=10.0,
             lambda_amp=10.0,
             phase_weight_mode='geo',
             use_phase_grad=False,
             lambda_phase_grad=0.0):
    G.train()
    D.train()
    LAMBDA = 100.0
    total_loss_g, total_loss_d = [], []
    # accumulators for loss components
    sum_adv = 0.0
    sum_l1 = 0.0
    sum_phase = 0.0
    sum_amp = 0.0
    sum_phase_grad = 0.0
    n_batches = 0

    for i, (input_img, real_img, smax, smin) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
        input_img = input_img.to(device)
        real_img = real_img.to(device)
        
        # =================== GENERATOR =================== #
        fake_img = G(input_img)
        pred_fake_for_g = D(real= input_img, fake=fake_img) if D.forward.__code__.co_argcount==3 else D(fake_img, input_img)
        if not use_hinge:
            # BCE style (legacy) – mantener por compatibilidad si se quiere probar
            valid = torch.ones_like(pred_fake_for_g)
            loss_g_adv = F.binary_cross_entropy_with_logits(pred_fake_for_g, valid)
        else:
            loss_g_adv = hinge_g_loss(pred_fake_for_g)

        loss_g_l1 = criterion_l1(fake_img, real_img) * lambda_l1

        # Amplitude (intensity) loss: compute amplitude from complex (real, imag)
        # Inputs in [0,1] after Sigmoid; map back to [-1,1] before inverse scaling
        def compute_amp(tensor):
            # tensor: (B, 2, H, W) with channels (real, imag) normalized in [0,1]
            t = tensor * 2.0 - 1.0
            amp = torch.sqrt(t[:,0]**2 + t[:,1]**2 + 1e-8)
            return amp

        amp_fake = compute_amp(fake_img)
        amp_real = compute_amp(real_img)
        loss_g_amp = F.l1_loss(amp_fake, amp_real) * lambda_amp

        if use_phase:
            loss_g_phase = phase_circular_loss(fake_img, real_img, weight_mode=phase_weight_mode) * lambda_phase
        else:
            loss_g_phase = torch.zeros(1, device=fake_img.device)

        if use_phase_grad:
            loss_g_phase_grad = phase_gradient_loss(fake_img, real_img, weight_mode=phase_weight_mode) * lambda_phase_grad
        else:
            loss_g_phase_grad = torch.zeros(1, device=fake_img.device)

        # Total generator loss with amplitude term
        loss_g = loss_g_adv + loss_g_l1 + loss_g_phase + loss_g_amp + loss_g_phase_grad
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)  # Prevent gradient explosion
        optimizer_g.step()
        
        # =================== DISCRIMINATOR =================== #
        with torch.no_grad():
            fake_detached = fake_img.detach()
        pred_real = D(real= input_img, fake= real_img) if D.forward.__code__.co_argcount==3 else D(real_img, input_img)
        pred_fake = D(real= input_img, fake= fake_detached) if D.forward.__code__.co_argcount==3 else D(fake_detached, input_img)

        if not use_hinge:
            valid = torch.ones_like(pred_real)
            fake  = torch.zeros_like(pred_fake)
            loss_d_real = F.binary_cross_entropy_with_logits(pred_real, valid)
            loss_d_fake = F.binary_cross_entropy_with_logits(pred_fake, fake)
            loss_d = (loss_d_real + loss_d_fake)
        else:
            loss_d = hinge_d_loss(pred_real, pred_fake)

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)  # Prevent gradient explosion
        optimizer_d.step()

        total_loss_g.append(loss_g.item())
        # accumulate components
        adv_val = loss_g_adv.item() if isinstance(loss_g_adv, torch.Tensor) else float(loss_g_adv)
        l1_val = loss_g_l1.item() if isinstance(loss_g_l1, torch.Tensor) else float(loss_g_l1)
        phase_val = loss_g_phase.item() if isinstance(loss_g_phase, torch.Tensor) else float(loss_g_phase)
        amp_val = loss_g_amp.item() if isinstance(loss_g_amp, torch.Tensor) else float(loss_g_amp)
        phase_grad_val = loss_g_phase_grad.item() if isinstance(loss_g_phase_grad, torch.Tensor) else float(loss_g_phase_grad)
        sum_adv += adv_val
        sum_l1 += l1_val
        sum_phase += phase_val
        sum_amp += amp_val
        sum_phase_grad += phase_grad_val
        n_batches += 1
        total_loss_d.append(loss_d.item())

    # compute average components
    if n_batches > 0:
        avg_components = dict(
            adv = sum_adv / n_batches,
            l1  = sum_l1 / n_batches,
            phase= sum_phase / n_batches,
            amp = sum_amp / n_batches,
            phase_grad = sum_phase_grad / n_batches
        )
    else:
        avg_components = dict(adv=0.0, l1=0.0, phase=0.0, amp=0.0, phase_grad=0.0)

    return (mean(total_loss_g),
            mean(total_loss_d),
            fake_img.detach().cpu(),
            input_img.detach().cpu(),
            real_img.detach().cpu(),
            smax.detach().cpu(),
            smin.detach().cpu(),
            avg_components)

def saving_img(fake_img, input_img, real_img, smax, smin, e, savePath):
    if fake_img.shape[0] > 1:
        fake_img = fake_img[0:1]
        input_img = input_img[0:1]
        real_img = real_img[0:1]
        smax = smax[0:1]
        smin = smin[0:1]
    os.makedirs(os.path.join(savePath, "generated"), exist_ok=True)


    def recover_and_save(img_tensor, smax, smin, name):
        if img_tensor.ndim == 4:
            img_tensor = img_tensor.squeeze(0)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (z, x, 2)
        img_np = img_np[np.newaxis, ...]  # (1, z, x, 2)
        # Convertir smax y smin a numpy si son tensores
        if isinstance(smax, torch.Tensor):
            smax = smax.cpu().numpy()
        if isinstance(smin, torch.Tensor):
            smin = smin.cpu().numpy()
        if smax.ndim == 1:
            smax = smax[:, np.newaxis, np.newaxis]
            smin = smin[:, np.newaxis, np.newaxis]
        img_rec = inverseLogScale(img_np, smax, smin)
        img_rec = img_rec[0]
        intImage = 20 * np.log10(np.abs(img_rec[..., 0] + 1j * img_rec[..., 1]))
        cx, cy = Correlation(img_rec)
        return intImage, cx, cy
        
    intImageFake, cx, cy = recover_and_save(fake_img, smax, smin, "fake")
    intImageInput, cx_input, cy_input = recover_and_save(input_img, smax, smin, "input")
    intImageReal, cx_real, cy_real = recover_and_save(real_img, smax, smin, "real")
    fig,axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(intImageFake, cmap='gray')
    axs[0].set_title("Fake Image")
    axs[0].axis("off")
    axs[1].imshow(intImageInput, cmap='gray')
    axs[1].set_title("Input Image")
    axs[1].axis("off")
    axs[2].imshow(intImageReal, cmap='gray')
    axs[2].set_title("Real Image")
    axs[2].axis("off")
    fig.savefig(os.path.join(savePath, "generated", f"epoch_{e+1}.png"))
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cx, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[0].set_title("Correlation Fake Image")
    axs[0].axis("off")
    axs[1].imshow(cx_input, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Correlation Input Image")
    axs[1].axis("off")
    axs[2].imshow(cx_real, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[2].set_title("Correlation Real Image")
    axs[2].axis("off")
    fig.savefig(os.path.join(savePath, "generated", f"correlation_x_epoch_{e+1}.png"))
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cy, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[0].set_title("Correlation Fake Image")
    axs[0].axis("off")
    axs[1].imshow(cy_input, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Correlation Input Image")
    axs[1].axis("off")
    axs[2].imshow(cy_real, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[2].set_title("Correlation Real Image")
    axs[2].axis("off")
    fig.savefig(os.path.join(savePath, "generated", f"correlation_y_epoch_{e+1}.png"))
    plt.close(fig)

def saving_model(D, G, e,savePath):
    os.makedirs(os.path.join(savePath,"weight"), exist_ok=True)
    torch.save(G.state_dict(),os.path.join(savePath,f"weight/G{str(e+1)}.pth"))
    torch.save(D.state_dict(),os.path.join(savePath,f"weight/D{str(e+1)}.pth"))
        
def show_losses(g, d):
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    ax = axes.ravel()
    ax[0].plot(np.arange(len(g)).tolist(), g)
    ax[0].set_title("Generator Loss")
    ax[1].plot(np.arange(len(d)).tolist(), d)
    ax[1].set_title("Discriminator Loss")

def plot_training_curves(result, savePath):
    """Plot and save training/validation curves for all components."""
    epochs = np.arange(1, len(result['loss_g']) + 1)
    has_phase_grad = 'train_phase_grad' in result and len(result['train_phase_grad']) == len(epochs)

    if has_phase_grad:
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Generator and Discriminator total loss
    axes[0, 0].plot(epochs, result['loss_g'], label='Generator', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, result['loss_d'], label='Discriminator', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Train components
    axes[0, 2].plot(epochs, result['train_adv'], label='Adversarial', color='orange')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Train Adversarial Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(epochs, result['train_l1'], label='L1', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Train L1 Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, result['train_phase'], label='Phase', color='purple')
    axes[1, 1].plot(epochs, result['val_phase_mae'], label='Val Phase MAE', color='purple', linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss / MAE')
    axes[1, 1].set_title('Phase Loss (Train) & Val MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(epochs, result['train_amp'], label='Amp (train)', color='brown')
    axes[1, 2].plot(epochs, result['val_amp_mae'], label='Val Amp MAE', color='brown', linestyle='--')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss / MAE')
    axes[1, 2].set_title('Amplitude Loss (Train) & Val MAE')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    if has_phase_grad:
        axes[0, 3].plot(epochs, result['train_phase_grad'], label='PhaseGrad', color='black')
        axes[0, 3].set_xlabel('Epoch')
        axes[0, 3].set_ylabel('Loss')
        axes[0, 3].set_title('Train Phase Gradient Loss')
        axes[0, 3].legend()
        axes[0, 3].grid(True)

        axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savePath, 'training_curves.png'), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {os.path.join(savePath, 'training_curves.png')}")

def train_loop(train_dl,
               val_dl,
               G,
               D,
               num_epoch,
               savePath,
               device,
               lr=5e-5,
               dlr=1e-5,
               betas=(0.5,0.999),
               lambda_l1=100.0,
               use_phase=True,
               lambda_phase=10.0,
               lambda_amp=10.0,
               phase_weight_mode='geo',
               use_phase_grad=False,
               lambda_phase_grad=0.0,
               use_hinge=True,
               use_ema=True,
               ema_decay=0.999,
               checkpoint_every=5,
               scheduler_patience=10,
               image_log_every=25):

    G.to(device)
    D.to(device)
    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=dlr, betas=betas)
    criterion_bce = nn.BCEWithLogitsLoss() # Binary Cross Entropy with logits
    criterion_l1 = nn.L1Loss() # MAE
    
    # Learning rate scheduler (reduces LR when amp_mae plateaus)
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=scheduler_patience, 
        verbose=True, min_lr=1e-7
    )
    
    # EMA model
    G_ema = None
    if use_ema:
        G_ema = copy.deepcopy(G).to(device)
        G_ema.eval()
        for param in G_ema.parameters():
            param.requires_grad = False
    
    # Ensure weight directory exists
    os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
    
    total_loss_d, total_loss_g = [], []
    val_amp_maes, val_phase_maes, val_l1s = [], [], []
    train_advs, train_l1s, train_phases, train_amps, train_phase_grads = [], [], [], [], []
    result = {}

    csv_path = os.path.join(savePath, 'metrics_epoch.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'loss_g', 'loss_d',
                'train_adv', 'train_l1', 'train_phase', 'train_amp', 'train_phase_grad',
                'val_amp_mae', 'val_phase_mae', 'val_l1'
            ])
    
    best_amp_mae = float('inf')
    best_epoch = -1
    
    for e in range(num_epoch):
        loss_g, loss_d, fake_img, input_img, real_img, smax, smin, comps = train_fn(
            train_dl, G, D,
            criterion_l1,
            optimizer_g, optimizer_d,
            epoch=e, num_epochs=num_epoch,
            lambda_l1=lambda_l1,
            use_hinge=use_hinge,
            use_phase=use_phase,
            lambda_phase=lambda_phase,
            lambda_amp=lambda_amp,
            phase_weight_mode=phase_weight_mode,
            use_phase_grad=use_phase_grad,
            lambda_phase_grad=lambda_phase_grad
        )
        total_loss_d.append(loss_d)
        total_loss_g.append(loss_g)
        train_advs.append(comps['adv'])
        train_l1s.append(comps['l1'])
        train_phases.append(comps['phase'])
        train_amps.append(comps['amp'])
        train_phase_grads.append(comps['phase_grad'])
        
        # Update EMA
        if use_ema and G_ema is not None:
            update_ema(G_ema, G, decay=ema_decay)
        
        # Validation
        val_metrics = validate_fn(val_dl, G, device, use_ema=use_ema, G_ema=G_ema)
        val_amp_maes.append(val_metrics['amp_mae'])
        val_phase_maes.append(val_metrics['phase_mae'])
        val_l1s.append(val_metrics['l1'])

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                e + 1,
                loss_g,
                loss_d,
                comps['adv'],
                comps['l1'],
                comps['phase'],
                comps['amp'],
                comps['phase_grad'],
                val_metrics['amp_mae'],
                val_metrics['phase_mae'],
                val_metrics['l1']
            ])
        
        # Scheduler step (based on validation amp MAE)
        scheduler_g.step(val_metrics['amp_mae'])
        
        # Save images sparsely to reduce I/O pressure
        if (e + 1) % image_log_every == 0 or e == 0 or e == num_epoch - 1:
            try:
                saving_img(fake_img, input_img, real_img, smax, smin, e, savePath)
            except Exception as ex:
                print(f"Warning: saving_img failed at epoch {e+1}: {ex}")
        
        # Periodic checkpoint (every N epochs)
        if (e + 1) % checkpoint_every == 0:
            saving_model(D, G, e, savePath)
            if use_ema and G_ema is not None:
                os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
                torch.save(G_ema.state_dict(), os.path.join(savePath, f"weight/G_ema{str(e+1)}.pth"))
        
        # Save best model based on validation amp MAE
        if val_metrics['amp_mae'] < best_amp_mae:
            best_amp_mae = val_metrics['amp_mae']
            best_epoch = e + 1
            os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
            torch.save(G.state_dict(), os.path.join(savePath, "weight/G_best.pth"))
            torch.save(D.state_dict(), os.path.join(savePath, "weight/D_best.pth"))
            if use_ema and G_ema is not None:
                torch.save(G_ema.state_dict(), os.path.join(savePath, "weight/G_ema_best.pth"))
            print(f"  --> New best amp_mae: {best_amp_mae:.6f} at epoch {best_epoch}")
        
        # Save result dict
        result["loss_d"] = total_loss_d
        result["loss_g"] = total_loss_g
        result["val_amp_mae"] = val_amp_maes
        result["val_phase_mae"] = val_phase_maes
        result["val_l1"] = val_l1s
        result["train_adv"] = train_advs
        result["train_l1"] = train_l1s
        result["train_phase"] = train_phases
        result["train_amp"] = train_amps
        result["train_phase_grad"] = train_phase_grads
        
        print(f"Epoch {e+1}/{num_epoch} "
              f"- G: {loss_g:.4f} (adv={comps['adv']:.4f}, l1={comps['l1']:.4f}, phase={comps['phase']:.4f}, amp={comps['amp']:.4f}, phase_grad={comps['phase_grad']:.4f}) "
              f"- D: {loss_d:.4f} | Val: amp_mae={val_metrics['amp_mae']:.6f}, phase_mae={val_metrics['phase_mae']:.4f}, l1={val_metrics['l1']:.4f}")
    
    # Final save
    saving_model(D, G, num_epoch-1, savePath)
    if use_ema and G_ema is not None:
        os.makedirs(os.path.join(savePath, "weight"), exist_ok=True)
        torch.save(G_ema.state_dict(), os.path.join(savePath, f"weight/G_ema_final.pth"))
    
    try:
        # Plot training curves
        plot_training_curves(result, savePath)
    except Exception as ex:
        print(f"Error plotting curves: {ex}")
    finally:
        print(f"Model Saved Successfully. Best amp_mae: {best_amp_mae:.6f} at epoch {best_epoch}")
        return G, D, best_amp_mae, best_epoch

print('starting data preparation...')
dataPath = '/home/rsg-dapulgaris/Data'
folders = ['phase']
filestrain = [
            'OpticNerveAOld.npy',
            'OpticNerveBOld.npy',
            'unpairCadaverhearth.npy',
            'ChickenBreastA.npy',
            'ChickenBreastB.npy',
            'NailA.npy',
            'NailB.npy',
            'S.Eye2A.npy',
            'S.Eye2B.npy'
            ]
tomList = []
tomccList = []
maxData = []
minData = []
maxDataCx = []
minDataCx = []
nfile = 0
for folder in tqdm(folders):
    files = os.listdir(os.path.join(dataPath, folder))
    for file in files:
        if file not in filestrain:
            continue
        tom = np.load(os.path.join(dataPath, folder, file))
        if len(tom.shape)==4:
            tom = tom[...,0] + 1j*tom[...,1] # y,x,2,z
        tomcc = mirrorArtifact(tom)
        tom = np.transpose(tom, (2, 0, 1)) # y,z,x
        tomcc = np.transpose(tomcc, (2, 0, 1)) # y,z,x
        logslicesData, slicesmaxData, slicesminData,_,_ = logScale(tom)
        # logslicesData = np.stack((amp,phase/np.pi),axis=3)
        logslicesCxData, slicesmaxCxData, slicesminCxData,_,_ = logScale(tomcc)
        # logslicesCxData = np.stack((ampcc,phasecc/np.pi),axis=3)
        tomList.append(logslicesData)
        tomccList.append(logslicesCxData)
        maxData.append(slicesmaxData)
        minData.append(slicesminData)
        maxDataCx.append(slicesmaxCxData)
        minDataCx.append(slicesminCxData)
        nfile += 1
print(f'number of files loaded: {nfile}')
tomTarget = np.concatenate(tomList, axis=0)
tomccInput = np.concatenate(tomccList, axis=0)
maxTarget = np.concatenate(maxData, axis=0)
minTarget = np.concatenate(minData, axis=0)
maxInput = np.concatenate(maxDataCx, axis=0)
minInput = np.concatenate(minDataCx, axis=0)
del tomList, tomccList, maxData, minData, maxDataCx, minDataCx 
print(f'len of dataset total: {tomTarget.shape[0]}')
trainDataset = []
for i in range(tomTarget.shape[0]):
    trainDataset.append((tomccInput[i], tomTarget[i]))
print(f"len of dataset training: {len(trainDataset)}")
print(f"len of input: {trainDataset[0][0].shape}")
print(f"len of output: {trainDataset[0][1].shape}")
del tomTarget, tomccInput
max_min = []
num_images = maxTarget.shape[0] 
for i in range(num_images):
    max_target = maxTarget[i, :, :]
    min_target = minTarget[i, :, :]
    max_min.append((max_target, min_target))
del maxTarget, minTarget
print(f"len of max_min: {len(max_min)}")  # must match with len(train)

# Split into train and validation (90% train, 10% val)
total_samples = len(trainDataset)
val_size = int(0.1 * total_samples)
train_size = total_samples - val_size

indices = list(range(total_samples))
random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_data = [trainDataset[i] for i in train_indices]
val_data = [trainDataset[i] for i in val_indices]
train_max_min = [max_min[i] for i in train_indices]
val_max_min = [max_min[i] for i in val_indices]

print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

train_ds = ArrayDataset(train_data, train_max_min)
val_ds = ArrayDataset(val_data, val_max_min)

input_tensor, output_tensor,_,_ = train_ds[0]
print(f"Input tensor shape: {input_tensor.shape}")
print(f"Output tensor shape: {output_tensor.shape}")

BATCH_SIZE = 32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'is on: {device}')

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f'Train batches: {len(train_dl)}, Val batches: {len(val_dl)} with batch size of {BATCH_SIZE}')
in_channels = train_ds[0][0].shape[0]  
out_channels = train_ds[0][1].shape[0]
print(f'in channels: {in_channels}, out channels: {out_channels}')

CONFIGS = [
    dict(name="M2_ComplexCGAN_NoPhase",     lambda_phase=0.0,   phase_weight_mode="none", use_phase_grad=False, lambda_phase_grad=0.0),
    dict(name="M3_PCCGAN_Reference",        lambda_phase=20.0,  phase_weight_mode="geo",  use_phase_grad=False, lambda_phase_grad=0.0),
    dict(name="M4_NoAmplitudeWeighting",    lambda_phase=20.0,  phase_weight_mode="none", use_phase_grad=False, lambda_phase_grad=0.0),
    dict(name="M5_PhaseWeight_Low",         lambda_phase=2.0,   phase_weight_mode="geo",  use_phase_grad=False, lambda_phase_grad=0.0),
    dict(name="M6_PhaseWeight_MidHigh",     lambda_phase=60.0,  phase_weight_mode="geo",  use_phase_grad=False, lambda_phase_grad=0.0),
    dict(name="M7_PhaseWeight_High",        lambda_phase=150.0, phase_weight_mode="geo",  use_phase_grad=False, lambda_phase_grad=0.0),
    dict(name="M8_PhaseGradientConstraint", lambda_phase=20.0,  phase_weight_mode="geo",  use_phase_grad=True,  lambda_phase_grad=20.0),
]

ablation_root = '/home/rsg-dapulgaris/Models/AblationPCCGAN'
os.makedirs(ablation_root, exist_ok=True)
ablation_summary_path = os.path.join(ablation_root, 'ablation_summary.json')
ablation_summary = []

def _write_ablation_summary(path, summary_obj):
    with open(path, 'w') as f:
        json.dump(summary_obj, f, indent=2)

EPOCH = 300

for cfg in CONFIGS:
    try:
        nseed = 0
        random.seed(nseed)
        np.random.seed(nseed)
        torch.manual_seed(nseed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(nseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(nseed)
            torch.cuda.manual_seed_all(nseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        G = GeneratorTf(in_ch=in_channels, out_ch=out_channels)
        D = DiscriminatorTf(in_ch=in_channels)

        savePath = os.path.join('/home/rsg-dapulgaris/Models/AblationPCCGAN', cfg['name'])
        os.makedirs(savePath, exist_ok=True)

        trained_G, trained_D, best_amp_mae, best_epoch = train_loop(
            train_dl, val_dl, G, D, EPOCH,
            savePath=savePath,
            device=device,
            lr=3e-5,
            dlr=5e-6,
            lambda_l1=100.0,
            use_phase=True,
            lambda_phase=cfg['lambda_phase'],
            lambda_amp=5.0,
            phase_weight_mode=cfg['phase_weight_mode'],
            use_phase_grad=cfg['use_phase_grad'],
            lambda_phase_grad=cfg['lambda_phase_grad'],
            use_hinge=True,
            use_ema=True,
            ema_decay=0.999,
            checkpoint_every=5,
            scheduler_patience=15,
            image_log_every=25
        )

        ablation_summary.append({
            'name': cfg['name'],
            'best_amp_mae': best_amp_mae,
            'best_epoch': best_epoch,
            'best_weight_path': os.path.join(savePath, 'weight', 'G_best.pth')
        })

    except Exception as ex:
        ablation_summary.append({
            'name': cfg['name'],
            'best_amp_mae': None,
            'best_epoch': None,
            'best_weight_path': os.path.join('/home/rsg-dapulgaris/Models/AblationPCCGAN', cfg['name'], 'weight', 'G_best.pth'),
            'error': str(ex),
            'traceback': traceback.format_exc(limit=10)
        })
    finally:
        _write_ablation_summary(ablation_summary_path, ablation_summary)
