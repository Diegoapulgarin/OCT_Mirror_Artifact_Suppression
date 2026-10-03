import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import random
import math
import time
import csv
import datetime 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3 #pip install timm
from tqdm import tqdm #pip install tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt #pip install matplotlib
import torch.optim as optim
import numpy as np
import os
import sys
import json
from numpy.fft import fft, ifft
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, Dataset
from numpy.fft import fftshift, ifftshift, fft, ifft

# logScale/inverseLogScale/mirrorArtifact/phase_circular_loss are reused as-is from the
# PC-CGAN complex-field script (M3 reference, lambda_phase=20.0) instead of reimplementing them.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from complex_field_utils import logScale, inverseLogScale, mirrorArtifact, phase_circular_loss

# ---------------------------------------------------------------------------
# Exploratory ROI experiment (D1-D4): native 256x256 ROIs, 2 channels (real, imag),
# no resize/interpolation of any kind. Tomograms follow OCT convention (Z, X, Y);
# the optional 4th axis is exclusively real/imaginary. Scaled up from the validated
# ROI128 script; D1-D4 are independent runs that never share weights.
# ---------------------------------------------------------------------------

ROI_SIZE = 256
ROI_MODE = "center"
N_BSCANS_PER_VOLUME = 40  # subsample per volume for a fast exploratory run
NUM_EPOCHS = 80
SEED = 0
ALLOW_OVERWRITE = False
ROI_ROOT = '/home/rsg-dapulgaris/Models/diffCxModels/ROI256_Ablation'


def extract_center_roi(bscan_complex, roi_size=128):
    """
    Extract a native, centered ROI from a single B-scan (no interpolation/resize).

    Args:
        bscan_complex (np.ndarray): complex array with shape (Z, X), e.g. (512, 512).
        roi_size (int): side of the square ROI to extract.

    Returns:
        np.ndarray: complex ROI with shape (roi_size, roi_size), a direct crop of the
            original array (native resolution, centered, no interpolation).
    """
    z, x = bscan_complex.shape
    if z < roi_size or x < roi_size:
        raise ValueError(f"B-scan {bscan_complex.shape} smaller than ROI {roi_size}")
    z0 = (z - roi_size) // 2
    x0 = (x - roi_size) // 2
    return bscan_complex[z0:z0+roi_size, x0:x0+roi_size]


def diagnose_gpu_and_model(batch_size=1, img_size=512, num_channels=1):
    """
    Diagnostic function to test GPU setup and memory usage before training.
    
    Args:
        batch_size: batch size for dummy forward pass
        img_size: spatial resolution (H=W)
        num_channels: number of input channels
    """
    print("\n" + "="*60)
    print("GPU DIAGNOSTIC TEST")
    print("="*60)
    
    # Calculate theoretical memory requirements
    n_pixels = img_size * img_size
    attention_mem_per_image = (n_pixels * n_pixels * 4) / (1024**3)  # float32 = 4 bytes
    print(f"\n📊 THEORETICAL MEMORY CALCULATION:")
    print(f"  Image size: {img_size}×{img_size} = {n_pixels:,} pixels")
    print(f"  Attention matrix per image: {n_pixels:,} × {n_pixels:,}")
    print(f"  Memory for attention per image: ~{attention_mem_per_image:.2f} GiB")
    print(f"  Total batch: {batch_size} images × {attention_mem_per_image:.2f} GiB = {batch_size * attention_mem_per_image:.2f} GiB")
    
    # Check CUDA and GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    n_gpu = torch.cuda.device_count()
    print(f"GPUs detected: {n_gpu}")
    
    if n_gpu == 0:
        print("WARNING: No GPUs detected!")
        return
    
    for i in range(n_gpu):
        try:
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GiB")
        except Exception as e:
            print(f"  GPU {i}: Unable to get info - {e}")
    
    print(f"\nBuilding UNET model for test...")
    model = UNET()
    
    # Wrap with DataParallel if multiple GPUs
    if n_gpu > 1:
        print(f"\nWrapping model with DataParallel for {n_gpu} GPUs")
        model = nn.DataParallel(model)
        print(f"  device_ids: {model.device_ids}")
        print(f"  output_device: {model.output_device}")
        
        # Explain how DataParallel splits work
        samples_per_gpu = batch_size // n_gpu
        remainder = batch_size % n_gpu
        print(f"\n⚠️  IMPORTANTE - Cómo DataParallel divide el trabajo:")
        print(f"  • Batch total: {batch_size} imágenes")
        print(f"  • GPU 0 procesará: {samples_per_gpu + remainder} imágenes de {img_size}×{img_size} COMPLETAS")
        print(f"  • GPU 1 procesará: {samples_per_gpu} imágenes de {img_size}×{img_size} COMPLETAS")
        print(f"  • Cada GPU necesita: ~{samples_per_gpu * attention_mem_per_image:.2f} GiB solo para atención")
        print(f"  • DataParallel NO divide las imágenes individuales entre GPUs")
        print(f"  • Si una imagen necesita 64GB, CADA GPU necesita 64GB por imagen")
    else:
        print("\nUsing single GPU (no DataParallel)")
    
    model = model.cuda()
    model.eval()
    
    # Clear cache before test
    torch.cuda.empty_cache()
    
    print(f"\nMemory BEFORE forward pass:")
    for i in range(n_gpu):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB")
    
    # Create dummy batch
    print(f"\nRunning dummy forward pass with batch_size={batch_size}, size={img_size}x{img_size}, channels={num_channels}...")
    try:
        dummy_x = torch.randn(batch_size, num_channels, img_size, img_size).cuda()
        dummy_cond = torch.randn(batch_size, num_channels, img_size, img_size).cuda()
        t = torch.tensor([0] * batch_size, device=dummy_x.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output = model(dummy_x, t, dummy_cond)
        
        print(f"Forward pass SUCCESS! Output shape: {output.shape}")
        
        print(f"\nMemory AFTER forward pass:")
        for i in range(n_gpu):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB")
        
        # Check if memory is distributed
        if n_gpu > 1:
            gpu0_alloc = torch.cuda.memory_allocated(0) / 1024**3
            gpu1_alloc = torch.cuda.memory_allocated(1) / 1024**3
            if gpu1_alloc < 1.0:
                print(f"\n⚠️  WARNING: GPU 1 has very low memory usage ({gpu1_alloc:.2f} GiB)")
                print("    DataParallel may not be distributing work properly.")
            else:
                print(f"\n✓ Memory appears distributed across GPUs")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ OUT OF MEMORY during forward pass!")
            print(f"Error: {e}")
            print(f"\n💡 SOLUCIONES RECOMENDADAS (en orden de prioridad):")
            print(f"\n1. 🚀 USAR FLASH ATTENTION (MEJOR SOLUCIÓN):")
            print(f"   - Reduce memoria de O(N²) a O(N) donde N={img_size}×{img_size}")
            print(f"   - PyTorch 2.0+ tiene soporte nativo")
            print(f"   - Cambiar F.scaled_dot_product_attention por versión optimizada")
            print(f"   - Con esto podrías usar batch_size=4 e imágenes de 512×512")
            print(f"\n2. GRADIENT CHECKPOINTING:")
            print(f"   - Recomputa activaciones en backward (ahorra ~50% memoria)")
            print(f"   - Agrega torch.utils.checkpoint.checkpoint() en capas costosas")
            print(f"\n3. Reducir batch_size:")
            print(f"   - Actual: {batch_size} → Probar: {max(1, batch_size//2)}")
            print(f"   - Con batch=1: cada GPU procesa 1 imagen = {attention_mem_per_image:.2f} GiB")
            print(f"\n4. ÚLTIMO RECURSO (no recomendado por tu experiencia):")
            print(f"   - Reducir trainSize o deshabilitar atención")
        else:
            print(f"\n❌ ERROR during forward pass: {e}")
        raise
    finally:
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    print("="*60)
    print("DIAGNOSTIC TEST COMPLETE")
    print("="*60 + "\n")


def diagnose_batch_forward(batch_size, img_size, num_channels=2):
    """
    Extra forward-only sanity check (no backward) at the actual training batch size,
    run right after the batch=1 diagnostic and before touching real data/training.
    Raises a clear exception if it fails, so an invalid job is never queued.
    """
    print(f"\nRunning batch-size forward check: batch_size={batch_size}, img_size={img_size}, channels={num_channels}...")
    model = UNET()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda().eval()
    torch.cuda.empty_cache()
    try:
        dummy_x = torch.randn(batch_size, num_channels, img_size, img_size).cuda()
        dummy_cond = torch.randn(batch_size, num_channels, img_size, img_size).cuda()
        t = torch.tensor([0] * batch_size, device=dummy_x.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output = model(dummy_x, t, dummy_cond)
        print(f"Batch-size forward check SUCCESS! Output shape: {output.shape}")
    except Exception as e:
        raise RuntimeError(
            f"Forward pass failed for batch_size={batch_size}, ROI_SIZE={img_size}: {e}"
        ) from e
    finally:
        del model
        torch.cuda.empty_cache()


class ArrayDataset(Dataset):
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Dataset to handle 2-channel (real, imag) complex-field numpy arrays.

        Args:
            data (List[Tuple[np.ndarray, np.ndarray]]): List of tuples (input, expected output),
                each array with shape (H, W, 2).
        """
        self.data = data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns input and expected output tensors for the given index.

        Args:
            idx (int): Index of the data to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors for training and validation, shape (2, H, W).
        """
        input_array, output_array = self.data[idx]
        # Channels (real, imag) already exist in the last axis; move to (2, H, W)
        input_tensor = torch.tensor(input_array.transpose(2, 0, 1), dtype=torch.float32)
        output_tensor = torch.tensor(output_array.transpose(2, 0, 1), dtype=torch.float32)
        return input_tensor, output_tensor

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of elements in the dataset.
        """
        return len(self.data)

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.register_buffer('embeddings', embeddings)  # Register as buffer to avoid being a model parameter

    def forward(self, x, t):
        # check the device of x and move embeddings if necessary
        embeds = self.embeddings.to(x.device)[t]
        return embeds[:, :, None, None]
    
# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x
    
class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        # contiguous() required: fused kernels need stride-1 last dim, rearrange+indexing breaks it
        q, k, v = x[0].contiguous(), x[1].contiguous(), x[2].contiguous()
        
        # Usar Flash Attention para reducir memoria de O(N²) a O(N)
        # Esto permite procesar imágenes de 512×512 con ~1GB en lugar de ~64GB
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,           # Habilitar Flash Attention (requiere compute capability >= 7.5)
            enable_math=True,            # Respaldo seguro: si Flash/mem-efficient no aplican (dtype/forma), evita abortar
            enable_mem_efficient=True    # Habilitar implementación memory-efficient como fallback
        ):
            x = F.scaled_dot_product_attention(
                q, k, v, 
                is_causal=False, 
                dropout_p=self.dropout_prob if self.training else 0.0
            )
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')
    
class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, False, False, True, True, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 2,
            input_channels: int = 4,
            output_channels: int = 2,
            time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t,condition):
        x = torch.cat((x, condition), dim=1)
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))
    
class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False).to(self.device)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).to(self.device)

    def forward(self, t):
        return self.beta[t], self.alpha[t]

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(train_ds,
          savePath,
          batch_size: int = 1,
          num_time_steps: int = 1000,
          num_epochs: int = 15,
          seed: int = -1,
          ema_decay: float = 0.9999,
          lr=2e-5,
          checkpoint_path: str = None,
          interpolate:bool=False,
          trainSize: int = 512,
          use_phase_loss: bool = False,
          lambda_phase_diff: float = 0.0,
          experiment_id: int = 0,
          roi_size: int = 256):

    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET()

    # Use DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    # mixed precision scaler (automatic)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    metrics_path = os.path.join(savePath, 'training_metrics.csv')
    csv_header = ['epoch', 'loss_total', 'mse_noise', 'phase_loss', 'seconds_per_epoch',
                  'samples_per_second', 'peak_gpu_memory_gb', 'experiment_id', 'roi_size',
                  'batch_size', 'lr']
    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    n_samples = len(train_ds)
    checked_batch_shape = False
    for i in range(num_epochs):
        total_loss = 0
        total_mse_noise = 0
        total_phase_loss = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()
        with tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}") as pbar:
            for condition, x in pbar:
                if not checked_batch_shape:
                    expected_shape = (x.shape[0], 2, roi_size, roi_size)
                    if tuple(condition.shape) != expected_shape or tuple(x.shape) != expected_shape:
                        raise ValueError(
                            f"Expected input/target batches of shape {expected_shape}, "
                            f"got condition={tuple(condition.shape)}, target={tuple(x.shape)}"
                        )
                    checked_batch_shape = True
                x = x.cuda()
                condition = condition.cuda()
                if interpolate:
                    condition = F.interpolate(condition, size=(trainSize, trainSize), mode='bilinear', align_corners=False)
                    x = F.interpolate(x, size=(trainSize, trainSize), mode='bilinear', align_corners=False)
                current_batch_size = x.shape[0]
                t = torch.randint(0, num_time_steps, (current_batch_size,))
                e = torch.randn_like(x, requires_grad=False)
                a = scheduler.alpha[t].view(current_batch_size, 1, 1, 1).to(x.device)
                x0_clean = x.clone()
                x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

                # Mixed precision forward/backward to reduce memory
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    output = model(x_noisy, t, condition)
                    mse_noise = criterion(output, e)
                    if use_phase_loss:
                        x0_pred = (x_noisy - torch.sqrt(1 - a) * output) / torch.sqrt(a)
                        phase_loss = phase_circular_loss(x0_pred, x0_clean, weight_mode='geo')
                        loss = mse_noise + lambda_phase_diff * phase_loss
                    else:
                        phase_loss = torch.zeros((), device=mse_noise.device)
                        loss = mse_noise
                total_loss += loss.item()
                total_mse_noise += mse_noise.item()
                total_phase_loss += phase_loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
        epoch_seconds = time.time() - epoch_start
        samples_per_second = n_samples / epoch_seconds if epoch_seconds > 0 else 0.0
        peak_gpu_memory_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
        avg_loss = total_loss / len(train_loader)
        avg_mse_noise = total_mse_noise / len(train_loader)
        avg_phase_loss = total_phase_loss / len(train_loader)
        print(f'Epoch {i+1} | Loss {avg_loss:.5f} '
              f'| mse_noise {avg_mse_noise:.5f} '
              f'| phase_loss {avg_phase_loss:.5f} '
              f'| sec/epoch {epoch_seconds:.2f} '
              f'| samples/sec {samples_per_second:.2f} '
              f'| peak_gpu_mem_gb {peak_gpu_memory_gb:.2f}')
        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([i+1, avg_loss, avg_mse_noise, avg_phase_loss, epoch_seconds,
                                     samples_per_second, peak_gpu_memory_gb, experiment_id, roi_size,
                                     batch_size, lr])


    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }

    torch.save(checkpoint, os.path.join(savePath, 'checkpoint.pth'))
    print(f"Model saved to {savePath}")

def display_reverse(images: List[torch.Tensor],
                     savePath):
    """
    Displays and saves the intermediate images.

    Args:
        images (List[torch.Tensor]): List of tensors with the images to process.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i]
        if isinstance(x, torch.Tensor):  # Check if it is a tensor
            if x.ndim == 4:  # If the image has shape (1,1, H, W)
                x = x.squeeze(0)  # Remove batch dimension
            x = x.numpy()  # Convert to numpy if it is a tensor
            x = x[0]
        ax.imshow(x, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    if savePath is not None:
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, 'reverse.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def inference(savePath: str,
              num_time_steps: int = 1000,
              ema_decay: float = 0.9999,
              input_img: torch.Tensor = None,
              train_size: int = 256,
              interpolate:bool=False):
    """
    Performs inference and displays the intermediate denormalized images.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        num_time_steps (int): Number of time steps for inference.
        ema_decay (float): EMA decay.
        input_img (torch.Tensor): Input image with dimensions (C, H, W).
        train_size (int): Size to which images are interpolated during training.
    """
    checkpoint_path = os.path.join(savePath, 'checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = UNET()
    
    # Handle DataParallel saved weights
    state_dict = checkpoint['weights']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix from DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model = model.cuda()
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []
    print('interpolate:',interpolate)
    print('train size:',train_size)
    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(1, 2, train_size, train_size).cuda()  # intial noise (real, imag)

        # Always ensure condition matches train_size to avoid shape mismatch
        condition = input_img.cuda()
        if condition.shape[-2:] != (train_size, train_size):
            condition = F.interpolate(condition, size=(train_size, train_size), mode='bilinear', align_corners=False)
        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.tensor([t], device=z.device)
            temp = (scheduler.beta[t_tensor] / ((torch.sqrt(1 - scheduler.alpha[t_tensor])) * (torch.sqrt(1 - scheduler.beta[t_tensor]))))
            
            # z, t_tensor and condition to the model
            z = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor]))) * z - (temp * model(z, t_tensor, condition))
            if t in times:
                images.append(z.clone().cpu())
            e = torch.randn_like(z)
            z = z + (e * torch.sqrt(scheduler.beta[t_tensor]))

        # last step for inference
        t_tensor = torch.tensor([0], device=z.device)
        temp = scheduler.beta[t_tensor] / ((torch.sqrt(1 - scheduler.alpha[t_tensor])) * (torch.sqrt(1 - scheduler.beta[t_tensor])))
        x = (1 / (torch.sqrt(1 - scheduler.beta[t_tensor]))) * z - (temp * model(z, t_tensor, condition))
        images.append(x.clone().cpu())
        display_reverse(images, savePath=savePath)


# Experiment configurations for the ROI-256 exploratory ablation. Only one is executed
# per run (selected via EXPERIMENT_ID); D1 is the default. D2-D4 are defined for later
# runs but never trained in the same execution, and none share weights.
CONFIGS = {
    1: dict(name="D1_ROI256_NoPhase", use_phase_loss=False, lambda_phase_diff=0.0, batch_size=8, lr=2e-5),
    2: dict(name="D2_ROI256_PhaseLoss", use_phase_loss=True, lambda_phase_diff=20.0, batch_size=8, lr=2e-5),
    3: dict(name="D3_ROI256_PhaseLoss_LRScaled", use_phase_loss=True, lambda_phase_diff=20.0, batch_size=16, lr=4e-5),
    4: dict(name="D4_ROI256_PhaseLoss_LRHigh", use_phase_loss=True, lambda_phase_diff=20.0, batch_size=16, lr=8e-5),
}

EXPERIMENT_ID = 4

if __name__ == '__main__':
    print('start')

    if EXPERIMENT_ID not in CONFIGS:
        raise ValueError("EXPERIMENT_ID must be one of 1, 2, 3, 4")
    cfg = CONFIGS[EXPERIMENT_ID]

    os.makedirs(ROI_ROOT, exist_ok=True)
    savePath = os.path.join(ROI_ROOT, cfg['name'])
    existing_checkpoint = os.path.join(savePath, 'checkpoint.pth')
    existing_metrics = os.path.join(savePath, 'training_metrics.csv')
    if not ALLOW_OVERWRITE and (os.path.exists(existing_checkpoint) or os.path.exists(existing_metrics)):
        raise RuntimeError(
            f"savePath '{savePath}' already contains a checkpoint.pth or training_metrics.csv "
            f"and ALLOW_OVERWRITE is False. Refusing to overwrite a previous experiment."
        )
    os.makedirs(savePath, exist_ok=True)

    run_metadata_path = os.path.join(savePath, 'run_metadata.json')
    run_start_time = time.time()
    with open(run_metadata_path, 'w') as f:
        json.dump({'experiment_id': EXPERIMENT_ID, 'name': cfg['name'], 'status': 'running',
                    'start_time': run_start_time}, f, indent=2)

    try:
        print(f"\nActive configuration: {cfg['name']} "
              f"| batch_size={cfg['batch_size']} | lr={cfg['lr']} "
              f"| lambda_phase_diff={cfg['lambda_phase_diff']} | use_phase_loss={cfg['use_phase_loss']} "
              f"| num_epochs={NUM_EPOCHS} | seed={SEED}")

        # Ejecutar prueba de diagnóstico antes de cargar datos. Debe fallar ruidosamente
        # si algo va mal, para no enviar un job inválido a la cola.
        print("\n" + "="*70)
        print("🔬 DIAGNÓSTICO: Testing GPU setup y Flash Attention (ROI_SIZE)")
        print("="*70)
        try:
            diagnose_gpu_and_model(batch_size=1, img_size=ROI_SIZE, num_channels=2)
            diagnose_batch_forward(batch_size=cfg['batch_size'], img_size=ROI_SIZE, num_channels=2)
        except Exception as e:
            print(f"GPU diagnostic FAILED: {e}")
            raise

        dataPath = '/home/rsg-dapulgaris/Data'
        folders = ['noPhase', 'phase', 'synthetic']
        filestrain = [
                'OpticNerveAOld.npy',
                'OpticNerveBOld.npy',
                'unpairCadaverhearth.npy',
                'ChickenBreastA.npy',
                'ChickenBreastB.npy',
                'NailA.npy',
                'NailB.npy',
                'S.Eye2A.npy',
                'S.Eye2B.npy',
                ]

        n_volumes = 0
        n_rois = 0
        trainDataset = []
        for folder in tqdm(folders):
            files = os.listdir(os.path.join(dataPath, folder))
            for file in files:
                tom = np.load(os.path.join(dataPath, folder, file))
                if len(tom.shape) == 4:
                    tom = tom[..., 0] + 1j * tom[..., 1]  # (Z, X, Y)
                n_volumes += 1
                n_y = tom.shape[2]
                if n_y > N_BSCANS_PER_VOLUME:
                    y_indices = np.linspace(0, n_y - 1, N_BSCANS_PER_VOLUME).astype(int)
                else:
                    y_indices = np.arange(n_y)
                for y_idx in y_indices:
                    bscan_complex = tom[:, :, y_idx]  # (Z, X), native resolution
                    target_roi = extract_center_roi(bscan_complex, roi_size=ROI_SIZE)
                    # Mirror artifact applied directly on the (256,256) complex ROI, axis 0 == Z
                    input_roi = mirrorArtifact(target_roi)
                    target_norm, _, _, _, _ = logScale(target_roi[np.newaxis, ...])
                    input_norm, _, _, _, _ = logScale(input_roi[np.newaxis, ...])
                    trainDataset.append((input_norm[0], target_norm[0]))
                    n_rois += 1

        print(f'Total volumes loaded: {n_volumes}')
        print(f'Total B-scans/ROIs: {n_rois}')
        train_ds = ArrayDataset(trainDataset)
        del trainDataset
        input_tensor, output_tensor = train_ds[0]
        print(f"Input tensor shape: {tuple(input_tensor.shape)}")
        print(f"Target tensor shape: {tuple(output_tensor.shape)}")
        print("native ROI, no interpolation")

        # Display GPU information
        print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
        print(f'GPUs available: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            try:
                print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
            except Exception:
                print(f'GPU {i}: name unavailable')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Primary device: {device}')

        set_seed(SEED)

        num_time_steps = 1000
        config = {
            'experiment_id': EXPERIMENT_ID,
            'name': cfg['name'],
            'use_phase_loss': cfg['use_phase_loss'],
            'lambda_phase_diff': cfg['lambda_phase_diff'],
            'batch_size': cfg['batch_size'],
            'lr': cfg['lr'],
            'num_epochs': NUM_EPOCHS,
            'num_time_steps': num_time_steps,
            'seed': SEED,
            'roi_size': ROI_SIZE,
            'roi_mode': ROI_MODE,
            'n_bscans_per_volume': N_BSCANS_PER_VOLUME,
            'data_path': dataPath,
            'folders': folders,
            'save_path': savePath,
            'native_roi_no_interpolation': True,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(savePath, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        train(train_ds,
              savePath=savePath,
              batch_size=cfg['batch_size'],
              num_time_steps=num_time_steps,
              num_epochs=NUM_EPOCHS,
              seed=SEED,
              lr=cfg['lr'],
              use_phase_loss=cfg['use_phase_loss'],
              lambda_phase_diff=cfg['lambda_phase_diff'],
              trainSize=ROI_SIZE,
              interpolate=False,
              experiment_id=EXPERIMENT_ID,
              roi_size=ROI_SIZE)

        # Qualitative inference sample, saved with savefig (no plt.show()).
        sample_input, _ = train_ds[0]
        inference(savePath=savePath,
                  num_time_steps=num_time_steps,
                  input_img=sample_input.unsqueeze(0),
                  train_size=ROI_SIZE,
                  interpolate=False)

        run_metadata = {
            'experiment_id': EXPERIMENT_ID,
            'name': cfg['name'],
            'status': 'ok',
            'start_time': run_start_time,
            'end_time': time.time(),
        }
        with open(run_metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)

    except Exception as e:
        run_metadata = {
            'experiment_id': EXPERIMENT_ID,
            'name': cfg['name'],
            'status': 'failed',
            'error': str(e),
            'start_time': run_start_time,
            'end_time': time.time(),
        }
        with open(run_metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)
        raise

    print('done!')
