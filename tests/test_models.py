import warnings

import pytest
import torch

from legacy_loader import load_legacy
from octmirror.models.diffusion import UNET, Attention, DDPMScheduler
from octmirror.models.pccgan import PatchGANDiscriminator, UNetGenerator

DIFF = "diffComplexField_ROI256_D1_D4.py"
GAN = "torchPix2Pix_AblationM2_M8.py"


# (b) shapes
def test_unet_generator_shape():
    G = UNetGenerator(in_ch=2, out_ch=2).eval()
    with torch.no_grad():
        y = G(torch.rand(1, 2, 512, 512))
    assert y.shape == (1, 2, 512, 512)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_patchgan_shape():
    D = PatchGANDiscriminator(in_ch=2).eval()
    with torch.no_grad():
        y = D(torch.rand(1, 2, 512, 512), torch.rand(1, 2, 512, 512))
    assert y.shape == (1, 1, 30, 30)


def test_diffusion_unet_shape():
    model = UNET().eval()
    assert model.shallow_conv.in_channels == 4
    assert model.output_conv.out_channels == 2
    x = torch.randn(1, 2, 256, 256)
    cond = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        y = model(x, torch.tensor([10]), cond)
    assert y.shape == (1, 2, 256, 256)


# golden tests: identical architectures and outputs
@pytest.mark.parametrize("source", [GAN, "evaluate_ablation_unified.py"])
def test_pccgan_models_match_legacy(source):
    legacy = load_legacy(source, ["GeneratorTf", "DiscriminatorTf"])
    torch.manual_seed(0)
    G_old = legacy.GeneratorTf(in_ch=2, out_ch=2)
    torch.manual_seed(0)
    G_new = UNetGenerator(in_ch=2, out_ch=2)
    assert G_old.state_dict().keys() == G_new.state_dict().keys()
    for k, v in G_old.state_dict().items():
        assert torch.equal(v, G_new.state_dict()[k])
    G_new.load_state_dict(G_old.state_dict())
    x = torch.rand(2, 2, 256, 256)
    with torch.no_grad():
        assert torch.equal(G_old.eval()(x), G_new.eval()(x))

    D_old = legacy.DiscriminatorTf(in_ch=2)
    D_new = PatchGANDiscriminator(in_ch=2)
    D_new.load_state_dict(D_old.state_dict())
    a, b = torch.rand(2, 2, 128, 128), torch.rand(2, 2, 128, 128)
    with torch.no_grad():
        assert torch.equal(D_old.eval()(a, b), D_new.eval()(a, b))


def test_diffusion_unet_matches_legacy():
    legacy = load_legacy(DIFF, ["UNET", "UnetLayer", "ResBlock", "Attention", "SinusoidalEmbeddings"])
    torch.manual_seed(0)
    old = legacy.UNET()
    torch.manual_seed(0)
    new = UNET()
    assert old.state_dict().keys() == new.state_dict().keys()
    for k, v in old.state_dict().items():
        assert torch.equal(v, new.state_dict()[k])
    x, cond = torch.randn(2, 2, 64, 64), torch.randn(2, 2, 64, 64)
    t = torch.tensor([3, 700])
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert torch.equal(old.eval()(x, t, cond), new.eval()(x, t, cond))


def test_ddpm_scheduler_matches_legacy():
    legacy = load_legacy(DIFF, ["DDPM_Scheduler"])
    old = legacy.DDPM_Scheduler(num_time_steps=1000, device="cpu")
    new = DDPMScheduler(num_time_steps=1000, device="cpu")
    assert torch.equal(old.beta, new.beta) and torch.equal(old.alpha, new.alpha)


# sdpa_kernel replacement: same enabled backends and identical output
def _backend_flags():
    b = torch.backends.cuda
    return (b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.math_sdp_enabled(),
            b.cudnn_sdp_enabled())


def test_sdpa_kernel_enables_same_backends_as_legacy_sdp_kernel():
    from octmirror.models.diffusion import _SDPA_BACKENDS
    from torch.nn.attention import sdpa_kernel

    defaults = _backend_flags()
    try:
        # start from a state where everything is disabled so that the contexts must enable
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True,
                                                enable_mem_efficient=True):
                legacy_flags = _backend_flags()
        with sdpa_kernel(_SDPA_BACKENDS):
            new_flags = _backend_flags()
    finally:
        torch.backends.cuda.enable_flash_sdp(defaults[0])
        torch.backends.cuda.enable_mem_efficient_sdp(defaults[1])
        torch.backends.cuda.enable_math_sdp(defaults[2])
        torch.backends.cuda.enable_cudnn_sdp(defaults[3])
    assert legacy_flags == new_flags


@pytest.mark.parametrize("training", [False, True])
def test_attention_output_identical_to_legacy(training):
    legacy = load_legacy(DIFF, ["Attention"])
    old = legacy.Attention(64, num_heads=2, dropout_prob=0.1)
    new = Attention(64, num_heads=2, dropout_prob=0.1)
    new.load_state_dict(old.state_dict())
    old.train(training)
    new.train(training)
    x = torch.randn(2, 64, 16, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        torch.manual_seed(1)
        y_old = old(x)
    torch.manual_seed(1)
    y_new = new(x)
    assert torch.equal(y_old, y_new)
