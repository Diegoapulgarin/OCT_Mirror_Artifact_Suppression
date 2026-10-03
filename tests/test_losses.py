import math

import pytest
import torch

from legacy_loader import load_legacy
from octmirror.losses import hinge_d_loss, hinge_g_loss, phase_circular_loss, phase_gradient_loss

GAN = "torchPix2Pix_AblationM2_M8.py"


def _complex_pair(seed=0, shape=(2, 16, 12)):
    g = torch.Generator().manual_seed(seed)
    amp = torch.rand(shape, generator=g, dtype=torch.float64) + 0.1
    phase = (torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    return amp, phase


def _to_channels(amp, phase):
    return torch.stack([amp * torch.cos(phase), amp * torch.sin(phase)], dim=1)


# (c)
@pytest.mark.parametrize("mode", ["geo", "none"])
def test_phase_circular_loss_zero_for_identical_inputs(mode):
    amp, phase = _complex_pair()
    x = _to_channels(amp, phase)
    assert phase_circular_loss(x, x.clone(), weight_mode=mode).item() == 0.0


@pytest.mark.parametrize("mode", ["geo", "none"])
def test_phase_circular_loss_invariant_to_two_pi(mode):
    amp, phase = _complex_pair(1)
    amp2, phase2 = _complex_pair(2)
    real = _to_channels(amp, phase)
    fake = _to_channels(amp2, phase2)
    fake_shifted = _to_channels(amp2, phase2 + 2 * math.pi)
    base = phase_circular_loss(fake, real, weight_mode=mode)
    assert base.item() > 0.1
    torch.testing.assert_close(phase_circular_loss(fake_shifted, real, weight_mode=mode), base)
    # same phase, shifted by 2*pi, gives (numerically) zero loss
    assert phase_circular_loss(_to_channels(amp, phase + 2 * math.pi), real, weight_mode=mode).item() < 1e-12


def test_unsupported_weight_modes_raise():
    x = _to_channels(*_complex_pair())
    for mode in ["real", "fake", "other"]:
        with pytest.raises(ValueError):
            phase_circular_loss(x, x, weight_mode=mode)
        with pytest.raises(ValueError):
            phase_gradient_loss(x, x, weight_mode=mode)


# golden
@pytest.mark.parametrize("source", [GAN, "complex_field_utils.py"])
@pytest.mark.parametrize("mode", ["geo", "none"])
def test_phase_circular_loss_matches_legacy(source, mode):
    legacy = load_legacy(source, ["phase_circular_loss"])
    torch.manual_seed(0)
    fake, real = torch.rand(2, 2, 32, 32), torch.rand(2, 2, 32, 32)
    assert torch.equal(phase_circular_loss(fake, real, mode), legacy.phase_circular_loss(fake, real, mode))


@pytest.mark.parametrize("mode", ["geo", "none"])
def test_phase_gradient_loss_matches_legacy(mode):
    legacy = load_legacy(GAN, ["phase_gradient_loss"])
    torch.manual_seed(0)
    fake, real = torch.rand(2, 2, 32, 32), torch.rand(2, 2, 32, 32)
    assert torch.equal(phase_gradient_loss(fake, real, mode), legacy.phase_gradient_loss(fake, real, mode))


def test_hinge_losses_match_legacy():
    legacy = load_legacy(GAN, ["hinge_d_loss", "hinge_g_loss"])
    torch.manual_seed(0)
    a, b = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    assert torch.equal(hinge_d_loss(a, b), legacy.hinge_d_loss(a, b))
    assert torch.equal(hinge_g_loss(b), legacy.hinge_g_loss(b))
