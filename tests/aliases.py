"""Original (legacy) names mapped to the refactored functions. Used only by golden tests."""
from octmirror.preprocessing import inverse_log_scale, log_scale, mirror_artifact

mirrorArtifact = mirror_artifact
logScale = log_scale


def inverseLogScale(oldslices, slicesMax, slicesMin):
    """PC-CGAN variant (no clipping)."""
    return inverse_log_scale(oldslices, slicesMax, slicesMin, max_log_amp=None)


def inverseLogScaleClipped(oldslices, slicesMax, slicesMin):
    """Diffusion variant (complex_field_utils.py, clip at 30.0)."""
    return inverse_log_scale(oldslices, slicesMax, slicesMin, max_log_amp=30.0)
