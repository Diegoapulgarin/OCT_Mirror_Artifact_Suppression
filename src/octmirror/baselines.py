"""Analytical baselines.

TODO: the Hilbert-transform analytic-signal baseline used in the paper is pending the
authors' code and is not implemented in this release.
"""
from __future__ import annotations

import numpy as np


def hilbert_analytic_signal(*args, **kwargs) -> np.ndarray:
    """Hilbert-transform analytic-signal baseline (not implemented in this release)."""
    raise NotImplementedError  # TODO: pending the authors' code
