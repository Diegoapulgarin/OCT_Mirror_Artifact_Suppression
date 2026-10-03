import pytest

from octmirror.baselines import hilbert_analytic_signal
from octmirror.inference import evaluate_diffusion


@pytest.mark.parametrize("fn", [hilbert_analytic_signal, evaluate_diffusion])
def test_unreleased_methods_are_stubs(fn):
    with pytest.raises(NotImplementedError):
        fn()
