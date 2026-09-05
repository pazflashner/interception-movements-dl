import numpy as np
import pytest
from scipy.stats import wilcoxon

from src.statistical_tests import paired_wilcoxon


def test_roundoff_does_not_split_tied_ranks():
    d = np.tile([0.1, -0.1, 0.2, 0.3], 7)
    perturbed = d + np.linspace(-2e-16, 2e-16, len(d))
    assert paired_wilcoxon(d) == paired_wilcoxon(perturbed)
    expected = wilcoxon(d, zero_method="pratt", method="auto")
    assert paired_wilcoxon(d).pvalue == expected.pvalue


def test_zero_differences_and_invalid_inputs():
    assert paired_wilcoxon(np.zeros(28)).pvalue == 1.0
    with pytest.raises(ValueError):
        paired_wilcoxon([np.nan, 1])
