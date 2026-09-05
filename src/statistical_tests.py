"""Numerically reproducible paired tests for the current model comparison."""
from collections import namedtuple

import numpy as np
from scipy.stats import wilcoxon

WILCOXON_DECIMALS = 12
_Result = namedtuple("PairedWilcoxonResult", "statistic pvalue")


def paired_wilcoxon(differences):
    """Two-sided Pratt test; round only differences before assigning ranks.

    CSV serialization and equivalent averages can perturb mathematical ties.
    Twelve decimal places in the metric's reporting units removes machine
    roundoff, far below the data resolution. Means/effect sizes are unrounded.
    SciPy's documented ``method='auto'`` is retained (record its version when
    freezing outputs); this does not resolve cross-validation dependence.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError("paired differences must be a nonempty finite vector")
    d = np.round(d, decimals=WILCOXON_DECIMALS)
    if not np.any(d):
        return _Result(0.0, 1.0)
    result = wilcoxon(d, zero_method="pratt", alternative="two-sided", method="auto")
    return _Result(float(result.statistic), float(result.pvalue))
