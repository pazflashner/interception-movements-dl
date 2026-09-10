"""Numerically reproducible paired tests for the current model comparison."""
from collections import namedtuple

import numpy as np
from scipy.stats import wilcoxon

WILCOXON_DECIMALS = 12
_Result = namedtuple("PairedWilcoxonResult", "statistic pvalue")


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a nonempty finite vector in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


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
