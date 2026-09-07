"""Controls for the nested latent-value test.

A null result is only informative if the procedure can detect a real effect at
this sample size, so the positive control is part of the evidence, not decoration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_latent_incremental_value import _loso_predictions, _r2, nested_test

N_SUBJECTS = 28
TARGETS = ("alpha", "beta")


def _pairs(rng: np.random.Generator, informative: bool) -> pd.DataFrame:
    """Synthetic probe/context predictions with or without extra signal."""
    rows = []
    for dim in (3, 8):
        for target in TARGETS:
            latent = rng.normal(size=N_SUBJECTS)
            context = rng.normal(size=N_SUBJECTS)
            # ``true`` always depends on context; only the informative case also
            # depends on the part of the probe that context cannot explain.
            true = 2.0 * context + rng.normal(scale=0.3, size=N_SUBJECTS)
            if informative:
                true = true + 1.5 * latent
            rows.extend({
                "latent_dim": dim, "outer_fold": i % 4, "subject": f"s{i:02d}",
                "target": target, "probe_prediction": latent[i],
                "context_prediction": context[i], "true": true[i],
            } for i in range(N_SUBJECTS))
    return pd.DataFrame(rows)


def test_detects_a_genuine_incremental_signal():
    """Positive control: an informative probe must be flagged as adding value."""
    pairs = _pairs(np.random.default_rng(11), informative=True)
    summary, _ = nested_test(pairs, (3, 8))
    assert len(summary) == 4
    assert (summary.delta_r2 > 0).all(), summary.delta_r2.tolist()
    assert summary.significant_fdr_0_05.all(), summary.wilcoxon_p_fdr_bh.tolist()


def test_reports_null_when_the_probe_adds_nothing():
    """Negative control: an uninformative probe must not be flagged."""
    pairs = _pairs(np.random.default_rng(12), informative=False)
    summary, _ = nested_test(pairs, (3, 8))
    assert not summary.significant_fdr_0_05.any(), summary.wilcoxon_p_fdr_bh.tolist()


def test_loso_predictions_never_use_the_held_out_row():
    """Leave-one-out must exclude the scored participant from its own fit."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 1))
    y = (3.0 * x[:, 0] + rng.normal(scale=0.1, size=20))
    contaminated = y.copy()
    contaminated[7] += 1000.0
    baseline = _loso_predictions(y, x)
    perturbed = _loso_predictions(contaminated, x)
    # Perturbing one observation must not change that observation's own
    # prediction, because the fit that produced it excluded the row.
    assert perturbed[7] == pytest.approx(baseline[7], abs=1e-9)
    assert not np.allclose(perturbed, baseline)


def test_r2_matches_the_explicit_definition():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _r2(y, y) == pytest.approx(1.0)
    assert _r2(y, np.full(4, y.mean())) == pytest.approx(0.0)
