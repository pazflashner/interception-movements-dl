"""Release-facing checks for the confirmatory dashboard assets and checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.confirmatory_dashboard import ASSETS, checkpoint_path, decode, load_model


def test_dashboard_manifest_matches_frozen_protocol() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["protocol_version"] == "strategy-confirmatory-v1"
    assert manifest["window_mode"] == "go_to_arrival"
    assert manifest["n_trials"] == 4732
    assert manifest["n_subjects"] == 28
    assert manifest["latent_dimensions"] == [2, 3, 4, 8]
    assert manifest["validation_dimensions"] == [3, 8]
    assert len(manifest["all_validation_subjects"]) == 28
    assert manifest["condition_controls_validated"] is False
    assert manifest["evaluation_version"] == "post-review-training-reference-v1"


def test_dashboard_validation_assets_cover_all_held_out_participants() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    expected = set(manifest["all_validation_subjects"])
    empirical = pd.read_csv(ASSETS / "empirical_query_features.csv")
    generated = pd.read_csv(ASSETS / "generated_validation.csv")

    assert set(empirical.subject) == expected
    for latent_dim in manifest["validation_dimensions"]:
        frame = generated[generated.latent_dim == latent_dim]
        assert set(frame.subject) == expected
        assert set(frame.outer_fold) == {0, 1, 2, 3}
        assert frame.groupby("subject").size().eq(10).all()


def test_live_reference_checkpoints_decode_finite_2d_outputs() -> None:
    for latent_dim in (2, 3, 4, 8):
        assert checkpoint_path(latent_dim).exists()
        model, norm = load_model(latent_dim)
        trajectory, timing = decode(
            model,
            norm,
            np.zeros(latent_dim, dtype=np.float32),
            sp=2,
            side=1,
            target_speed=0.635,
        )

        assert trajectory.shape == (1, 100, 2)
        assert timing.shape == (1, 2)
        assert np.isfinite(trajectory).all()
        assert np.isfinite(timing).all()
        assert (timing >= 0).all()


def test_dashboard_reports_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "output" / "pdf"
    assert (report_dir / "Interception_Movements_Final_Scientific_Report.pdf").exists()
    assert (report_dir / "Interception_Movements_Results_Guide.pdf").exists()


def test_dashboard_exposes_negative_probes_and_fair_timing_comparisons():
    probe = pd.read_csv(ASSETS / "behavioral_probe_summary.csv")
    selected = probe[(probe.model_family == "cvae") & (probe.latent_dim == 3) & (probe.fingerprint == "mean")]
    assert len(selected) == 14
    assert (selected.r2_oof_mean < 0).sum() == 10
    timing = pd.read_csv(ASSETS / "timing_fairness_paired.csv")
    assert len(timing) == 8
    assert timing.n_participants.eq(28).all()
