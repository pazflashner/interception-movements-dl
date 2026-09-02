"""Create a self-contained, raw-data-free dashboard bundle for sharing."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


SHARE = config.STUDY_ROOT / "output" / "share"
BUNDLE = SHARE / (
    "Interception_Strategy_Dashboard_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)


def copy(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_text_files() -> None:
    (BUNDLE / "launch_dashboard.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "python -m streamlit run src\\strategy_dashboard.py\r\n"
        "pause\r\n",
        encoding="ascii",
    )
    (BUNDLE / "setup_and_launch.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "python -m pip install -r requirements_dashboard.txt\r\n"
        "if errorlevel 1 pause & exit /b 1\r\n"
        "call launch_dashboard.bat\r\n",
        encoding="ascii",
    )
    (BUNDLE / "README.txt").write_text(
        """INTERCEPTION MOVEMENT STRATEGY DASHBOARD

QUICK START (WINDOWS)
1. Extract the ZIP.
2. Double-click setup_and_launch.bat on the first run.
3. Later, double-click launch_dashboard.bat.

CONTENTS
- Both temporal representations: movement onset to arrival and target motion onset to arrival.
- Conditional VAE widths n=2, 3, 4, and 8, each trained with seeds 42, 43, and 44.
- Latent controls, task conditions, generated trajectories, timing, velocity, and minimum-jerk decomposition.
- Held-out participant metrics, representative distribution checks, and latent association heatmaps.

SCIENTIFIC BOUNDARIES
- Condition 2 only; 2-D x-y table plane; 100 phase samples.
- Physical movement and initiation time are withheld from the encoder and predicted separately.
- n=3 is the smallest stable strategy-inclusive model; n=8 is a capacity comparator.
- Latent coordinates can rotate, reflect, or swap between seeds.
- Minimum-jerk components are kinematic descriptions, not cognitive-strategy labels.

No raw Dropbox trajectories or participant-identifying information are included.
""",
        encoding="utf-8",
    )


def write_compact_results() -> None:
    out = BUNDLE / "relevant_results"
    out.mkdir(parents=True, exist_ok=True)
    copy(
        config.RESULTS_DIR / "summary" / "model_comparison_by_seed.csv",
        out / "model_comparison_by_seed.csv",
    )
    copy(
        config.RESULTS_DIR / "latent_associations" / "latent_submovement_associations.csv",
        out / "latent_submovement_associations.csv",
    )
    for window_mode in config.WINDOW_MODES:
        copy(
            config.RESULTS_DIR / window_mode / "baselines" / "kmeans_selection_corrected.csv",
            out / f"{window_mode}_kmeans.csv",
        )
        copy(
            config.RESULTS_DIR / window_mode / "generation" / "generation_summary.csv",
            out / f"{window_mode}_representative_generation.csv",
        )

    sub = pd.read_csv(config.RESULTS_DIR / "submovements_real.csv")
    counts = sub.mj_n_components.value_counts().sort_index().rename_axis(
        "component_count"
    ).reset_index(name="n_trials")
    counts["fraction"] = counts.n_trials / counts.n_trials.sum()
    counts.to_csv(out / "recorded_submovement_count_summary.csv", index=False)
    outcomes = sub.groupby(["responseText", "mj_n_components"]).size().rename(
        "n_trials"
    ).reset_index()
    outcomes["within_outcome_fraction"] = outcomes.n_trials / outcomes.groupby(
        "responseText"
    ).n_trials.transform("sum")
    outcomes.to_csv(out / "recorded_submovements_by_outcome.csv", index=False)


def write_email_draft() -> None:
    (SHARE / "EMAIL_DRAFT.txt").write_text(
        """Subject: Interception movement fingerprints - updated results and dashboard

Hi Jason and Moni,

We are attaching our updated preliminary report and an interactive dashboard for the interception-movement project.

Following Jason's concern about temporal resampling, we tested two matched trajectory definitions on the same condition-2 cohort: finger movement onset to arrival, and target motion onset to arrival. The second definition preserves the participant's waiting interval. In both cases, movement time and initiation time are withheld from the CVAE encoder and predicted separately by the decoder.

The dashboard lets you switch between both temporal definitions, latent widths n=2, 3, 4, and 8, and three training seeds; change the task condition and latent controls; inspect generated trajectory, timing, velocity, and minimum-jerk components; and review held-out distribution and participant-enrollment results.

The main result is a tradeoff rather than one universally best representation. The target-motion window gives substantially better initiation-time prediction and timing-distribution fidelity. The movement-only window better preserves physical execution and submovement-count distributions. n=3 is the smallest stable strategy-inclusive model, while n=8 is the stronger capacity comparison.

The report ends with the assumptions we would especially like Jason to confirm, including the one-second late-arrival threshold and the minimum-jerk component constraints/order rule. All decisions are configurable and rerunnable.

The ZIP contains no raw Dropbox trajectories or MAT files. On Windows, extract it and run setup_and_launch.bat once.

Best,
Seman and Paz
""",
        encoding="utf-8",
    )


def build() -> Path:
    SHARE.mkdir(parents=True, exist_ok=True)
    resolved_share = SHARE.resolve()
    resolved_bundle = BUNDLE.resolve()
    if resolved_share not in resolved_bundle.parents:
        raise RuntimeError("Refusing to rebuild a bundle outside the study share directory")
    BUNDLE.mkdir(parents=True)

    copy(ROOT / "config.py", BUNDLE / "config.py")
    copy(ROOT / "requirements_dashboard.txt", BUNDLE / "requirements_dashboard.txt")
    copy(ROOT / ".streamlit" / "config.toml", BUNDLE / ".streamlit" / "config.toml")
    for name in [
        "__init__.py", "strategy_dashboard.py", "features.py", "submovements.py",
        "vae_model.py", "preprocessing.py",
    ]:
        copy(ROOT / "src" / name, BUNDLE / "src" / name)

    study = BUNDLE / "studies" / "strategy_window_comparison"
    copy(config.RESULTS_DIR / "model_seed_results.csv", study / "results" / "model_seed_results.csv")
    for source in (config.RESULTS_DIR / "dashboard").glob("*"):
        copy(source, study / "results" / "dashboard" / source.name)
    associations = config.RESULTS_DIR / "latent_associations"
    if associations.exists():
        for source in associations.rglob("*"):
            if source.is_file():
                copy(source, study / "results" / "latent_associations" / source.relative_to(associations))
    for window_mode in config.WINDOW_MODES:
        model_root = config.RESULTS_DIR / window_mode / "models"
        for run in model_root.glob("cvae_*_z*_seed*"):
            copy(
                run / "checkpoint.pt",
                study / "results" / window_mode / "models" / run.name / "checkpoint.pt",
            )
        generation = config.RESULTS_DIR / window_mode / "generation"
        if generation.exists():
            for source in generation.glob("*.csv"):
                copy(source, study / "results" / window_mode / "generation" / source.name)
            copy(generation / "protocol.json", study / "results" / window_mode / "generation" / "protocol.json")

    report = config.STUDY_ROOT / "output" / "pdf" / "Interception_Strategy_Window_Comparison.pdf"
    copy(report, study / "output" / "pdf" / report.name)
    copy(report, SHARE / report.name)
    write_text_files()
    write_compact_results()
    write_email_draft()

    zip_path = SHARE / "Interception_Strategy_Dashboard.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(SHARE))
    print(f"Dashboard bundle: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    return zip_path


if __name__ == "__main__":
    build()
