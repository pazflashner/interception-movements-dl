"""Create an email-sized dashboard bundle with models and compact result data."""
from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SHARE = ROOT / "output" / "share"
BUNDLE = SHARE / "Interception_Movement_Dashboard"


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_launchers() -> None:
    (BUNDLE / "launch_dashboard.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "python -m streamlit run src\\dashboard.py\r\n"
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


def write_readme() -> None:
    text = """INTERCEPTION MOVEMENT FINGERPRINT DASHBOARD

QUICK START (WINDOWS)
1. Extract this ZIP file.
2. Double-click setup_and_launch.bat the first time.
3. On later runs, double-click launch_dashboard.bat.
4. Streamlit opens the dashboard in the default browser.

The first setup installs Python packages and can take several minutes. Python 3.11 or newer is recommended.

DASHBOARD CONTENTS
- Generate: n=2, n=3, or n=8; latent sliders; task-condition controls; trajectory, velocity, timing, and minimum-jerk outputs.
- Distribution check: held-out recorded versus generated distributions with KS/Wasserstein or JSD/total variation.
- Model comparison: the repeated-seed study results.
- Protocol and downloads: assumptions, questions, PDF, and compact tables.

SCIENTIFIC SCOPE
- The models use the x-y table plane and condition 2 only.
- True timing is withheld from the encoder and predicted by the decoder.
- Participant presets use context trials; comparisons use disjoint query trials.
- n=8 gives the strongest overall fidelity. n=2 and n=3 provide lower-dimensional controls but do not reproduce every distribution.
- Minimum-jerk components are kinematic patterns, not direct cognitive-strategy labels.

DATA INCLUDED
This bundle contains trained model checkpoints, generated validation samples, held-out query feature summaries, and aggregate results. It contains no raw Dropbox trajectory CSVs or identifying participant information.
"""
    (BUNDLE / "README.txt").write_text(text, encoding="utf-8")


def write_email_draft() -> None:
    text = """Subject: Interception movement fingerprint - preliminary report and dashboard

Hi Jason,

Attached are our current report and an interactive dashboard for the interception-movement model.

The dashboard lets you switch between n=2, n=3, and n=8, change the latent variables and task condition, inspect generated trajectories, velocity and minimum-jerk components, and compare recorded and generated held-out distributions for different outputs.

The main result so far is that participant-specific information is learnable, while exact distribution reproduction remains feature-dependent. n=8 gives the strongest overall fidelity; n=2 and n=3 preserve the low-dimensional control goal but lose some detail. We have kept the modeling assumptions and questions requiring your confirmation explicit in both the report and dashboard.

To run the dashboard, extract the ZIP and double-click setup_and_launch.bat. No raw experiment files are included.

We would appreciate your feedback, especially on the assumptions listed in the final section of the report.

Best,
Seman and Paz
"""
    (SHARE / "EMAIL_DRAFT.txt").write_text(text, encoding="utf-8")


def compact_results() -> None:
    results_dir = BUNDLE / "relevant_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_results = pd.read_csv(ROOT / "results" / "final_study" / "core_models" / "seed_results.csv")
    generation = pd.read_csv(ROOT / "results" / "final_study" / "generation" / "generation_summary.csv")
    seed_results.to_csv(results_dir / "cvae_seed_results.csv", index=False)
    generation.to_csv(results_dir / "generation_validation_summary.csv", index=False)

    sub = pd.read_csv(ROOT / "results" / "final_study" / "submovements_real.csv")
    sub = sub[sub.mj_fit_success == True].copy()  # noqa: E712
    count_summary = sub.mj_n_components.value_counts().sort_index().rename_axis("component_count").reset_index(name="n_trials")
    count_summary["fraction"] = count_summary.n_trials / count_summary.n_trials.sum()
    count_summary.to_csv(results_dir / "submovement_count_summary.csv", index=False)
    outcome = (
        sub.groupby(["responseText", "mj_n_components"]).size()
        .rename("n_trials").reset_index()
    )
    outcome["within_outcome_fraction"] = outcome.n_trials / outcome.groupby("responseText").n_trials.transform("sum")
    outcome.to_csv(results_dir / "submovement_by_recorded_outcome.csv", index=False)


def build() -> Path:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    copy(ROOT / "src" / "dashboard.py", BUNDLE / "src" / "dashboard.py")
    for name in ["__init__.py", "features.py", "submovements.py", "vae_model.py", "preprocessing.py"]:
        copy(ROOT / "src" / name, BUNDLE / "src" / name)
    copy(ROOT / "config.py", BUNDLE / "config.py")
    copy(ROOT / ".streamlit" / "config.toml", BUNDLE / ".streamlit" / "config.toml")
    copy(ROOT / "requirements_dashboard.txt", BUNDLE / "requirements_dashboard.txt")
    report = ROOT / "output" / "final_report" / "Interception_Movement_Fingerprints_Final.pdf"
    copy(report, BUNDLE / report.name)
    copy(report, SHARE / report.name)

    for source in (ROOT / "results" / "final_study" / "dashboard").glob("*"):
        copy(source, BUNDLE / "results" / "final_study" / "dashboard" / source.name)
    for run in (ROOT / "results" / "final_study" / "core_models").glob("trajectory_only_z*_seed*"):
        copy(run / "checkpoint.pt", BUNDLE / "results" / "final_study" / "core_models" / run.name / "checkpoint.pt")
    copy(
        ROOT / "results" / "final_study" / "core_models" / "seed_results.csv",
        BUNDLE / "results" / "final_study" / "core_models" / "seed_results.csv",
    )
    generation_dir = ROOT / "results" / "final_study" / "generation"
    for source in generation_dir.glob("trajectory_only_z*_seed*_generated.csv"):
        copy(source, BUNDLE / "results" / "final_study" / "generation" / source.name)
    copy(generation_dir / "generation_summary.csv", BUNDLE / "results" / "final_study" / "generation" / "generation_summary.csv")

    write_launchers()
    write_readme()
    compact_results()
    write_email_draft()

    zip_path = SHARE / "Interception_Movement_Dashboard.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(SHARE))
    return zip_path


if __name__ == "__main__":
    result = build()
    print(result)
    print(f"size_mb={result.stat().st_size / 1024 / 1024:.2f}")
