"""Build compact and full raw-data-free advisor review packages."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


SHARE = config.STUDY_ROOT / "output" / "advisor_release"
STAGING = ROOT / "tmp" / "release_staging"
BRIEF = (
    config.STUDY_ROOT
    / "output"
    / "advisor_brief"
    / "Interception_Movement_Advisor_Brief.pdf"
)
EMAIL_DASHBOARD_ZIP = SHARE / "Interception_Strategy_Dashboard_Email.zip"
FULL_DASHBOARD_ZIP = SHARE / "Interception_Strategy_Dashboard_Full.zip"
ADVISOR_PACKAGE_ZIP = SHARE / "Interception_Advisor_Review_Package.zip"


def copy(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def zip_directory(source: Path, destination: Path, base: Path) -> None:
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(base))


def dashboard_readme(seeds: list[int], compact: bool) -> str:
    seed_text = ", ".join(map(str, seeds))
    package_note = (
        "This email-sized build uses seed 42 for live generation. The Model "
        "comparison tab still contains aggregate evidence from seeds 42, 43, and 44."
        if compact
        else "This full build contains live checkpoints for seeds 42, 43, and 44."
    )
    return f"""INTERCEPTION MOVEMENT STRATEGY DASHBOARD

QUICK START (WINDOWS)
1. Extract this ZIP.
2. Double-click setup_and_launch.bat on the first run.
3. Later, double-click launch_dashboard.bat.

WHAT IS INCLUDED
- One dashboard for both temporal protocols.
- Latent widths n=2, 3, 4, and 8.
- Live checkpoint seed(s): {seed_text}.
- Task-condition controls: start/speed category, side, and executed target speed.
- Population-center and enrolled held-out-participant fingerprints.
- Generated x-y trajectory, timing, speed, and minimum-jerk components.
- Held-out validation, repeated-seed comparison, latent heatmaps, and protocol notes.

RECOMMENDED START
Select Target motion onset -> arrival, n=3, and seed 42. This is the primary
low-dimensional strategy-inclusive model. Use n=8 as the capacity comparison.

PACKAGE NOTE
{package_note}

INTERPRETATION BOUNDARIES
- Condition 2 only; 2-D x-y table plane; 100 phase samples.
- Initiation and movement time are withheld from the encoder and predicted separately.
- Latent axes can rotate, reflect, or swap across independently trained models.
- Minimum-jerk components are kinematic descriptions, not cognitive labels.
- Held-out fingerprint accuracy is closed-set enrollment, not zero-shot identification.

No raw Dropbox trajectories, MAT files, or processed trial caches are included.
"""


def write_launchers(bundle: Path) -> None:
    (bundle / "launch_dashboard.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "python -m streamlit run src\\strategy_dashboard.py\r\n"
        "pause\r\n",
        encoding="ascii",
    )
    (bundle / "setup_and_launch.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "python -m pip install -r requirements_dashboard.txt\r\n"
        "if errorlevel 1 pause & exit /b 1\r\n"
        "call launch_dashboard.bat\r\n",
        encoding="ascii",
    )


def build_dashboard_bundle(seeds: list[int], zip_path: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bundle = STAGING / f"{label}_{stamp}"
    bundle.mkdir(parents=True)

    copy(ROOT / "config.py", bundle / "config.py")
    copy(ROOT / "requirements_dashboard.txt", bundle / "requirements_dashboard.txt")
    copy(ROOT / ".streamlit" / "config.toml", bundle / ".streamlit" / "config.toml")
    for name in (
        "__init__.py",
        "strategy_dashboard.py",
        "features.py",
        "submovements.py",
        "vae_model.py",
        "preprocessing.py",
    ):
        copy(ROOT / "src" / name, bundle / "src" / name)

    study = bundle / "studies" / "strategy_window_comparison"
    copy(
        config.RESULTS_DIR / "model_seed_results.csv",
        study / "results" / "model_seed_results.csv",
    )
    dashboard_assets = config.RESULTS_DIR / "dashboard"
    for source in dashboard_assets.glob("*"):
        if source.name != "manifest.json":
            copy(source, study / "results" / "dashboard" / source.name)
    manifest = json.loads(
        (dashboard_assets / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["model_seeds"] = seeds
    manifest_path = study / "results" / "dashboard" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    associations = config.RESULTS_DIR / "latent_associations"
    for source in associations.rglob("*"):
        if source.is_file():
            copy(
                source,
                study
                / "results"
                / "latent_associations"
                / source.relative_to(associations),
            )

    for window_mode in config.WINDOW_MODES:
        for dim in (2, 3, 4, 8):
            for seed in seeds:
                run = f"cvae_{window_mode}_z{dim}_seed{seed}"
                copy(
                    config.RESULTS_DIR
                    / window_mode
                    / "models"
                    / run
                    / "checkpoint.pt",
                    study
                    / "results"
                    / window_mode
                    / "models"
                    / run
                    / "checkpoint.pt",
                )
        generation = config.RESULTS_DIR / window_mode / "generation"
        for source in generation.glob("*.csv"):
            copy(
                source,
                study / "results" / window_mode / "generation" / source.name,
            )
        copy(
            generation / "protocol.json",
            study / "results" / window_mode / "generation" / "protocol.json",
        )

    copy(
        BRIEF,
        study
        / "output"
        / "advisor_brief"
        / "Interception_Movement_Advisor_Brief.pdf",
    )
    write_launchers(bundle)
    (bundle / "README.txt").write_text(
        dashboard_readme(seeds, compact=len(seeds) == 1), encoding="utf-8"
    )

    zip_directory(bundle, zip_path, STAGING)
    return zip_path


def email_text() -> str:
    return """Subject: Interception-movement project - progress update and dashboard

Dear Jason and Moni,

We apologize for the two-month delay. We had to focus on obligations in other courses, but during the past two weeks Paz and I have returned to the project and made substantial progress. We are happy to share our current analysis and dashboard with you.

After testing several alternatives, we focused on a conditional VAE and compared two matched temporal definitions on the same condition-2 trials:

1. Target motion onset to finger arrival. This preserves the participant's waiting period and therefore includes information about movement-initiation strategy.
2. Finger movement onset to arrival. This removes the waiting period and dedicates the normalized trajectory to movement execution and correction.

Because the trajectories are resampled to a fixed number of points, physical initiation and movement time are withheld from the encoder and predicted separately by the decoder. The two definitions provide complementary information, so the report presents the strategy-inclusive model as the primary analysis and the movement-only model as an execution-focused comparison.

The attached PDF briefly explains the methods, held-out results, latent-variable heatmaps, minimum-jerk submovement analysis, and the assumptions we made while preparing the study.

One assumption particularly needs Jason's input. Some condition-2 trials contain the MAT response "Not fixating on the dot enough!!!", even though condition 2 allows free eye movements. We retained these trials provisionally. Could you please confirm whether this message is a technical or irrelevant flag in condition 2, or whether those trials should be excluded?

We also attached one interactive dashboard covering both temporal definitions and latent dimensions n=2, 3, 4, and 8. It allows the task conditions and latent values to be changed and shows the resulting trajectory, timing, speed, submovement decomposition, held-out validation, and latent associations. The attached compact version uses seed 42 for live generation, while its comparison tab reports the repeated-seed results. Instructions are included in the ZIP.

We would appreciate any comments on the approach, evaluation design, interpretations, and assumptions listed at the end of the PDF. After your review, we can make the necessary adjustments and rerun the affected stages. The code is modular, so most technical or preprocessing changes can be reproduced without rebuilding the project manually.

Thank you for your time and guidance. We look forward to your feedback.

Best,
Simaan and Paz
"""


def release_readme() -> str:
    return """ADVISOR RELEASE FILES

RECOMMENDED EMAIL ATTACHMENTS
1. Interception_Movement_Advisor_Brief.pdf (about 3 MB): attach directly so it
   can be previewed without extracting an archive.
2. Interception_Strategy_Dashboard_Email.zip (about 11 MB): compact interactive
   dashboard with both protocols and n=2/3/4/8.

SINGLE-ARCHIVE ALTERNATIVE
- Interception_Advisor_Review_Package.zip (about 14 MB): advisor brief, compact
  dashboard, email draft, and selected machine-readable result tables.

OPTIONAL FULL DASHBOARD
- Interception_Strategy_Dashboard_Full.zip (24.8 MB): both temporal protocols,
  n=2/3/4/8, and live checkpoints for seeds 42/43/44.

STANDALONE COMPACT DASHBOARD
- Interception_Strategy_Dashboard_Email.zip (11.1 MB): both temporal protocols
  and n=2/3/4/8, with seed 42 for live generation and all-seed comparison tables.

There is one dashboard interface, not one dashboard per temporal protocol.
The protocol selector switches between movement onset -> arrival and target
motion onset -> arrival.
"""


def write_selected_results(destination: Path) -> None:
    files = {
        config.RESULTS_DIR / "model_seed_results.csv": "model_seed_results.csv",
        config.RESULTS_DIR
        / "summary"
        / "model_comparison_by_seed.csv": "model_comparison_by_seed.csv",
        config.RESULTS_DIR
        / "latent_associations"
        / "latent_submovement_associations.csv": "latent_submovement_associations.csv",
        config.RESULTS_DIR
        / "go_to_arrival"
        / "generation"
        / "generation_summary.csv": "strategy_generation_summary.csv",
        config.RESULTS_DIR
        / "movement_only"
        / "generation"
        / "generation_summary.csv": "execution_generation_summary.csv",
        config.STUDY_ROOT
        / "data_audit"
        / "condition2_trial_completion_audit.csv": "condition2_trial_completion_audit.csv",
        config.STUDY_ROOT
        / "data_audit"
        / "TRIAL_COMPLETION_AUDIT.md": "TRIAL_COMPLETION_AUDIT.md",
    }
    for source, name in files.items():
        copy(source, destination / name)


def build_outer_package() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    package = STAGING / f"Interception_Advisor_Review_{stamp}"
    package.mkdir(parents=True)
    copy(BRIEF, package / BRIEF.name)
    copy(EMAIL_DASHBOARD_ZIP, package / EMAIL_DASHBOARD_ZIP.name)
    (package / "EMAIL_DRAFT.txt").write_text(email_text(), encoding="utf-8")
    (package / "README_FIRST.txt").write_text(
        """INTERCEPTION MOVEMENT ADVISOR REVIEW PACKAGE

1. Interception_Movement_Advisor_Brief.pdf
   Eight-page main story followed by a complete technical appendix.

2. Interception_Strategy_Dashboard_Email.zip
   One dashboard for both temporal protocols and n=2/3/4/8. The compact build
   uses seed 42 for live generation and retains all-seed aggregate comparisons.

3. selected_results/
   Machine-readable result tables supporting the report. These are optional;
   the brief and dashboard contain the intended narrative.

4. EMAIL_DRAFT.txt
   Suggested covering message.

No raw Dropbox files, MAT files, or processed trajectory caches are included.
The separate Full dashboard ZIP contains live checkpoints for all three seeds.
""",
        encoding="utf-8",
    )
    write_selected_results(package / "selected_results")
    zip_directory(package, ADVISOR_PACKAGE_ZIP, STAGING)
    return ADVISOR_PACKAGE_ZIP


def main() -> None:
    SHARE.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    if not BRIEF.exists():
        raise FileNotFoundError(
            f"Build the advisor brief before packaging: {BRIEF}"
        )
    build_dashboard_bundle(
        [42], EMAIL_DASHBOARD_ZIP, "Interception_Strategy_Dashboard_Email"
    )
    build_dashboard_bundle(
        [42, 43, 44], FULL_DASHBOARD_ZIP, "Interception_Strategy_Dashboard_Full"
    )
    package = build_outer_package()
    copy(BRIEF, SHARE / BRIEF.name)
    (SHARE / "EMAIL_DRAFT.txt").write_text(email_text(), encoding="utf-8")
    (SHARE / "README.txt").write_text(release_readme(), encoding="utf-8")
    print(f"Advisor package: {package}")
    print(f"Advisor package size: {package.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Email dashboard: {EMAIL_DASHBOARD_ZIP.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Full dashboard: {FULL_DASHBOARD_ZIP.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
