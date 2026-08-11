"""Validate the advisor PDF and both shareable dashboard packages."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


RELEASE = config.STUDY_ROOT / "output" / "advisor_release"
BRIEF = (
    config.STUDY_ROOT
    / "output"
    / "advisor_brief"
    / "Interception_Movement_Advisor_Brief.pdf"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def only_name(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    require(len(matches) == 1, f"archive contains exactly one {suffix}")
    return matches[0]


def validate_dashboard(path: Path, expected_seeds: list[int]) -> None:
    require(path.exists() and path.stat().st_size > 1_000_000, f"{path.name} exists")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        forbidden = [
            name
            for name in names
            if "canonical_trials" in name.lower()
            or name.lower().endswith((".mat", ".pkl"))
        ]
        require(not forbidden, f"{path.name} contains no raw or processed trial cache")
        manifest_name = only_name(names, "/results/dashboard/manifest.json")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        require(manifest["model_seeds"] == expected_seeds, f"{path.name} exposes the packaged seeds")
        require(manifest["latent_dimensions"] == [2, 3, 4, 8], f"{path.name} exposes n=2/3/4/8")
        require(manifest["windows"] == ["movement_only", "go_to_arrival"], f"{path.name} contains both protocols")
        checkpoints = [name for name in names if name.endswith("/checkpoint.pt")]
        require(len(checkpoints) == 8 * len(expected_seeds), f"{path.name} has every expected checkpoint")
        results_name = only_name(names, "/results/model_seed_results.csv")
        results = pd.read_csv(io.BytesIO(archive.read(results_name)))
        require(len(results) == 24, f"{path.name} retains all repeated-seed metrics")
        only_name(names, "/output/advisor_brief/Interception_Movement_Advisor_Brief.pdf")
        only_name(names, "/src/strategy_dashboard.py")
        only_name(names, "/setup_and_launch.bat")


def validate_outer_package(path: Path) -> None:
    require(path.exists() and path.stat().st_size > 1_000_000, "advisor review package exists")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        only_name(names, "/Interception_Movement_Advisor_Brief.pdf")
        dashboard_name = only_name(names, "/Interception_Strategy_Dashboard_Email.zip")
        only_name(names, "/EMAIL_DRAFT.txt")
        only_name(names, "/README_FIRST.txt")
        selected = [name for name in names if "/selected_results/" in name and not name.endswith("/")]
        require(len(selected) == 7, "advisor package contains the seven selected result files")
        with zipfile.ZipFile(io.BytesIO(archive.read(dashboard_name))) as dashboard:
            dashboard_names = dashboard.namelist()
            manifest_name = only_name(dashboard_names, "/results/dashboard/manifest.json")
            manifest = json.loads(dashboard.read(manifest_name).decode("utf-8"))
            require(manifest["model_seeds"] == [42], "nested email dashboard is the compact build")


def main() -> None:
    require(BRIEF.exists() and BRIEF.stat().st_size > 100_000, "advisor brief exists and is nontrivial")
    reader = PdfReader(BRIEF)
    require(
        len(reader.pages) == 21,
        "advisor brief has eight main pages, an appendix cover, and twelve appendix pages",
    )
    first_text = reader.pages[0].extract_text() or ""
    require("Interception-Movement Fingerprints" in first_text, "advisor brief title is readable")
    full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    for heading in (
        "A4. Held-out posterior trajectory reconstruction",
        "A5. Fingerprint-conditioned held-out generation",
        "A6. Minimum-jerk analysis after generation",
    ):
        require(heading in full_text, f"advisor brief includes {heading}")

    validate_dashboard(RELEASE / "Interception_Strategy_Dashboard_Email.zip", [42])
    validate_dashboard(RELEASE / "Interception_Strategy_Dashboard_Full.zip", [42, 43, 44])
    validate_outer_package(RELEASE / "Interception_Advisor_Review_Package.zip")
    print("Advisor release validation completed successfully.")


if __name__ == "__main__":
    main()
