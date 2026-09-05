"""Build the email-ready advisor package without raw data or launch scripts."""
from __future__ import annotations

from pathlib import Path
import shutil
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "release"
PACKAGE_NAME = "Interception_Movements_Advisor_Package"
ARCHIVE = OUTPUT / "Interception_Movements_Advisor_Package.zip"
ASSETS = ROOT / "studies" / "review_corrected_evaluation" / "results" / "dashboard"
RUNS = ROOT / "studies" / "final_strategy_evaluation" / "runs" / "cvae" / "fold0"

SOURCE_FILES = [
    ROOT / ".streamlit" / "config.toml",
    ROOT / "src" / "__init__.py",
    ROOT / "src" / "confirmatory_dashboard.py",
    ROOT / "src" / "features.py",
    ROOT / "src" / "preprocessing.py",
    ROOT / "src" / "submovements.py",
    ROOT / "src" / "vae_model.py",
]
REPORTS = [
    ROOT / "output" / "pdf" / "Interception_Movements_Final_Scientific_Report.pdf",
    ROOT / "output" / "pdf" / "Interception_Movements_Results_Guide.pdf",
    ROOT / "output" / "pdf" / "Interception_Movements_Supplementary_Appendix.pdf",
]
RELEASE_DOCS = [
    ROOT / "release" / "README.txt",
    ROOT / "release" / "requirements_dashboard.txt",
    ROOT / "release" / "THIRD_PARTY_NOTICES.txt",
    ROOT / "PAZ_REVIEW_HANDOFF.md",
]
ASSET_FILES = [
    "manifest.json",
    "latent_stats.json",
    "subject_fingerprints.csv",
    "empirical_query_features.csv",
    "generated_validation.csv",
    "model_comparison.csv",
    "condition_speed_ranges.csv",
    "condition_trajectory_summary.csv",
    "timing_outlier_audit.csv",
    "minimum_jerk_summary.csv",
    "minimum_jerk_sensitivity.csv",
    "behavioral_probe_summary.csv",
    "timing_fairness_summary.csv",
    "timing_fairness_paired.csv",
    "submovement_sampling_reference.csv",
    "event_audit.json",
    "fingerprint_control_summary.csv",
    "fingerprint_control_paired.csv",
    "direct_context_summary.csv",
    "matched_component_summary.csv",
    "matched_component_diagnostics.csv",
    "matched_component_sampling.csv",
]


def copy_to_bundle(source: Path, relative: Path, bundle: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination = bundle / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_bundle(bundle: Path) -> None:
    if bundle.exists():
        raise RuntimeError(f"refusing to replace an existing staging directory: {bundle}")
    bundle.mkdir(parents=True)

    for source in RELEASE_DOCS:
        copy_to_bundle(source, Path(source.name), bundle)
    for source in sorted((ROOT / "review_evidence").iterdir()):
        if source.is_file() and source.suffix in {".md", ".csv", ".json"}:
            copy_to_bundle(source, Path("review_evidence") / source.name, bundle)
    copy_to_bundle(ROOT / "release" / "config.py", Path("config.py"), bundle)
    for source in SOURCE_FILES:
        copy_to_bundle(source, source.relative_to(ROOT), bundle)
    for source in REPORTS:
        copy_to_bundle(source, Path("reports") / source.name, bundle)
    for name in ASSET_FILES:
        source = ASSETS / name
        copy_to_bundle(
            source,
            Path("studies") / "review_corrected_evaluation" / "results" / "dashboard" / name,
            bundle,
        )
    for latent_dim in (2, 3, 4, 8):
        run = f"cvae_z{latent_dim}_seed42"
        source = RUNS / run / "checkpoint.pt"
        copy_to_bundle(
            source,
            Path("studies") / "final_strategy_evaluation" / "runs" / "cvae" /
            "fold0" / run / "checkpoint.pt",
            bundle,
        )

    corrected = ROOT / "studies/review_corrected_evaluation"
    copy_to_bundle(corrected / "PROTOCOL.md", Path("evaluation/PROTOCOL.md"), bundle)
    copy_to_bundle(corrected / "VERIFICATION.json", Path("evaluation/VERIFICATION.json"), bundle)
    for folder in ("analysis", "behavioral_probes", "timing_fairness", "sampling_reference", "event_audit", "review_controls"):
        for path in sorted((corrected / "results" / folder).glob("*")):
            if path.is_file() and path.suffix in {".csv", ".json"}:
                copy_to_bundle(path, Path("evaluation") / folder / path.name, bundle)
    for path in sorted((corrected / "results/review_controls/fingerprint").glob("*.csv")):
        copy_to_bundle(path, Path("evaluation/review_controls/fingerprint") / path.name, bundle)
    licence = ROOT / "external/jason-submovements/LICENSE"
    if not licence.exists():
        raise FileNotFoundError("Upstream submovements licence is required in the bundle")
    copy_to_bundle(licence, Path("licenses/submovements-LICENSE.txt"), bundle)
    for relative in ("external/jason-submovements/LICENSE", "external/jason-submovements/python/movement_decompose_2d.py"):
        copy_to_bundle(ROOT / relative, Path(relative), bundle)

    # Reports are available both beside the README and under output/pdf so the
    # standalone dashboard resolves the same paths as the research checkout.
    for source in REPORTS:
        copy_to_bundle(source, Path("output") / "pdf" / source.name, bundle)


def build_archive(bundle: Path) -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, Path(PACKAGE_NAME) / path.relative_to(bundle))


def audit_archive() -> None:
    forbidden_suffixes = {".bat", ".cmd", ".ps1", ".exe", ".dll"}
    forbidden_fragments = {"dropbox", "data_raw", "processed_trials", "__pycache__"}
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        if not names:
            raise RuntimeError("empty release archive")
        for name in names:
            lower = name.lower()
            if Path(name).suffix.lower() in forbidden_suffixes:
                raise RuntimeError(f"forbidden executable content in release: {name}")
            if any(fragment in lower for fragment in forbidden_fragments):
                raise RuntimeError(f"forbidden raw/cache path in release: {name}")
            if Path(name).suffix.lower() in {".py", ".txt", ".toml", ".json"}:
                text = archive.read(name).decode("utf-8", errors="ignore").lower()
                if "d:\\dropbox" in text or "d:/dropbox" in text:
                    raise RuntimeError(f"local raw-data path leaked into release: {name}")
        if not any(name.endswith("confirmatory_dashboard.py") for name in names):
            raise RuntimeError("dashboard source is missing")
        if sum(name.endswith("checkpoint.pt") for name in names) != 4:
            raise RuntimeError("expected four live reference checkpoints")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    staging_root = ROOT / ".tmp" / f"advisor_package_{uuid.uuid4().hex}"
    resolved_tmp = (ROOT / ".tmp").resolve()
    if resolved_tmp not in staging_root.resolve().parents:
        raise RuntimeError(f"invalid staging path: {staging_root}")
    try:
        bundle = staging_root / PACKAGE_NAME
        build_bundle(bundle)
        build_archive(bundle)
    finally:
        if staging_root.exists() and resolved_tmp in staging_root.resolve().parents:
            shutil.rmtree(staging_root, ignore_errors=True)
    audit_archive()
    with zipfile.ZipFile(ARCHIVE) as archive:
        print(f"Built {ARCHIVE}")
        print(f"Entries: {len(archive.namelist())}")
        print(f"Size: {ARCHIVE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
