"""Restore review data from the committed release; no raw recordings are required."""
from pathlib import Path, PurePosixPath
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(root: Path, replace: bool = False) -> int:
    root = root.resolve()
    archive = root / "output/release/Interception_Movements_Advisor_Package.zip"
    pending = []
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"Unsafe archive entry: {name}")
            if len(member.parts) < 2 or name.endswith("/"):
                continue
            relative = PurePosixPath(*member.parts[1:])
            path = relative.as_posix()
            if path.startswith("evaluation/") and len(relative.parts) >= 3 and relative.suffix in {".csv", ".json"}:
                relative = PurePosixPath("studies/review_corrected_evaluation/results", *relative.parts[1:])
            elif not (path.startswith("studies/review_corrected_evaluation/results/dashboard/")
                      or path.startswith("studies/final_strategy_evaluation/runs/cvae/fold0/")
                      or path in {"external/jason-submovements/LICENSE", "external/jason-submovements/python/movement_decompose_2d.py"}):
                continue
            destination = (root / Path(relative)).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Destination escapes checkout: {destination}")
            data = bundle.read(name)
            if destination.exists():
                if destination.read_bytes() == data:
                    continue
                if not replace:
                    raise FileExistsError(f"Different existing artifact: {destination}. Use --replace-artifacts only to restore the published review snapshot.")
            pending.append((destination, data))
    # Validate every destination before writing any files.
    for destination, data in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return len(pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-artifacts", action="store_true")
    args = parser.parse_args()
    print(f"Restored {restore(ROOT, args.replace_artifacts)} review artifacts. Raw-data training is a separate workflow.")
