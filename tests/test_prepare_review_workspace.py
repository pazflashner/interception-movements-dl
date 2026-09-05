"""Release restoration must neither escape the checkout nor overwrite work silently."""
import zipfile
import pytest
from scripts.prepare_review_workspace import restore


def archive(root, entries):
    path = root / "output/release/Interception_Movements_Advisor_Package.zip"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in entries.items():
            bundle.writestr("Package/" + name, data)


def test_restore_maps_evidence_and_preserves_source(tmp_path):
    archive(tmp_path, {"evaluation/review_controls/example.csv": "value\n1\n", "src/example.py": "unwanted"})
    assert restore(tmp_path) == 1
    assert (tmp_path / "studies/review_corrected_evaluation/results/review_controls/example.csv").read_text() == "value\n1\n"
    assert not (tmp_path / "src/example.py").exists()
    assert restore(tmp_path) == 0


def test_restore_refuses_conflicting_artifact(tmp_path):
    archive(tmp_path, {"evaluation/review_controls/example.csv": "new"})
    target = tmp_path / "studies/review_corrected_evaluation/results/review_controls/example.csv"
    target.parent.mkdir(parents=True)
    target.write_text("existing")
    with pytest.raises(FileExistsError):
        restore(tmp_path)
    assert target.read_text() == "existing"
    assert restore(tmp_path, replace=True) == 1


def test_restore_rejects_path_escape_before_writing(tmp_path):
    archive(tmp_path, {"evaluation/review_controls/example.csv": "ok", "../escaped.csv": "bad"})
    with pytest.raises(ValueError):
        restore(tmp_path)
    assert not (tmp_path / "studies").exists()
