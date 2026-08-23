"""Frozen participant-level protocol for the final strategy-window study.

The outer folds measure uncertainty from changing the held-out participant
cohort. Training seeds are deliberately separate: they measure optimisation
noise while leaving the participant assignment unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


PROTOCOL_VERSION = "strategy-confirmatory-v1"
OUTER_FOLD_SEED = 2026
VALIDATION_SEED = 2027
N_OUTER_FOLDS = 4
N_TRAIN = 17
N_VALIDATION = 4
N_TEST = 7


@dataclass(frozen=True)
class ParticipantFold:
    """One outer participant fold with a train-only validation partition."""

    fold: int
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]

    def to_dict(self, trial_counts: dict[str, int] | None = None) -> dict:
        payload = {
            "fold": self.fold,
            "train_subjects": list(self.train_subjects),
            "validation_subjects": list(self.validation_subjects),
            "test_subjects": list(self.test_subjects),
        }
        if trial_counts is not None:
            payload["trial_counts"] = {
                split: int(sum(trial_counts[s] for s in subjects))
                for split, subjects in (
                    ("train", self.train_subjects),
                    ("validation", self.validation_subjects),
                    ("test", self.test_subjects),
                )
            }
        return payload


def make_participant_folds(subjects: list[str] | tuple[str, ...]) -> list[ParticipantFold]:
    """Create four deterministic, disjoint seven-participant outer folds."""
    unique = np.asarray(sorted(set(subjects)), dtype=object)
    expected = N_OUTER_FOLDS * N_TEST
    if len(unique) != expected:
        raise ValueError(
            f"protocol requires exactly {expected} participants; found {len(unique)}"
        )

    outer_rng = np.random.RandomState(OUTER_FOLD_SEED)
    shuffled = unique.copy()
    outer_rng.shuffle(shuffled)

    folds: list[ParticipantFold] = []
    for fold_index, test_array in enumerate(np.array_split(shuffled, N_OUTER_FOLDS)):
        test = set(test_array.tolist())
        development = np.asarray([s for s in unique if s not in test], dtype=object)
        validation_rng = np.random.RandomState(VALIDATION_SEED + fold_index)
        validation_rng.shuffle(development)
        validation = tuple(sorted(development[:N_VALIDATION].tolist()))
        train = tuple(sorted(development[N_VALIDATION:].tolist()))
        folds.append(
            ParticipantFold(
                fold=fold_index,
                train_subjects=train,
                validation_subjects=validation,
                test_subjects=tuple(sorted(test)),
            )
        )

    validate_participant_folds(folds, unique.tolist())
    return folds


def validate_participant_folds(
    folds: list[ParticipantFold], all_subjects: list[str] | tuple[str, ...]
) -> None:
    """Refuse manifests with leakage, missing participants, or repeated tests."""
    expected = set(all_subjects)
    if len(folds) != N_OUTER_FOLDS:
        raise ValueError(f"expected {N_OUTER_FOLDS} folds; found {len(folds)}")

    test_appearances: Counter[str] = Counter()
    for split in folds:
        train = set(split.train_subjects)
        validation = set(split.validation_subjects)
        test = set(split.test_subjects)
        if (len(train), len(validation), len(test)) != (N_TRAIN, N_VALIDATION, N_TEST):
            raise ValueError(
                f"fold {split.fold} has sizes {(len(train), len(validation), len(test))}"
            )
        if train & validation or train & test or validation & test:
            raise ValueError(f"participant leakage in fold {split.fold}")
        if train | validation | test != expected:
            raise ValueError(f"fold {split.fold} does not partition the full cohort")
        test_appearances.update(test)

    if set(test_appearances) != expected or set(test_appearances.values()) != {1}:
        raise ValueError("every participant must appear in exactly one outer test fold")


def partition_trials(trials: list[dict], fold: ParticipantFold):
    """Partition trial dictionaries using an already frozen participant fold."""
    train_ids = set(fold.train_subjects)
    validation_ids = set(fold.validation_subjects)
    test_ids = set(fold.test_subjects)
    train = [t for t in trials if t["metadata"]["subject"] in train_ids]
    validation = [t for t in trials if t["metadata"]["subject"] in validation_ids]
    test = [t for t in trials if t["metadata"]["subject"] in test_ids]
    if len(train) + len(validation) + len(test) != len(trials):
        raise ValueError("one or more trials do not belong to the frozen participant cohort")
    return train, validation, test


def write_or_verify_manifest(path: Path, trials: list[dict]) -> list[ParticipantFold]:
    """Write the immutable split manifest, or verify an existing one byte-for-data."""
    subjects = [t["metadata"]["subject"] for t in trials]
    counts = Counter(subjects)
    folds = make_participant_folds(subjects)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "outer_fold_seed": OUTER_FOLD_SEED,
        "validation_seed": VALIDATION_SEED,
        "participants": sorted(counts),
        "participant_trial_counts": dict(sorted(counts.items())),
        "folds": [fold.to_dict(counts) for fold in folds],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"existing split manifest differs from {PROTOCOL_VERSION}: {path}"
            )
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return folds


def write_or_verify_protocol(path: Path, payload: dict) -> None:
    """Persist a protocol once and reject semantic changes on later runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing protocol differs from requested protocol: {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
