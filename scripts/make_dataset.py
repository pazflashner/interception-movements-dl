"""
Build the model-ready dataset from raw trial CSVs.

Reads ``data/raw/<subject>/li_*.csv``, runs the preprocessing pipeline, and
writes three artefacts into ``data/processed/``:

    dataset.npz    arrays the model consumes
    metadata.csv   one row per trial: identifiers + kinematic features
    splits.json    the leave-N-subjects-out assignment for a given seed

Contents of ``dataset.npz``
---------------------------
    trajectories  (N, T, 3) float32  temporally + spatially normalised, mm
    timing        (N, 2)    float32  [movement_time_s, initiation_time_s]
    conditions    (N, 4)    float32  one-hot starting position (3) + side (1)
    subjects      (N,)      str      subject id per trial
    trial_ids     (N,)      str      unique trial id
    sp/side/rep   (N,)      int      raw metadata, for regrouping
    timing_features, normalised_length, recording_hz, condition   scalars/meta

``timing`` is the point of the whole file. Resampling every trial to T frames
equalises the input dimension but discards how long the movement took, leaving
the model blind to velocity. Carrying movement time and initiation time as
explicit channels lets the VAE reconstruct them (see ``src/vae_model.py``), so
a generated sample is a shape *and* a duration.

Usage
-----
    python scripts/make_dataset.py                 # build (reuses trial cache)
    python scripts/make_dataset.py --force         # re-run preprocessing
    python scripts/make_dataset.py --seed 7        # splits.json for seed 7
    python scripts/make_dataset.py --raw-dir path/to/raw
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_loading import load_dataset
from src.features import extract_features_dataframe
from src.preprocessing import preprocess_dataset
from src.train import split_subject_ids
from src.vae_model import encode_condition, encode_timing


# ── Trial preparation ─────────────────────────────────────────────────────────
def build_trials(raw_dir: Path, cache_path: Path, force: bool = False) -> list[dict]:
    """
    Preprocessed trials, from the pickle cache when it is available.

    Preprocessing the full dataset takes minutes, so the cache is reused by
    default; ``--force`` rebuilds it after any change to the preprocessing code.
    """
    if cache_path.exists() and not force:
        print(f"Loading cached trials from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Loading raw data from {raw_dir}")
    raw = load_dataset(raw_dir=raw_dir)
    print("Preprocessing...")
    trials = preprocess_dataset(raw)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(trials, f)
    print(f"Cached {len(trials)} preprocessed trials to {cache_path}")
    return trials


# ── Array assembly ────────────────────────────────────────────────────────────
def build_arrays(trials: list[dict]) -> dict[str, np.ndarray]:
    """Stack the per-trial dicts into the arrays stored in ``dataset.npz``."""
    trajectories, timing, conditions = [], [], []
    subjects, trial_ids, sp, side, rep = [], [], [], [], []

    for t in trials:
        meta = t["metadata"]
        trajectories.append(np.asarray(t["pos_norm"], dtype=np.float32))
        timing.append(encode_timing(t))
        conditions.append(encode_condition(meta.get("sp", 1), meta.get("side", 1)))
        subjects.append(str(meta.get("subject", "")))
        trial_ids.append(str(meta.get("trial_id", "")))
        sp.append(int(meta.get("sp", 0)))
        side.append(int(meta.get("side", 0)))
        rep.append(int(meta.get("rep", 0)))

    return {
        "trajectories": np.stack(trajectories),
        "timing": np.stack(timing),
        "conditions": np.stack(conditions),
        "subjects": np.array(subjects),
        "trial_ids": np.array(trial_ids),
        "sp": np.array(sp, dtype=np.int16),
        "side": np.array(side, dtype=np.int16),
        "rep": np.array(rep, dtype=np.int16),
    }


def validate(arrays: dict[str, np.ndarray]) -> None:
    """
    Fail loudly on anything that would silently poison training.

    A NaN in one trial propagates through the loss and turns every gradient into
    NaN, which is much harder to diagnose after the fact than here.
    """
    n = len(arrays["trajectories"])
    if n == 0:
        raise ValueError("No trials survived preprocessing.")

    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} rows, expected {n}")
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            bad = int((~np.isfinite(arr)).any(axis=tuple(range(1, arr.ndim))).sum())
            raise ValueError(f"{name} contains NaN/Inf in {bad} trials")

    T = config.NORMALISED_LENGTH
    if arrays["trajectories"].shape[1:] != (T, 3):
        raise ValueError(
            f"trajectories have shape {arrays['trajectories'].shape[1:]}, expected ({T}, 3)"
        )

    # Every trial starts at the origin by construction — a violation means
    # spatial normalisation was skipped somewhere.
    starts = np.abs(arrays["trajectories"][:, 0, :]).max()
    if starts > 1e-4:
        raise ValueError(f"trajectories are not origin-aligned (max |start| = {starts:.6f})")

    move_time = arrays["timing"][:, 0]
    if (move_time <= 0).any():
        raise ValueError(f"{int((move_time <= 0).sum())} trials have non-positive movement time")


# ── Timing quality control ────────────────────────────────────────────────────
def timing_plausible(timing: np.ndarray) -> np.ndarray:
    """
    Boolean mask of trials whose segmented timing fits the task constraints.

    A ballistic interception is over in well under a second; a much longer
    window means segmentation overran the interception rather than the
    participant being slow.
    """
    move_time, init_time = timing[:, 0], timing[:, 1]
    return (
        (move_time >= config.MIN_MOVEMENT_TIME_S)
        & (move_time <= config.MAX_MOVEMENT_TIME_S)
        & (init_time >= 0)
        & (init_time <= config.MAX_INITIATION_TIME_S)
    )


def report_timing_qc(timing: np.ndarray, subjects: np.ndarray, plausible: np.ndarray) -> None:
    """Print how much of the dataset the timing bounds would exclude, and where."""
    n_bad = int((~plausible).sum())
    print(f"\n  timing QC: {n_bad} / {len(timing)} trials ({100 * (~plausible).mean():.1f}%) "
          f"outside [{config.MIN_MOVEMENT_TIME_S}, {config.MAX_MOVEMENT_TIME_S}] s movement "
          f"or > {config.MAX_INITIATION_TIME_S} s initiation")
    if not n_bad:
        return

    for i, name in enumerate(config.TIMING_FEATURES):
        v = timing[:, i]
        print(f"    {name:18s}: median {np.median(v):.3f}  p99 {np.percentile(v, 99):.3f}  "
              f"max {v.max():.3f}  (s)")

    per_subject = {s: int((~plausible)[subjects == s].sum()) for s in np.unique(subjects)}
    worst = sorted(per_subject.items(), key=lambda kv: -kv[1])[:5]
    print("    most affected subjects: " + ", ".join(f"{s} ({n})" for s, n in worst if n))
    print("    flagged in dataset.npz['timing_plausible'] and metadata.csv; "
          "rebuild with --drop-implausible to exclude them")


# ── Splits ────────────────────────────────────────────────────────────────────
def build_splits(subjects: np.ndarray, seed: int) -> dict:
    """
    Leave-N-subjects-out assignment, recorded as both subject ids and trial
    indices into the ``dataset.npz`` arrays.

    Uses ``src.train.split_subject_ids``, the same function the training loop
    calls, so this file is a faithful record of the split rather than a
    reimplementation that can drift out of step.
    """
    train_ids, val_ids, test_ids = split_subject_ids(
        list(subjects),
        n_train=config.N_TRAIN,
        n_val=config.N_VAL,
        n_test=config.N_TEST,
        seed=seed,
    )

    splits = {
        "seed": seed,
        "n_train": config.N_TRAIN,
        "n_val": config.N_VAL,
        "n_test": config.N_TEST,
        "splits": {},
    }
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        idx = np.flatnonzero(np.isin(subjects, ids))
        splits["splits"][name] = {
            "subjects": sorted(ids),
            "n_subjects": len(ids),
            "n_trials": int(len(idx)),
            "indices": idx.tolist(),
        }

    assigned = sum(s["n_trials"] for s in splits["splits"].values())
    splits["n_trials_unassigned"] = int(len(subjects) - assigned)
    return splits


# ── Entry point ───────────────────────────────────────────────────────────────
def make_dataset(
    raw_dir: Path,
    out_dir: Path,
    seed: int = config.SEED,
    force: bool = False,
    drop_implausible: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials(raw_dir, out_dir / "trials.pkl", force=force)
    arrays = build_arrays(trials)
    validate(arrays)

    plausible = timing_plausible(arrays["timing"])
    report_timing_qc(arrays["timing"], arrays["subjects"], plausible)

    if drop_implausible:
        keep = np.flatnonzero(plausible)
        print(f"    dropping {len(plausible) - len(keep)} flagged trials")
        arrays = {k: v[keep] for k, v in arrays.items()}
        trials = [trials[i] for i in keep]
        plausible = plausible[keep]

    arrays["timing_plausible"] = plausible

    # ── dataset.npz ──
    npz_path = out_dir / "dataset.npz"
    np.savez_compressed(
        npz_path,
        **arrays,
        timing_features=np.array(config.TIMING_FEATURES),
        normalised_length=np.int32(config.NORMALISED_LENGTH),
        recording_hz=np.int32(config.RECORDING_HZ),
        condition=np.int32(config.CONDITION_FREE_EYE),
        dropped_implausible=np.bool_(drop_implausible),
        created_at=np.array(datetime.now().isoformat(timespec="seconds")),
    )

    # ── metadata.csv ──
    meta_df = extract_features_dataframe(trials)
    meta_df.insert(0, "index", np.arange(len(meta_df)))
    meta_df["timing_plausible"] = plausible
    meta_path = out_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    # ── splits.json ──
    splits = build_splits(arrays["subjects"], seed)
    splits_path = out_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    # ── Summary ──
    n = len(arrays["trajectories"])
    move_time = arrays["timing"][:, 0]
    init_time = arrays["timing"][:, 1]

    print("\n" + "=" * 60)
    print("DATASET BUILT")
    print("=" * 60)
    print(f"  {npz_path}   ({npz_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {meta_path}  ({len(meta_df)} rows x {len(meta_df.columns)} cols)")
    print(f"  {splits_path}")
    print(f"\n  trials              : {n}")
    print(f"  subjects            : {len(np.unique(arrays['subjects']))}")
    print(f"  trajectories        : {arrays['trajectories'].shape} float32")
    print(f"  timing              : {arrays['timing'].shape} {config.TIMING_FEATURES}")
    print(f"  conditions          : {arrays['conditions'].shape}")
    print(
        f"  movement time (s)   : mean {move_time.mean():.3f}  sd {move_time.std():.3f}  "
        f"range [{move_time.min():.3f}, {move_time.max():.3f}]"
    )
    print(
        f"  initiation time (s) : mean {init_time.mean():.3f}  sd {init_time.std():.3f}  "
        f"range [{init_time.min():.3f}, {init_time.max():.3f}]"
    )
    print(f"\n  split (seed {seed}):")
    for name, s in splits["splits"].items():
        print(f"    {name:5s}: {s['n_subjects']:2d} subjects, {s['n_trials']:5d} trials")
    if splits["n_trials_unassigned"]:
        print(f"    unassigned: {splits['n_trials_unassigned']} trials")

    return {"arrays": arrays, "metadata": meta_df, "splits": splits}


def main():
    parser = argparse.ArgumentParser(description="Build dataset.npz + metadata.csv + splits.json")
    parser.add_argument("--raw-dir", type=str, default=None, help="Raw data directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Seed for the subject split")
    parser.add_argument(
        "--force", action="store_true", help="Re-run preprocessing instead of using trials.pkl"
    )
    parser.add_argument(
        "--drop-implausible", action="store_true",
        help="Exclude trials whose segmented timing violates the task constraints",
    )
    args = parser.parse_args()

    make_dataset(
        raw_dir=Path(args.raw_dir) if args.raw_dir else config.DATA_RAW_DIR,
        out_dir=Path(args.out_dir) if args.out_dir else config.DATA_PROCESSED_DIR,
        seed=args.seed,
        force=args.force,
        drop_implausible=args.drop_implausible,
    )


if __name__ == "__main__":
    main()
