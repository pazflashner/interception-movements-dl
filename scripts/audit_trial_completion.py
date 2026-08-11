"""Audit condition-2 trial completion and premature termination rules.

This is deliberately independent of the model preprocessing. It inspects every
paired CSV/MAT trial and records enough evidence to distinguish:

* a completed interception (MAT ``pressedTime`` is present),
* a recording that ended before the target started moving,
* a valid but early arrival (``responseText == 'Too early'``), and
* an endpoint that is unusual relative to the same participant's trials.

No trial is removed by this script. The output is an audit table used to define
and justify the preprocessing policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.data_loading import load_trial_csv, load_trial_metadata, parse_filename
from src.preprocessing import lowpass_filter, regularise_frame_grid


def robust_endpoint_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag endpoint outliers relative to each participant's endpoint cluster."""
    frame = frame.copy()
    frame["endpoint_distance_from_subject_median"] = np.nan
    frame["endpoint_cluster_outlier"] = False
    for _, indices in frame.groupby("subject").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        values = frame.loc[idx, ["end_x", "end_y"]].to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1)
        if finite.sum() < 5:
            continue
        center = np.median(values[finite], axis=0)
        distances = np.linalg.norm(values - center, axis=1)
        median_distance = float(np.median(distances[finite]))
        mad = float(np.median(np.abs(distances[finite] - median_distance)))
        # The absolute floor prevents a near-zero MAD from flagging harmless
        # tracker jitter. Position units remain unlabelled pending confirmation.
        threshold = max(median_distance + 6.0 * 1.4826 * mad, 1.0)
        frame.loc[idx, "endpoint_distance_from_subject_median"] = distances
        frame.loc[idx, "endpoint_cluster_outlier"] = distances > threshold
    return frame


def inspect_trial(csv_path: Path) -> dict:
    meta = parse_filename(csv_path.name)
    if meta is None:
        raise ValueError(f"unexpected trial filename: {csv_path.name}")
    suffix = csv_path.stem[len("li_") :]
    mat_path = csv_path.parent / f"trialinfo_{suffix}.mat"
    tmeta = load_trial_metadata(mat_path) if mat_path.exists() else {}

    row = {
        "subject": csv_path.parent.name,
        "trial_id": f"{csv_path.parent.name}_{csv_path.stem}",
        "csv_name": csv_path.name,
        "mat_exists": mat_path.exists(),
        **meta,
        **tmeta,
    }
    row["has_arrival"] = pd.notna(row.get("arrival_s"))
    row["has_go_signal"] = pd.notna(row.get("go_signal_s"))
    row["arrival_before_or_at_go"] = bool(
        row["has_arrival"]
        and row["has_go_signal"]
        and float(row["arrival_s"]) <= float(row["go_signal_s"])
    )
    row["arrival_after_go_s"] = (
        float(row["arrival_s"]) - float(row["go_signal_s"])
        if row["has_arrival"] and row["has_go_signal"]
        else np.nan
    )
    row["late_after_window_s"] = (
        float(row["arrival_s"]) - float(row["arrival_window_end_s"])
        if row["has_arrival"] and pd.notna(row.get("arrival_window_end_s"))
        else np.nan
    )

    try:
        df = load_trial_csv(csv_path)
        pos, _, frames, quality = regularise_frame_grid(df)
        filtered = lowpass_filter(pos)
        duration_s = float((frames[-1] - frames[0]) / config.RECORDING_HZ)
        go_idx = (
            int(round(float(row["go_signal_s"]) * config.RECORDING_HZ))
            if row["has_go_signal"]
            else 0
        )
        go_idx = int(np.clip(go_idx, 0, len(filtered) - 1))
        delta = filtered[-1, :2] - filtered[go_idx, :2]
        row.update(
            {
                "csv_duration_s": duration_s,
                "csv_ends_before_or_at_go": duration_s <= float(row["go_signal_s"])
                if row["has_go_signal"]
                else False,
                "go_idx": go_idx,
                "n_regular_frames": len(filtered),
                "n_interpolated_frames": quality["n_interpolated_frames"],
                "start_x": float(filtered[0, 0]),
                "start_y": float(filtered[0, 1]),
                "go_x": float(filtered[go_idx, 0]),
                "go_y": float(filtered[go_idx, 1]),
                "end_x": float(filtered[-1, 0]),
                "end_y": float(filtered[-1, 1]),
                "go_to_end_lateral": float(delta[0]),
                "go_to_end_forward": float(delta[1]),
                "go_to_end_distance": float(np.linalg.norm(delta)),
            }
        )
    except Exception as exc:
        row["csv_error"] = str(exc)
        for name in (
            "csv_duration_s", "go_idx", "n_regular_frames", "n_interpolated_frames",
            "start_x", "start_y", "go_x", "go_y", "end_x", "end_y",
            "go_to_end_lateral", "go_to_end_forward", "go_to_end_distance",
        ):
            row[name] = np.nan
        row["csv_ends_before_or_at_go"] = False
    return row


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    labels = frame.groupby(["responseText", "successful"], dropna=False).size().sort_values(ascending=False)
    too_early = frame[frame.responseText.eq("Too early")]
    suspicious = frame[
        frame.arrival_before_or_at_go
        | frame.csv_ends_before_or_at_go
        | ~frame.has_arrival
        | frame.endpoint_cluster_outlier
    ]
    lines = [
        "# Condition-2 trial completion audit",
        "",
        f"Trials inspected: {len(frame):,}",
        f"Participants: {frame.subject.nunique()}",
        f"MAT arrival present: {int(frame.has_arrival.sum()):,}/{len(frame):,}",
        f"Arrival at/before target motion onset: {int(frame.arrival_before_or_at_go.sum()):,}",
        f"CSV recording ends at/before target motion onset: {int(frame.csv_ends_before_or_at_go.sum()):,}",
        f"Endpoint-cluster outliers: {int(frame.endpoint_cluster_outlier.sum()):,}",
        f"Too early labels: {len(too_early):,}",
        f"Too early labels with arrival at/before go: {int(too_early.arrival_before_or_at_go.sum()):,}",
        f"Too early labels with CSV ending at/before go: {int(too_early.csv_ends_before_or_at_go.sum()):,}",
        "",
        "## Exact outcome labels",
        "",
        "| responseText | successful | count |",
        "|---|---:|---:|",
    ]
    for (label, code), count in labels.items():
        lines.append(f"| {label} | {code} | {count} |")
    lines += [
        "",
        "## Timing and endpoint summaries",
        "",
        "```text",
        frame[[
            "arrival_after_go_s", "late_after_window_s", "csv_duration_s",
            "go_to_end_forward", "go_to_end_distance",
            "endpoint_distance_from_subject_median",
        ]].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string(),
        "```",
        "",
        "## Flagged cases",
        "",
        f"The machine-readable table contains {len(suspicious):,} rows matching at least one audit flag.",
        "No audit flag is an automatic exclusion until its event semantics are justified.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path(r"D:\DropBox\Dropbox\results"))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "studies" / "strategy_window_comparison" / "data_audit",
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.raw.glob("subject*/li_2_*.csv"))
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        inspected = pool.map(inspect_trial, paths)
        for index, row in enumerate(inspected, 1):
            rows.append(row)
            if index % 500 == 0:
                print(f"Inspected {index:,}/{len(paths):,}", flush=True)
    frame = robust_endpoint_flags(pd.DataFrame(rows))
    frame.to_csv(args.out / "condition2_trial_completion_audit.csv", index=False)
    write_summary(frame, args.out / "TRIAL_COMPLETION_AUDIT.md")
    print((args.out / "TRIAL_COMPLETION_AUDIT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
