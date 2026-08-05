"""
Streamlit dashboard for the Interception Movements CVAE.

Two modes:
  1. Inference Mode   - upload a subject's trial CSV(s); extract each trial's
     latent code and aggregate them into that subject's fingerprint (mean +
     spread). A single trial is the one-file special case.
  2. Exploration Mode - move the latent sliders to generate trajectories.

Segmentation matches the training pipeline (event-based: object-motion onset ->
finger arrival). The go-signal and arrival come from the paired trialinfo_*.mat
when supplied; otherwise they fall back to editable defaults, so a bare CSV still
works.

Run with:  streamlit run src/dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.run_config import RunConfig, find_runs
from src.vae_model import ConditionalVAE, NormStats, encode_condition
from src.data_loading import parse_filename, _mat_scalar, object_motion_onset_s
from src.preprocessing import (
    lowpass_filter,
    find_stimulus_onset,
    find_movement_onset,
    normalise_temporal,
    normalise_spatial,
)

FS = config.RECORDING_HZ


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, latent_dim: int):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ConditionalVAE(
        latent_dim=ckpt.get("latent_dim", latent_dim),
        timing_dim=ckpt.get("timing_dim", 0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, NormStats.from_checkpoint(ckpt)


# ── Segmentation (mirrors preprocessing.preprocess_trial, no filtering) ───────
def segment_trajectory(pos_raw, markers, go_signal_s, arrival_idx=None):
    """
    Event-based segmentation for one uploaded trial. Returns
    (pos_norm (100,3), movement_time_s, wait_time_s, idx dict) or None if short.
    """
    stim_idx = find_stimulus_onset(markers)
    if stim_idx is None:
        stim_idx = 0
    pos_filt = lowpass_filter(pos_raw)
    n = len(pos_filt)
    if arrival_idx is None:
        arrival_idx = n - 1
    arrival_idx = int(min(max(arrival_idx, 0), n - 1))

    if go_signal_s is None or (isinstance(go_signal_s, float) and np.isnan(go_signal_s)):
        go_idx = stim_idx
    else:
        go_idx = stim_idx + int(round(go_signal_s * FS))
    go_idx = int(min(max(go_idx, 0), n - 1))

    move_start = find_movement_onset(pos_filt, go_idx, arrival_idx)
    movement = pos_filt[move_start : arrival_idx + 1]
    if len(movement) < 4:
        return None

    pos_norm = normalise_spatial(normalise_temporal(movement))
    wait_time_s = (move_start - go_idx) / FS
    movement_time_s = (arrival_idx - move_start) / FS
    return pos_norm, movement_time_s, wait_time_s, {
        "stim": stim_idx, "go": go_idx, "onset": move_start, "arrival": arrival_idx,
    }


def mat_meta_from_upload(buffer) -> dict:
    """go_signal_s / arrival_s / responseText from an uploaded trialinfo_*.mat."""
    try:
        m = sio.loadmat(buffer, squeeze_me=True, struct_as_record=False)["thistrial"]
    except Exception:
        return {}
    start_t = _mat_scalar(m, "starttime")
    pressed_t = _mat_scalar(m, "pressedTime")
    arrival_s = (pressed_t - start_t) if (pressed_t is not None and start_t is not None) else None
    go_s = object_motion_onset_s(np.array(getattr(m, "dotArray", [])))
    return {"go_signal_s": go_s, "arrival_s": arrival_s,
            "responseText": str(getattr(m, "responseText", ""))}


def encode_mu(model, norm, pos_norm, movement_time_s, wait_time_s, sp, side):
    """Encode one processed trial to its latent mean mu (1, latent_dim)."""
    tm, ts, tim_m, tim_s = norm.torch("cpu")
    cond = torch.tensor(encode_condition(sp, side)).unsqueeze(0)
    traj_z = (torch.tensor(pos_norm.flatten(), dtype=torch.float32).unsqueeze(0) - tm) / ts
    timing_z = None
    if model.timing_dim:
        timing = torch.tensor([[movement_time_s, wait_time_s]], dtype=torch.float32)
        timing_z = (timing - tim_m) / tim_s
    with torch.no_grad():
        mu, _ = model.encode(traj_z, cond, timing_z)
    return mu.numpy()[0]


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_3d_trajectory(pos: np.ndarray, title: str = "Trajectory"):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=2)
    ax.scatter(*pos[0], color="green", s=80, label="Start", zorder=5)
    ax.scatter(*pos[-1], color="red", s=80, label="End", zorder=5)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title); ax.legend()
    return fig


def plot_latent_cloud(mus: np.ndarray, title: str = "Subject fingerprint"):
    """Scatter each trial's latent code with the subject mean marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if mus.shape[1] == 1:
        ax.scatter(mus[:, 0], np.zeros(len(mus)), alpha=0.4, label="trials")
        ax.scatter(mus[:, 0].mean(), 0, color="red", s=140, marker="X", label="fingerprint")
        ax.set_xlabel("z0"); ax.set_yticks([])
    else:
        ax.scatter(mus[:, 0], mus[:, 1], alpha=0.4, label="trials")
        ax.scatter(mus[:, 0].mean(), mus[:, 1].mean(), color="red", s=160,
                   marker="X", label="fingerprint", zorder=5)
        ax.set_xlabel("z0"); ax.set_ylabel("z1")
    ax.set_title(title); ax.legend()
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Interception Movements VAE", layout="wide")
    st.title("🎯 Interception Movements – CVAE Dashboard")

    st.sidebar.header("Model")
    runs = find_runs()
    if not runs:
        st.warning(
            f"No trained runs found in {config.RUNS_DIR}. "
            "Train a model first with `python main.py --phase 3`."
        )
        return

    selected_run = st.sidebar.selectbox(
        "Select run", runs, format_func=lambda p: p.name, index=len(runs) - 1
    )
    run_cfg = RunConfig.load(selected_run)
    latent_dim = run_cfg.latent_dim
    model, norm = load_model(str(selected_run / "checkpoint.pt"), latent_dim)
    st.sidebar.caption(
        f"seed {run_cfg.seed} · z={run_cfg.latent_dim} · lr={run_cfg.lr} · β={run_cfg.kl_weight}"
    )
    with st.sidebar.expander("Run config"):
        st.code(yaml.safe_dump(run_cfg.to_dict(), sort_keys=False), language="yaml")

    tm, ts, tim_m, tim_s = norm.torch("cpu")
    mode = st.sidebar.radio("Mode", ["Inference", "Exploration"])

    # ── Inference Mode ────────────────────────────────────────────────────────
    if mode == "Inference":
        st.header("📊 Inference – subject fingerprint")
        st.write(
            "Upload **all of a subject's trial CSVs** to build their fingerprint "
            "(mean latent code + spread). A single file gives one trial's code."
        )
        import pandas as pd

        uploaded = st.file_uploader("Trial CSV(s)", type="csv", accept_multiple_files=True)

        st.sidebar.subheader("Segmentation metadata")
        src_choice = st.sidebar.radio(
            "Go-signal & arrival from", [".mat files (default)", "Manual defaults"]
        )
        default_go = st.sidebar.number_input(
            "Fallback go-signal (s after object appears)", 0.0, 1.0, 0.30, 0.01
        )
        mats = {}
        if src_choice.startswith(".mat"):
            mat_files = st.sidebar.file_uploader(
                "Matching trialinfo_*.mat (optional)", type="mat", accept_multiple_files=True
            )
            for mf in mat_files or []:
                tag = Path(mf.name).stem.replace("trialinfo_", "")
                mats[tag] = mat_meta_from_upload(mf)

        # Condition fallback when the filename is not li_<c>_<sp>_<side>_<rep>
        sp_default = st.sidebar.number_input("Fallback start position (1-3)", 1, 3, 2)
        side_default = st.sidebar.number_input("Fallback side (1=L, 2=R)", 1, 2, 1)

        if uploaded:
            mus, per_trial = [], []
            for uf in uploaded:
                df = pd.read_csv(uf, header=None, names=config.CSV_COLUMNS)
                pos_raw = df[["x", "y", "z"]].values.astype(float)
                markers = df["marker"].values

                # metadata: filename → sp/side; .mat (if matched) → go-signal/arrival
                stem = Path(uf.name).stem
                meta = parse_filename(uf.name) or {}
                sp = meta.get("sp", sp_default)
                side = meta.get("side", side_default)

                tag = stem.replace("li_", "")
                mm = mats.get(tag, {})
                go_s = mm.get("go_signal_s")
                if go_s is None or (isinstance(go_s, float) and np.isnan(go_s)):
                    go_s = default_go
                arrival_idx = None  # default = CSV end (arrival)

                seg = segment_trajectory(pos_raw, markers, go_s, arrival_idx)
                if seg is None:
                    st.warning(f"{uf.name}: movement segment too short, skipped.")
                    continue
                pos_norm, mt, wt, idx = seg
                mu = encode_mu(model, norm, pos_norm, mt, wt, sp, side)
                mus.append(mu)
                per_trial.append((uf.name, pos_norm, mt, wt))

            if not mus:
                st.error("No usable trials.")
                return
            mus = np.stack(mus)

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_3d_trajectory(per_trial[0][1], f"Trajectory · {per_trial[0][0]}"))
                st.caption(
                    f"{len(mus)} trial(s) · movement {per_trial[0][2]*1000:.0f} ms · "
                    f"wait {per_trial[0][3]*1000:.0f} ms"
                )
            with c2:
                if len(mus) > 1:
                    st.pyplot(plot_latent_cloud(mus, "Subject fingerprint (latent space)"))

            st.subheader("Fingerprint (mean ± spread across trials)")
            cols = st.columns(latent_dim)
            for i in range(latent_dim):
                spread = mus[:, i].std() if len(mus) > 1 else 0.0
                cols[i].metric(f"z{i}", f"{mus[:, i].mean():.3f}", f"± {spread:.3f}")

    # ── Exploration Mode ──────────────────────────────────────────────────────
    else:
        st.header("🔬 Exploration – generate from the latent space")
        st.write("Adjust the latent variables to generate a movement.")

        sp = st.sidebar.number_input("Starting position (1-3)", 1, 3, 2)
        side = st.sidebar.number_input("Starting side (1=L, 2=R)", 1, 2, 1)
        cond = torch.tensor(encode_condition(sp, side)).unsqueeze(0)

        z_values = [st.slider(f"z{i}", -3.0, 3.0, 0.0, 0.1) for i in range(latent_dim)]
        z = torch.tensor([z_values], dtype=torch.float32)

        with torch.no_grad():
            recon_z, recon_timing_z = model.decode(z, cond)
            pos = (recon_z * ts + tm).numpy()[0].reshape(config.NORMALISED_LENGTH, 3)
            timing = (
                (recon_timing_z * tim_s + tim_m).numpy()[0]
                if recon_timing_z is not None else None
            )

        st.pyplot(plot_3d_trajectory(pos, "Generated Trajectory"))

        if timing is not None:
            cols = st.columns(len(config.TIMING_FEATURES) + 1)
            for col, name, value in zip(cols, config.TIMING_FEATURES, timing):
                col.metric(name.replace("_s", "").replace("_", " ").title(), f"{value*1000:.0f} ms")

            move_time = float(timing[config.TIMING_FEATURES.index("movement_time_s")])
            if move_time > 0:
                dt = move_time / (config.NORMALISED_LENGTH - 1)
                speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) / dt
                cols[-1].metric("Peak speed", f"{speed.max():.0f} units/s")
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(np.linspace(0, move_time, config.NORMALISED_LENGTH) * 1000, speed)
                ax.set_xlabel("Time (ms)"); ax.set_ylabel("Speed (units/s)")
                ax.set_title("Generated speed profile")
                st.pyplot(fig)


if __name__ == "__main__":
    main()
