"""
Streamlit dashboard for the Interception Movements CVAE.

Two modes:
  1. Inference Mode  – upload new trajectory CSV, extract latent fingerprint
  2. Exploration Mode – manipulate latent sliders, generate trajectories

Run with:  streamlit run src/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.run_config import RunConfig, find_runs
from src.vae_model import ConditionalVAE, NormStats, encode_condition
from src.preprocessing import (
    lowpass_filter,
    find_stimulus_onset,
    find_movement_window,
    normalise_temporal,
    normalise_spatial,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def plot_3d_trajectory(pos: np.ndarray, title: str = "Trajectory"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=2)
    ax.scatter(*pos[0], color="green", s=80, label="Start", zorder=5)
    ax.scatter(*pos[-1], color="red", s=80, label="End", zorder=5)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)
    ax.legend()
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Interception Movements VAE", layout="wide")
    st.title("🎯 Interception Movements – CVAE Dashboard")

    # Sidebar: model loading — one checkpoint per run directory
    st.sidebar.header("Model")
    runs = find_runs()

    if not runs:
        st.warning(
            f"No trained runs found in {config.RUNS_DIR}. "
            "Train a model first with `python main.py --phase 3`."
        )
        return

    selected_run = st.sidebar.selectbox(
        "Select run",
        runs,
        format_func=lambda p: p.name,
        index=len(runs) - 1,
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

    # ── Inference Mode ────────────────────────────────────────────────────
    if mode == "Inference":
        st.header("📊 Inference Mode")
        st.write("Upload a raw trial CSV to extract its latent fingerprint.")

        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded is not None:
            import pandas as pd

            df = pd.read_csv(uploaded, header=None, names=config.CSV_COLUMNS)
            pos_raw = df[["x", "y", "z"]].values.astype(float)
            markers = df["marker"].values

            stim_idx = find_stimulus_onset(markers)
            if stim_idx is None:
                st.error("No stimulus onset marker (5) found in data.")
                return

            pos_filt = lowpass_filter(pos_raw)
            start, end = find_movement_window(pos_filt, stim_idx)
            movement = pos_filt[start : end + 1]

            if len(movement) < 4:
                st.error("Movement segment too short.")
                return

            pos_norm = normalise_spatial(normalise_temporal(movement))

            # Timing, in seconds, from the raw frame indices — the part that
            # temporal normalisation discards.
            fs = config.RECORDING_HZ
            movement_time_s = (end - start) / fs
            initiation_time_s = (start - stim_idx) / fs

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_3d_trajectory(pos_norm, "Processed Trajectory"))
                st.caption(
                    f"Movement time {movement_time_s * 1000:.0f} ms · "
                    f"Initiation time {initiation_time_s * 1000:.0f} ms"
                )

            # Encode
            sp = st.sidebar.number_input("Starting position (1-3)", 1, 3, 2)
            side = st.sidebar.number_input("Starting side (1=L, 2=R)", 1, 2, 1)
            cond = torch.tensor(encode_condition(sp, side)).unsqueeze(0)
            traj_flat = torch.tensor(pos_norm.flatten(), dtype=torch.float32).unsqueeze(0)
            traj_z = (traj_flat - tm) / ts

            timing_z = None
            if model.timing_dim:
                timing = torch.tensor(
                    [[movement_time_s, initiation_time_s]], dtype=torch.float32
                )
                timing_z = (timing - tim_m) / tim_s

            with torch.no_grad():
                mu, logvar = model.encode(traj_z, cond, timing_z)

            with col2:
                st.subheader("Latent Fingerprint")
                for i in range(latent_dim):
                    st.metric(f"z{i}", f"{mu[0, i].item():.4f}")

    # ── Exploration Mode ──────────────────────────────────────────────────
    else:
        st.header("🔬 Exploration Mode")
        st.write("Adjust latent variables to generate trajectories.")

        sp = st.sidebar.number_input("Starting position (1-3)", 1, 3, 2)
        side = st.sidebar.number_input("Starting side (1=L, 2=R)", 1, 2, 1)
        cond = torch.tensor(encode_condition(sp, side)).unsqueeze(0)

        z_values = []
        for i in range(latent_dim):
            val = st.slider(f"z{i}", -3.0, 3.0, 0.0, 0.1)
            z_values.append(val)

        z = torch.tensor([z_values], dtype=torch.float32)

        with torch.no_grad():
            recon_z, recon_timing_z = model.decode(z, cond)
            recon = (recon_z * ts + tm).numpy()[0]
            pos = recon.reshape(config.NORMALISED_LENGTH, 3)
            timing = (
                (recon_timing_z * tim_s + tim_m).numpy()[0]
                if recon_timing_z is not None
                else None
            )

        st.pyplot(plot_3d_trajectory(pos, "Generated Trajectory"))

        # The generated movement is a shape *and* a duration: without the timing
        # head the sliders would only ever produce a path, with no speed.
        if timing is not None:
            cols = st.columns(len(config.TIMING_FEATURES) + 1)
            for col, name, value in zip(cols, config.TIMING_FEATURES, timing):
                col.metric(name.replace("_s", "").replace("_", " ").title(), f"{value * 1000:.0f} ms")

            move_time = float(timing[config.TIMING_FEATURES.index("movement_time_s")])
            if move_time > 0:
                # Re-attaching the duration turns the normalised shape back into
                # a velocity profile in physical units.
                dt = move_time / (config.NORMALISED_LENGTH - 1)
                speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) / dt
                cols[-1].metric("Peak speed", f"{speed.max():.0f} mm/s")

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(np.linspace(0, move_time, config.NORMALISED_LENGTH) * 1000, speed)
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Speed (mm/s)")
                ax.set_title("Generated speed profile")
                st.pyplot(fig)


if __name__ == "__main__":
    main()
