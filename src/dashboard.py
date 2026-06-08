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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.vae_model import ConditionalVAE, encode_condition
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
    model = ConditionalVAE(latent_dim=ckpt.get("latent_dim", latent_dim))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    train_mean = np.array(ckpt["train_mean"], dtype=np.float32)
    train_std = np.array(ckpt["train_std"], dtype=np.float32)
    return model, train_mean, train_std


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

    # Sidebar: model loading
    st.sidebar.header("Model")
    model_dir = Path(config.MODELS_DIR)
    model_files = sorted(model_dir.glob("cvae_z*_best.pt"))

    if not model_files:
        st.warning("No trained models found. Train a model first with `python main.py`.")
        return

    selected = st.sidebar.selectbox(
        "Select model",
        model_files,
        format_func=lambda p: p.stem,
    )
    latent_dim = int(selected.stem.split("z")[1].split("_")[0])
    model, train_mean, train_std = load_model(str(selected), latent_dim)

    tm = torch.tensor(train_mean)
    ts = torch.tensor(train_std)

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

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_3d_trajectory(pos_norm, "Processed Trajectory"))

            # Encode
            sp = st.sidebar.number_input("Starting position (1-3)", 1, 3, 2)
            side = st.sidebar.number_input("Starting side (1=L, 2=R)", 1, 2, 1)
            cond = torch.tensor(encode_condition(sp, side)).unsqueeze(0)
            traj_flat = torch.tensor(pos_norm.flatten(), dtype=torch.float32).unsqueeze(0)
            traj_z = (traj_flat - tm) / ts

            with torch.no_grad():
                mu, logvar = model.encode(traj_z, cond)

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
            recon_z = model.decode(z, cond)
            recon = (recon_z * ts + tm).numpy()[0]
            pos = recon.reshape(config.NORMALISED_LENGTH, 3)

        st.pyplot(plot_3d_trajectory(pos, "Generated Trajectory"))


if __name__ == "__main__":
    main()
