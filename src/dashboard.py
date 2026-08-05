"""
Streamlit dashboard for the Interception Movements CVAE.

  • Inference  — upload a subject's trial CSV(s); each trial is encoded and the
    codes are aggregated into that subject's fingerprint (mean + spread).
  • Exploration — move the latent sliders to generate a movement.

Segmentation matches the training pipeline (go-signal → finger arrival); the
go-signal / arrival come from the paired trialinfo_*.mat when supplied, otherwise
from editable defaults, so a bare CSV still works.

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
    lowpass_filter, find_stimulus_onset, find_movement_onset,
    normalise_temporal, normalise_spatial,
)

FS = config.RECORDING_HZ
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, latent_dim: int):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ConditionalVAE(latent_dim=ckpt.get("latent_dim", latent_dim),
                           timing_dim=ckpt.get("timing_dim", 0))
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, NormStats.from_checkpoint(ckpt)


# ── Core logic (mirrors preprocessing.preprocess_trial, no filtering) ─────────
def segment_trajectory(pos_raw, markers, go_signal_s, arrival_idx=None):
    stim_idx = find_stimulus_onset(markers)
    if stim_idx is None:
        stim_idx = 0
    pos_filt = lowpass_filter(pos_raw)
    n = len(pos_filt)
    arrival_idx = n - 1 if arrival_idx is None else int(min(max(arrival_idx, 0), n - 1))
    if go_signal_s is None or (isinstance(go_signal_s, float) and np.isnan(go_signal_s)):
        go_idx = stim_idx
    else:
        go_idx = stim_idx + int(round(go_signal_s * FS))
    go_idx = int(min(max(go_idx, 0), n - 1))
    move_start = find_movement_onset(pos_filt, go_idx, arrival_idx)
    movement = pos_filt[move_start: arrival_idx + 1]
    if len(movement) < 4:
        return None
    pos_norm = normalise_spatial(normalise_temporal(movement))
    return pos_norm, (arrival_idx - move_start) / FS, (move_start - go_idx) / FS


def mat_meta_from_upload(buffer) -> dict:
    try:
        m = sio.loadmat(buffer, squeeze_me=True, struct_as_record=False)["thistrial"]
    except Exception:
        return {}
    st_, pt = _mat_scalar(m, "starttime"), _mat_scalar(m, "pressedTime")
    return {"go_signal_s": object_motion_onset_s(np.array(getattr(m, "dotArray", []))),
            "arrival_s": (pt - st_) if (pt is not None and st_ is not None) else None}


def encode_mu(model, norm, pos_norm, movement_time_s, wait_time_s, sp, side):
    tm, ts, tim_m, tim_s = norm.torch("cpu")
    cond = torch.tensor(encode_condition(sp, side)).unsqueeze(0)
    traj_z = (torch.tensor(pos_norm.flatten(), dtype=torch.float32).unsqueeze(0) - tm) / ts
    timing_z = None
    if model.timing_dim:
        timing_z = (torch.tensor([[movement_time_s, wait_time_s]], dtype=torch.float32) - tim_m) / tim_s
    with torch.no_grad():
        mu, _ = model.encode(traj_z, cond, timing_z)
    return mu.numpy()[0]


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_3d(pos, title="Trajectory"):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=2.2, color="#0277BD")
    ax.scatter(*pos[0], color="#2ca02c", s=90, label="start")
    ax.scatter(*pos[-1], color="#d62728", s=90, label="interception")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(title); ax.legend()
    return fig


def plot_latent_cloud(mus, title="Subject fingerprint"):
    fig, ax = plt.subplots(figsize=(6, 5))
    if mus.shape[1] == 1:
        ax.scatter(mus[:, 0], np.zeros(len(mus)), alpha=0.35, label="trials")
        ax.scatter(mus[:, 0].mean(), 0, color="#d62728", s=180, marker="X", label="fingerprint")
        ax.set_yticks([])
    else:
        ax.scatter(mus[:, 0], mus[:, 1], alpha=0.35, label="trials")
        ax.scatter(mus[:, 0].mean(), mus[:, 1].mean(), color="#d62728", s=200, marker="X",
                   label="fingerprint", zorder=5)
        ax.set_ylabel("z1")
    ax.set_xlabel("z0"); ax.set_title(title); ax.legend()
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Interception Movements VAE", page_icon="🎯", layout="wide")
    st.markdown("""<style>
        .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.25rem;}
        h1 {font-size: 1.9rem;} h2 {font-size: 1.3rem;}
        #MainMenu, footer {visibility: hidden;}
    </style>""", unsafe_allow_html=True)

    st.title("🎯 Interception Movements — CVAE Dashboard")
    st.caption("Encode real reaches into a low-dimensional fingerprint, or generate new movements "
               "from the latent space. Model trained on the free eye-movement condition (held-out subjects).")

    # ── Sidebar: model ──
    runs = find_runs()
    if not runs:
        st.warning(f"No trained runs found in {config.RUNS_DIR}. Train first: "
                   "`python main.py --phase 3`.")
        return
    st.sidebar.header("⚙️ Model")
    selected_run = st.sidebar.selectbox("Checkpoint (run)", runs,
                                        format_func=lambda p: p.name, index=len(runs) - 1)
    run_cfg = RunConfig.load(selected_run)
    latent_dim = run_cfg.latent_dim
    model, norm = load_model(str(selected_run / "checkpoint.pt"), latent_dim)
    st.sidebar.success(f"Loaded · fingerprint size **z = {latent_dim}** · seed {run_cfg.seed}")
    st.sidebar.caption("Tip: pick a **z=3** run for a small, interpretable fingerprint.")
    with st.sidebar.expander("Full run config"):
        st.code(yaml.safe_dump(run_cfg.to_dict(), sort_keys=False), language="yaml")

    tm, ts, tim_m, tim_s = norm.torch("cpu")
    mode = st.sidebar.radio("Mode", ["🔬 Exploration", "📊 Inference"], index=0)

    # ── Exploration ──
    if mode.endswith("Exploration"):
        st.subheader("🔬 Exploration — generate a movement from the latent space")
        st.write("Move the sliders to walk through the fingerprint space; the model generates the "
                 "corresponding 3-D reach and its timing in real time.")
        with st.expander("ℹ️ What do the sliders do?"):
            st.markdown(
                "Each latent variable is a *style knob*. In the analysis, all axes mostly control "
                "**timing and speed** (reaction time, movement time, peak speed) in different mixes — "
                "e.g. one axis trades a longer wait for a shorter, faster reach. See the report's "
                "correlation heatmap for the exact mapping.")

        left, right = st.columns([1, 1.4], gap="large")
        with left:
            st.markdown("**Task condition**")
            c1, c2 = st.columns(2)
            sp = c1.selectbox("Start position", [1, 2, 3], index=1)
            side = c2.selectbox("Side", [1, 2], index=0, format_func=lambda s: "Left" if s == 1 else "Right")
            st.markdown("**Latent variables**")
            z_values = [st.slider(f"z{i}", -3.0, 3.0, 0.0, 0.1) for i in range(latent_dim)]
            if st.button("↺ Reset to centre"):
                st.rerun()

        cond = torch.tensor(encode_condition(sp, side)).unsqueeze(0)
        z = torch.tensor([z_values], dtype=torch.float32)
        with torch.no_grad():
            recon_z, recon_timing_z = model.decode(z, cond)
            pos = (recon_z * ts + tm).numpy()[0].reshape(config.NORMALISED_LENGTH, 3)
            timing = (recon_timing_z * tim_s + tim_m).numpy()[0] if recon_timing_z is not None else None

        with right:
            st.pyplot(plot_3d(pos, "Generated reach"))

        if timing is not None:
            st.markdown("**Generated movement properties**")
            move_time = float(timing[config.TIMING_FEATURES.index("movement_time_s")])
            cols = st.columns(3)
            for col, name, value in zip(cols, config.TIMING_FEATURES, timing):
                col.metric(name.replace("_s", "").replace("_", " ").title(), f"{value*1000:.0f} ms")
            if move_time > 0:
                dt = move_time / (config.NORMALISED_LENGTH - 1)
                speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) / dt
                cols[2].metric("Peak speed", f"{speed.max():.0f} units/s")
                fig, ax = plt.subplots(figsize=(9, 2.6))
                ax.plot(np.linspace(0, move_time, config.NORMALISED_LENGTH) * 1000, speed, color="#0277BD")
                ax.fill_between(np.linspace(0, move_time, config.NORMALISED_LENGTH) * 1000, speed, alpha=0.15, color="#0277BD")
                ax.set_xlabel("Time (ms)"); ax.set_ylabel("Speed (units/s)"); ax.set_title("Generated speed profile")
                st.pyplot(fig)

    # ── Inference ──
    else:
        st.subheader("📊 Inference — read out a subject's fingerprint")
        st.write("Upload **all of a subject's trial CSVs** (`li_2_*.csv` from a subject folder) to build "
                 "their fingerprint — the mean latent code plus its trial-to-trial spread. One file gives "
                 "a single trial's code.")
        import pandas as pd

        st.sidebar.divider()
        st.sidebar.subheader("📄 Segmentation metadata")
        src_choice = st.sidebar.radio("Go-signal & arrival from", ["Manual defaults", ".mat files"])
        default_go = st.sidebar.slider("Fallback go-signal (s after target appears)", 0.0, 1.0, 0.30, 0.01)
        mats = {}
        if src_choice == ".mat files":
            for mf in st.sidebar.file_uploader("trialinfo_*.mat (optional, matched by name)",
                                               type="mat", accept_multiple_files=True) or []:
                mats[Path(mf.name).stem.replace("trialinfo_", "")] = mat_meta_from_upload(mf)
        sp_d = st.sidebar.selectbox("Fallback start position", [1, 2, 3], index=1)
        side_d = st.sidebar.selectbox("Fallback side", [1, 2], index=0,
                                      format_func=lambda s: "Left" if s == 1 else "Right")

        uploaded = st.file_uploader("Trial CSV(s)", type="csv", accept_multiple_files=True)
        if not uploaded:
            st.info("⬆️ Upload one or more `li_2_*.csv` files to begin. "
                    "Tip: grab ~15–20 from a single `subjectNN` folder to see a fingerprint cloud form.")
            return

        mus, per_trial, skipped = [], [], 0
        for uf in uploaded:
            df = pd.read_csv(uf, header=None, names=config.CSV_COLUMNS)
            meta = parse_filename(uf.name) or {}
            tag = Path(uf.name).stem.replace("li_", "")
            go_s = mats.get(tag, {}).get("go_signal_s")
            if go_s is None or (isinstance(go_s, float) and np.isnan(go_s)):
                go_s = default_go
            seg = segment_trajectory(df[["x", "y", "z"]].values.astype(float), df["marker"].values, go_s)
            if seg is None:
                skipped += 1; continue
            pos_norm, mt, wt = seg
            mus.append(encode_mu(model, norm, pos_norm, mt, wt,
                                 meta.get("sp", sp_d), meta.get("side", side_d)))
            per_trial.append((uf.name, pos_norm, mt, wt))
        if not mus:
            st.error("No usable trials in the upload."); return
        mus = np.stack(mus)

        st.success(f"Encoded **{len(mus)}** trial(s)" + (f" ({skipped} skipped as too short)" if skipped else ""))
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.pyplot(plot_3d(per_trial[0][1], f"Example trial · {per_trial[0][0]}"))
            st.caption(f"movement {per_trial[0][2]*1000:.0f} ms · wait {per_trial[0][3]*1000:.0f} ms")
        with c2:
            if len(mus) > 1:
                st.pyplot(plot_latent_cloud(mus, "Trials (blue) and fingerprint (red ✕)"))
            else:
                st.info("Upload more than one trial to see the fingerprint cloud.")

        st.markdown("**Fingerprint** — mean ± spread across the uploaded trials")
        cols = st.columns(latent_dim)
        for i in range(latent_dim):
            spread = mus[:, i].std() if len(mus) > 1 else 0.0
            cols[i].metric(f"z{i}", f"{mus[:, i].mean():.2f}", f"± {spread:.2f}", delta_color="off")


if __name__ == "__main__":
    main()
