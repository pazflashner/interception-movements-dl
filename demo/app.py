"""Presentation-only view of the existing trained trajectory decoders."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.confirmatory_dashboard import (
    ASSETS, MULTI_ASSETS, MODEL_LABELS, decode, load_selected_model,
    read_csv, read_json, movement_start_index,
)
from src.dashboard_models import reference_name


def trajectory_figure(path, movement, initiation):
    """Plot one decoded path; timing determines the displayed onset marker."""
    onset = movement_start_index(movement, initiation, len(path))
    fig, ax = plt.subplots(figsize=(6.2, 4.7), dpi=120)
    ax.plot(path[:, 0], path[:, 1], color="#087C83", lw=2.8)
    ax.scatter(*path[0], color="#2166DF", s=48, label="Window start", zorder=4)
    ax.scatter(*path[onset], color="#D78A20", s=55, label="Predicted movement onset", zorder=5)
    ax.scatter(*path[-1], color="#9B4267", s=48, label="Window end", zorder=4)
    ax.set(xlabel="Lateral position x (cm)", ylabel="Forward position y (cm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.16)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout(pad=1.3)
    return fig


def main():
    st.set_page_config(page_title="Interception demo", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    .block-container {padding-top:1.1rem; padding-bottom:.2rem; max-width:1480px;}
    h1 {font-size:1.8rem !important;} h3 {font-size:1.15rem !important;}
    [data-testid="stSidebar"] {min-width:330px; max-width:330px; background:#f4f7fb;}
    [data-testid="stSidebar"] .block-container {padding-top:1rem;}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.45rem;}
    [data-testid="stSidebar"] p {font-size:.85rem;}
    [data-testid="stMetric"] {padding:.8rem 1rem; background:#f4f7fb; border-radius:8px; margin-bottom:.45rem;}
    [data-testid="stMetricValue"] {font-size:2.2rem;}
    #MainMenu, footer, [data-testid="stToolbar"] {visibility:hidden;}
    </style>""", unsafe_allow_html=True)

    manifest = read_json(str(MULTI_ASSETS / "manifest.json"))
    latent_stats = read_json(str(MULTI_ASSETS / "latent_stats.json"))
    fingerprints = read_csv(str(MULTI_ASSETS / "subject_fingerprints.csv"))
    speed_ranges = read_csv(str(ASSETS / "condition_speed_ranges.csv"))

    with st.sidebar:
        st.subheader("Latent controls")
        families = list(manifest["models"])
        family = st.selectbox("Model", families, index=families.index(manifest["default_model"]),
                              format_func=lambda f: MODEL_LABELS[f])
        dim = st.segmented_control("Latent dimension", manifest["models"][family],
                                   default=8, key=f"dimension_{family}") or 8
        name = reference_name(family, int(dim))
        stats = latent_stats[name]
        available = fingerprints[fingerprints.run == name]
        source = st.selectbox("Fingerprint", ["Training-population center"] + sorted(available.subject.unique()))
        if source == "Training-population center":
            base = np.asarray(stats["training_center"], dtype=float)
        else:
            row = available[available.subject == source].iloc[0]
            base = row[[f"z{i+1}" for i in range(dim)]].to_numpy(float)
        st.caption("Offsets from the selected center, in training SD.")
        columns = st.columns(2)
        offsets = [columns[i % 2].slider(f"z{i+1}", -2.5, 2.5, 0.0, .1,
                    key=f"{name}_{source}_{i}") for i in range(dim)]
        latent = base + np.asarray(offsets) * np.asarray(stats["training_scale"])
        with st.expander("Task conditions", expanded=False):
            unconditional = family == "unconditional_vae"
            sp = st.selectbox("Start/speed category", [1, 2, 3], index=1, disabled=unconditional,
                              format_func=lambda s: {1:"120 / slow",2:"140 / medium",3:"160 / fast"}[s])
            side = 1 if st.radio("Starting side", ["Left", "Right"], horizontal=True,
                                disabled=unconditional) == "Left" else 2
            r = speed_ranges[speed_ranges.sp == sp].iloc[0]
            speed = st.slider("Executed target speed", float(r.speed_min), float(r.speed_max),
                              float(r.speed_median), disabled=unconditional)
            if unconditional:
                st.caption("VAE does not use task conditions.")
            elif family == "spline_pca":
                st.caption("Conditions affect timing only for spline + PCA.")

    model, norm = load_selected_model(family, int(dim))
    paths, timing = decode(model, norm, latent, int(sp), side, speed)
    movement, initiation = map(float, timing[0])  # Decoder order: movement, initiation.
    st.title("Interception movement explorer")
    st.caption(f"Demo · under construction  |  {MODEL_LABELS[family]} · n={dim}  |  Move a latent slider to update the output")
    trajectory, prediction = st.columns([2.15, 1], gap="large")
    with trajectory:
        st.subheader("Generated trajectory")
        fig = trajectory_figure(paths[0], movement, initiation)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with prediction:
        st.subheader("Predicted timing")
        st.metric("Initiation time", f"{initiation * 1000:.0f} ms")
        st.metric("Movement time", f"{movement * 1000:.0f} ms")
        st.metric("Total modeled time", f"{(movement + initiation) * 1000:.0f} ms")
        st.caption("Initiation: target motion to finger movement. Movement: finger movement to the window end.")


if __name__ == "__main__":
    main()
