"""Interactive dashboard for the final 2-D interception-movement CVAEs.

Run from the repository root with:
    python -m streamlit run src/dashboard.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.features import features_from_arrays
from src.submovements import (
    SubmovementConfig,
    decompose_normalized_trajectory,
    minimum_jerk_velocity,
)
from src.vae_model import ConditionalVAE, NormStats, encode_trial_condition


MODELS_DIR = ROOT / "results" / "final_study" / "core_models"
GENERATION_DIR = ROOT / "results" / "final_study" / "generation"
ASSETS_DIR = ROOT / "results" / "final_study" / "dashboard"
REPORT_PATH = ROOT / "output" / "final_report" / "Interception_Movement_Fingerprints_Final.pdf"

COLORS = {
    "navy": "#17324D",
    "teal": "#167C80",
    "orange": "#D97728",
    "green": "#2E7D32",
    "red": "#B23A48",
    "gray": "#687684",
    "light": "#EEF3F6",
}

OUTPUTS = {
    "Movement duration": "movement_time_s",
    "Initiation time": "initiation_time_s",
    "Peak speed": "peak_speed_tracker_units_s",
    "Path length": "path_length",
    "Curvature index": "curvature_index",
    "Maximum lateral deviation": "max_lateral_deviation",
    "Minimum-jerk fit error": "mj_fit_error",
    "First component duration": "mj_first_duration_s",
    "First component amplitude": "mj_first_amplitude",
    "Secondary amplitude fraction": "mj_secondary_amplitude_fraction",
    "Mean component overlap": "mj_mean_overlap_pct",
    "Component count": "mj_n_components",
}

UNITS = {
    "movement_time_s": "s",
    "initiation_time_s": "s",
    "peak_speed_tracker_units_s": "tracker units/s",
    "path_length": "tracker units",
    "curvature_index": "ratio",
    "max_lateral_deviation": "tracker units",
    "mj_fit_error": "normalized error",
    "mj_first_duration_s": "s",
    "mj_first_amplitude": "tracker units",
    "mj_secondary_amplitude_fraction": "fraction",
    "mj_mean_overlap_pct": "%",
    "mj_n_components": "components",
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1420px; padding-top: 1.35rem; padding-bottom: 2rem;}
        h1 {font-size: 1.78rem !important; letter-spacing: 0 !important; margin-bottom: .2rem;}
        h2 {font-size: 1.25rem !important; letter-spacing: 0 !important;}
        h3 {font-size: 1.02rem !important; letter-spacing: 0 !important;}
        [data-testid="stMetric"] {background: #f6f8fa; border: 1px solid #d8e0e6; border-radius: 6px; padding: .75rem;}
        [data-testid="stMetricValue"] {font-size: 1.28rem; color: #17324D;}
        [data-testid="stSidebar"] {background: #f4f7f8; border-right: 1px solid #d8e0e6;}
        .stTabs [data-baseweb="tab-list"] {gap: 1.35rem; border-bottom: 1px solid #d8e0e6;}
        .stTabs [data-baseweb="tab"] {height: 2.8rem; padding-left: 0; padding-right: 0;}
        div[data-testid="stExpander"] {border-radius: 6px;}
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stStatusWidget"] {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource
def load_model(run_name: str) -> tuple[ConditionalVAE, NormStats]:
    checkpoint = torch.load(
        MODELS_DIR / run_name / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    cfg = checkpoint["config"]
    model_cfg = cfg.get("model", cfg)
    model = ConditionalVAE(
        input_dim=int(checkpoint.get("input_dim", len(checkpoint["train_mean"]))),
        condition_dim=int(checkpoint.get("condition_dim", 4)),
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        timing_dim=int(checkpoint["timing_dim"]),
        encoder_uses_timing=bool(checkpoint.get("encoder_uses_timing", True)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, NormStats.from_checkpoint(checkpoint)


def run_name(latent_dim: int, seed: int) -> str:
    return f"trajectory_only_z{latent_dim}_seed{seed}"


def decode(
    model: ConditionalVAE,
    norm: NormStats,
    latent: np.ndarray,
    sp: int,
    side: int,
    target_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    latent = np.atleast_2d(np.asarray(latent, dtype=np.float32))
    metadata = {
        "sp": int(sp),
        "side": int(side),
        "target_speed_screen_s": float(target_speed),
    }
    condition = encode_trial_condition(metadata, model.condition_dim)
    condition = np.repeat(condition[None, :], len(latent), axis=0)
    train_mean, train_std, _, _ = norm.torch("cpu")
    with torch.no_grad():
        trajectory_z, timing_z = model.decode(
            torch.as_tensor(latent), torch.as_tensor(condition)
        )
    channels = model.input_dim // config.NORMALISED_LENGTH
    trajectory = ((trajectory_z * train_std + train_mean).numpy()).reshape(
        len(latent), config.NORMALISED_LENGTH, channels
    )
    timing = norm.denormalise_timing(timing_z.numpy())
    timing[:, 0] = np.maximum(timing[:, 0], 1e-3)
    timing[:, 1] = np.maximum(timing[:, 1], 0.0)
    return trajectory, timing


@st.cache_data(show_spinner=False)
def quick_submovement_fit(
    trajectory_bytes: bytes, shape: tuple[int, int], movement_time_s: float
) -> dict:
    trajectory = np.frombuffer(trajectory_bytes, dtype=np.float64).reshape(shape)
    cfg = SubmovementConfig(restarts=1, max_nfev=300)
    result = decompose_normalized_trajectory(
        trajectory, movement_time_s, cfg, "dashboard-generated"
    )
    return {
        "summary": result.summary(),
        "parameters": result.selected.parameters,
        "time": result.selected.time,
        "reconstructed_velocity": result.selected.reconstructed_velocity,
    }


def basic_features(trajectories: np.ndarray, timing: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            features_from_arrays(trajectories[i], timing[i, 0], timing[i, 1])
            for i in range(len(trajectories))
        ]
    )


def plot_trajectory(trajectory: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    phase = np.linspace(0, 1, len(trajectory))
    points = ax.scatter(
        trajectory[:, 0], trajectory[:, 1], c=phase, cmap="viridis", s=15, zorder=2
    )
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=COLORS["navy"], lw=1.5, alpha=.75)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], s=75, color=COLORS["green"], label="start")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=75, color=COLORS["red"], label="arrival")
    ax.set_xlabel("Lateral x (tracker units)")
    ax.set_ylabel("Forward y (tracker units)")
    ax.set_title("Generated table-plane trajectory")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.colorbar(points, ax=ax, label="Movement phase")
    fig.tight_layout()
    return fig


def velocity_arrays(trajectory: np.ndarray, movement_time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0, movement_time_s, len(trajectory))
    velocity = np.gradient(trajectory, time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return time, velocity, speed


def plot_velocity(trajectory: np.ndarray, movement_time_s: float) -> plt.Figure:
    time, velocity, speed = velocity_arrays(trajectory, movement_time_s)
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(time, speed, color=COLORS["navy"], lw=2.4, label="speed")
    ax.plot(time, velocity[:, 1], color=COLORS["teal"], lw=1.5, label="forward velocity")
    ax.plot(time, velocity[:, 0], color=COLORS["orange"], lw=1.5, label="lateral velocity")
    ax.axhline(0, color="#777777", lw=.7)
    ax.set_xlabel("Movement time (s)")
    ax.set_ylabel("Tracker units/s")
    ax.set_title("Generated velocity profile")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_submovements(trajectory: np.ndarray, movement_time_s: float, fit: dict) -> plt.Figure:
    time, _, speed = velocity_arrays(trajectory, movement_time_s)
    fit_time = np.asarray(fit["time"])
    reconstructed = np.linalg.norm(np.asarray(fit["reconstructed_velocity"]), axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(time, speed, color="#111111", lw=2.2, label="generated")
    ax.plot(fit_time, reconstructed, color=COLORS["red"], lw=1.8, label="fitted sum")
    for index, row in enumerate(np.asarray(fit["parameters"])):
        component = minimum_jerk_velocity(fit_time, row[0], row[1], row[2:4])
        ax.plot(
            fit_time,
            np.linalg.norm(component, axis=1),
            lw=1.2,
            ls="--",
            label=f"component {index + 1}",
        )
    ax.set_xlabel("Movement time (s)")
    ax.set_ylabel("Speed (tracker units/s)")
    ax.set_title("Minimum-jerk decomposition")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def histogram(values: np.ndarray, label: str, unit: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.hist(values, bins="auto", color=COLORS["teal"], alpha=.86, edgecolor="white")
    ax.axvline(np.mean(values), color=COLORS["orange"], lw=2, label="mean")
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("Generated samples")
    ax.set_title(f"Generated {label.lower()} distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def comparison_plot(real: np.ndarray, generated: np.ndarray, label: str, unit: str, discrete: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    if discrete:
        support = np.arange(1, 5)
        width = .36
        real_rate = np.array([(real == value).mean() for value in support])
        generated_rate = np.array([(generated == value).mean() for value in support])
        ax.bar(support - width / 2, real_rate, width, color=COLORS["navy"], label="recorded query")
        ax.bar(support + width / 2, generated_rate, width, color=COLORS["orange"], label="generated")
        ax.set_xticks(support)
        ax.set_ylabel("Probability")
    else:
        combined = np.r_[real, generated]
        bins = np.histogram_bin_edges(combined, bins="auto")
        ax.hist(real, bins=bins, density=True, alpha=.58, color=COLORS["navy"], label="recorded query")
        ax.hist(generated, bins=bins, density=True, alpha=.58, color=COLORS["orange"], label="generated")
        ax.set_ylabel("Density")
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_title(f"Recorded and generated {label.lower()}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def latent_controls(
    latent_dim: int,
    selected_run: str,
    source: str,
    center: np.ndarray,
    scale: np.ndarray,
    fingerprints: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if source == "Population center":
        base = center.copy()
    else:
        row = fingerprints[
            (fingerprints.run == selected_run) & (fingerprints.subject == source)
        ].iloc[0]
        base = row[[f"z{i + 1}" for i in range(latent_dim)]].to_numpy(dtype=float)

    st.markdown("#### Latent controls")
    columns = st.columns(2 if latent_dim <= 3 else 4)
    offsets = []
    for index in range(latent_dim):
        with columns[index % len(columns)]:
            offsets.append(
                st.slider(
                    f"Latent {index + 1}",
                    -2.5,
                    2.5,
                    0.0,
                    0.1,
                    key=f"latent_{selected_run}_{source}_{index}",
                    help="Offset from the selected fingerprint, measured in training-set standard deviations.",
                )
            )
    offsets = np.asarray(offsets, dtype=float)
    return base + offsets * scale, offsets


def generator_tab(
    model: ConditionalVAE,
    norm: NormStats,
    selected_run: str,
    latent_dim: int,
    fingerprints: pd.DataFrame,
    stats_for_run: dict,
    speed_ranges: pd.DataFrame,
    sp: int,
    side: int,
    target_speed: float,
) -> None:
    source_options = ["Population center"] + sorted(
        fingerprints[fingerprints.run == selected_run].subject.unique().tolist()
    )
    source = st.selectbox("Fingerprint source", source_options, index=0)
    if source != "Population center":
        fp_row = fingerprints[
            (fingerprints.run == selected_run) & (fingerprints.subject == source)
        ].iloc[0]
        st.caption(
            f"Held-out participant fingerprint from {int(fp_row.n_context)} context trials; "
            f"query trials were not used to calculate it."
        )

    center = np.asarray(stats_for_run["training_center"], dtype=float)
    scale = np.asarray(stats_for_run["training_scale"], dtype=float)
    latent, offsets = latent_controls(
        latent_dim, selected_run, source, center, scale, fingerprints
    )
    trajectory_batch, timing_batch = decode(
        model, norm, latent, sp, side, target_speed
    )
    trajectory = trajectory_batch[0]
    timing = timing_batch[0]
    features = features_from_arrays(trajectory, timing[0], timing[1])

    view = st.segmented_control(
        "Generated output",
        ["Trajectory", "Velocity", "Submovements"],
        default="Trajectory",
        selection_mode="single",
    ) or "Trajectory"

    fit = None
    fit_error = np.nan
    component_count = np.nan
    if view == "Submovements":
        with st.spinner("Fitting one to four minimum-jerk components..."):
            fit = quick_submovement_fit(
                np.ascontiguousarray(trajectory, dtype=np.float64).tobytes(),
                trajectory.shape,
                float(timing[0]),
            )
        fit_error = float(fit["summary"]["mj_fit_error"])
        component_count = int(fit["summary"]["mj_n_components"])

    metrics = st.columns(5)
    metrics[0].metric("Movement duration", f"{1000 * timing[0]:.0f} ms")
    metrics[1].metric("Initiation time", f"{1000 * timing[1]:.0f} ms")
    metrics[2].metric("Peak speed", f"{features['peak_speed_tracker_units_s']:.1f}")
    metrics[3].metric("Path length", f"{features['path_length']:.2f}")
    metrics[4].metric(
        "Components" if np.isfinite(component_count) else "Curvature",
        f"{component_count:d}" if np.isfinite(component_count) else f"{features['curvature_index']:.3f}",
    )

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        if view == "Trajectory":
            st.pyplot(plot_trajectory(trajectory), width="stretch")
        elif view == "Velocity":
            st.pyplot(plot_velocity(trajectory, float(timing[0])), width="stretch")
        else:
            st.pyplot(
                plot_submovements(trajectory, float(timing[0]), fit),
                width="stretch",
            )
    with right:
        st.markdown("#### Current control point")
        control_table = pd.DataFrame(
            {
                "control": [f"latent {i + 1}" for i in range(latent_dim)],
                "offset_sd": offsets,
                "absolute_z": latent,
            }
        )
        st.dataframe(
            control_table.style.format({"offset_sd": "{:.1f}", "absolute_z": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )
        if fit is not None:
            st.markdown("#### Kinematic decomposition")
            st.write(
                f"Selected {component_count} component(s), normalized fit error "
                f"{fit_error:.3f}. This is a kinematic description, not a cognitive-strategy label."
            )
        export = pd.DataFrame(
            {
                "phase": np.linspace(0, 1, len(trajectory)),
                "lateral_x": trajectory[:, 0],
                "forward_y": trajectory[:, 1],
            }
        )
        st.download_button(
            "Download generated trajectory",
            export.to_csv(index=False).encode("utf-8"),
            file_name=f"generated_n{latent_dim}_{source.replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch",
        )

    st.divider()
    st.markdown("### Sample a distribution at this control point")
    sample_col, output_col, seed_col = st.columns([1, 1.4, 1])
    sample_count = sample_col.slider("Samples", 25, 500, 150, 25)
    output_label = output_col.selectbox(
        "Output", list(OUTPUTS.keys())[:6], index=0, key="live_output"
    )
    sampling_seed = seed_col.number_input("Sampling seed", 0, 100000, 2026, 1)
    covariance = np.asarray(stats_for_run["shared_covariance"], dtype=float)
    rng = np.random.default_rng(int(sampling_seed))
    sampled_latent = rng.multivariate_normal(latent, covariance, size=sample_count)
    sampled_trajectory, sampled_timing = decode(
        model, norm, sampled_latent, sp, side, target_speed
    )
    sampled_features = basic_features(sampled_trajectory, sampled_timing)
    output_column = OUTPUTS[output_label]
    values = sampled_features[output_column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    plot_col, summary_col = st.columns([1.55, 1], gap="large")
    with plot_col:
        st.pyplot(
            histogram(values, output_label, UNITS[output_column]),
            width="stretch",
        )
    with summary_col:
        st.markdown("#### Generated distribution")
        summary = pd.DataFrame(
            {
                "statistic": ["mean", "standard deviation", "5th percentile", "median", "95th percentile"],
                "value": [np.mean(values), np.std(values), *np.percentile(values, [5, 50, 95])],
            }
        )
        st.dataframe(summary.style.format({"value": "{:.4f}"}), hide_index=True, width="stretch")
        st.download_button(
            "Download generated samples",
            sampled_features.to_csv(index=False).encode("utf-8"),
            file_name=f"generated_distribution_n{latent_dim}.csv",
            mime="text/csv",
            width="stretch",
        )


def validation_tab(selected_run: str, latent_dim: int, empirical: pd.DataFrame) -> None:
    generated_path = GENERATION_DIR / f"{selected_run}_generated.csv"
    if not generated_path.exists():
        st.error(f"Missing generated validation data: {generated_path.name}")
        return
    generated = load_csv(str(generated_path))
    if "mj_fit_success" in generated:
        generated = generated[generated.mj_fit_success == True].copy()  # noqa: E712

    controls = st.columns([1, 1.3])
    subject = controls[0].selectbox(
        "Held-out participant", sorted(empirical.subject.unique()), key="validation_subject"
    )
    output_label = controls[1].selectbox(
        "Output distribution", list(OUTPUTS.keys()), key="validation_output"
    )
    column = OUTPUTS[output_label]
    real = empirical.loc[empirical.subject == subject, column].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().to_numpy(dtype=float)
    simulated = generated.loc[generated.subject == subject, column].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().to_numpy(dtype=float)
    discrete = column == "mj_n_components"

    metric_cols = st.columns(5)
    metric_cols[0].metric("Recorded query", len(real))
    metric_cols[1].metric("Generated", len(simulated))
    if discrete:
        support = np.arange(1, 5)
        p = np.array([(real == value).mean() for value in support])
        q = np.array([(simulated == value).mean() for value in support])
        tv = float(.5 * np.abs(p - q).sum())
        jsd = float(jensenshannon(p, q, base=2.0) ** 2)
        metric_cols[2].metric("JSD", f"{jsd:.3f}")
        metric_cols[3].metric("Total variation", f"{tv:.3f}")
        metric_cols[4].metric("Exact match", "No test")
    else:
        ks = stats.ks_2samp(real, simulated)
        wasserstein = stats.wasserstein_distance(real, simulated)
        metric_cols[2].metric("KS statistic", f"{ks.statistic:.3f}")
        metric_cols[3].metric("KS p-value", f"{ks.pvalue:.3g}")
        metric_cols[4].metric("Wasserstein", f"{wasserstein:.3g}")

    chart_col, note_col = st.columns([1.7, 1], gap="large")
    with chart_col:
        st.pyplot(
            comparison_plot(real, simulated, output_label, UNITS[column], discrete),
            width="stretch",
        )
    with note_col:
        st.markdown("#### Held-out comparison")
        st.write(
            "The participant fingerprint uses only the context half of that participant's trials. "
            "Recorded values come from the disjoint query half. Generated samples use the same query-condition mixture."
        )
        if not discrete:
            st.write(
                "A low KS statistic means closer distributions. A small p-value rejects exact equality; "
                "it does not mean the model contains no useful similarity."
            )
        else:
            st.write("JSD and total variation are used because component count is discrete.")
        export = pd.concat(
            [
                pd.DataFrame({"source": "recorded_query", "value": real}),
                pd.DataFrame({"source": "generated", "value": simulated}),
            ],
            ignore_index=True,
        )
        st.download_button(
            "Download compared values",
            export.to_csv(index=False).encode("utf-8"),
            file_name=f"{subject}_n{latent_dim}_{column}.csv",
            mime="text/csv",
            width="stretch",
        )


def comparison_tab(seed_results: pd.DataFrame, generation_summary: pd.DataFrame) -> None:
    grouped = seed_results.groupby("latent_dim").agg(
        trajectory_mse=("reconstruction_mse_tracker_units2", "mean"),
        movement_r2=("movement_time_s_r2", "mean"),
        initiation_r2=("initiation_time_s_r2", "mean"),
        fingerprint_accuracy=("fingerprint_balanced_accuracy", "mean"),
    )
    dimensions = grouped.index.to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4))
    specs = [
        ("trajectory_mse", "Trajectory MSE", True),
        ("movement_r2", "Movement-time R2", False),
        ("initiation_r2", "Initiation-time R2", False),
        ("fingerprint_accuracy", "Enrollment accuracy", False),
    ]
    for ax, (column, title, lower) in zip(axes, specs):
        ax.bar(dimensions.astype(str), grouped[column], color=[COLORS["teal"], COLORS["orange"], COLORS["navy"]])
        ax.set_title(title)
        ax.set_xlabel("Latent n")
        ax.grid(axis="y", alpha=.2)
        if not lower:
            ax.axhline(0, color="#777777", lw=.8)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    generation = generation_summary.copy()
    generation["latent_dim"] = generation.run.str.extract(r"_z(\d+)_", expand=False).astype(int)
    generated_group = generation.groupby("latent_dim").agg(
        mean_continuous_ks=("mean_ks_movement_time_s", "mean"),
        component_count_jsd=("mean_count_jsd", "mean"),
        fdr_rejections=("mean_ks_rejected_fdr", "mean"),
    )
    display = grouped.join(generated_group).reset_index()
    display["fingerprint_accuracy"] *= 100
    st.dataframe(
        display.style.format(
            {
                "trajectory_mse": "{:.3f}",
                "movement_r2": "{:.3f}",
                "initiation_r2": "{:.3f}",
                "fingerprint_accuracy": "{:.1f}%",
                "mean_continuous_ks": "{:.3f}",
                "component_count_jsd": "{:.3f}",
                "fdr_rejections": "{:.1f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "n=8 is the strongest capacity model. n=2 and n=3 remain the scientific low-dimensional targets, "
        "but they sacrifice reconstruction and distribution fidelity. Initiation time is weak at every n."
    )


def protocol_tab(manifest: dict, empirical: pd.DataFrame, seed_results: pd.DataFrame) -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("### Final protocol")
        protocol = pd.DataFrame(
            [
                ("Data", f"{manifest['n_trials']:,} condition-2 trials, {manifest['n_subjects']} participants"),
                ("Plane", "x lateral and y forward; z treated as off-plane variation"),
                ("Input", "100 movement-phase points x 2 coordinates"),
                ("Condition", "start/speed category, side, exact executed target speed"),
                ("Encoder", "trajectory and condition; true timing withheld"),
                ("Decoder", "trajectory, movement duration, initiation time"),
                ("Evaluation", "17/4/7 subject split; disjoint context/query test trials"),
                ("Models", "n=2, n=3, n=8; three initialization seeds"),
            ],
            columns=["item", "decision"],
        )
        st.dataframe(protocol, hide_index=True, width="stretch")
    with right:
        st.markdown("### Questions retained for Prof. Friedman")
        st.markdown(
            """
            1. Confirm the 2-D x-y table plane and treatment of z as off-plane variation.
            2. Confirm the minimum-jerk duration and onset-spacing constraints.
            3. Confirm the component-order rule and error thresholds.
            4. Confirm that MAT `dotArray` is authoritative when it differs from an external stimulus file.
            5. Confirm the 1920-pixel normalization used for exact target speed.
            6. Clarify the free-eye-condition `Not fixating on the dot enough!!!` label.
            7. Confirm tracker position units and the late-arrival exclusion.
            """
        )

    st.markdown("### Export")
    buttons = st.columns(3)
    if REPORT_PATH.exists():
        buttons[0].download_button(
            "Download final report",
            REPORT_PATH.read_bytes(),
            file_name=REPORT_PATH.name,
            mime="application/pdf",
            width="stretch",
        )
    buttons[1].download_button(
        "Download model results",
        seed_results.to_csv(index=False).encode("utf-8"),
        file_name="cvae_seed_results.csv",
        mime="text/csv",
        width="stretch",
    )
    buttons[2].download_button(
        "Download recorded query features",
        empirical.to_csv(index=False).encode("utf-8"),
        file_name="heldout_query_features.csv",
        mime="text/csv",
        width="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="Interception Movement Fingerprints",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_style()
    required = [
        ASSETS_DIR / "manifest.json",
        ASSETS_DIR / "latent_stats.json",
        ASSETS_DIR / "subject_fingerprints.csv",
        ASSETS_DIR / "empirical_query_features.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        st.error("Dashboard assets are missing. Run `python scripts/build_dashboard_assets.py`.")
        st.code("\n".join(str(path) for path in missing))
        return

    manifest = load_json(str(ASSETS_DIR / "manifest.json"))
    latent_stats = load_json(str(ASSETS_DIR / "latent_stats.json"))
    fingerprints = load_csv(str(ASSETS_DIR / "subject_fingerprints.csv"))
    empirical = load_csv(str(ASSETS_DIR / "empirical_query_features.csv"))
    speed_ranges = load_csv(str(ASSETS_DIR / "condition_speed_ranges.csv"))
    seed_results = load_csv(str(MODELS_DIR / "seed_results.csv"))
    generation_summary = load_csv(str(GENERATION_DIR / "generation_summary.csv"))

    st.title("Interception movement fingerprint explorer")
    st.caption(
        "Final 2-D trajectory-only conditional VAE | condition 2 | held-out subject evaluation"
    )

    st.sidebar.header("Model")
    latent_dim = st.sidebar.segmented_control(
        "Latent dimensions", manifest["latent_dimensions"], default=2, selection_mode="single"
    ) or 2
    with st.sidebar.expander("Initialization", expanded=False):
        seed = st.selectbox("Model seed", manifest["model_seeds"], index=0)
        st.caption("Latent axes can rotate or reflect between seeds. Seed 42 matches the report traversal.")
    selected_run = run_name(int(latent_dim), int(seed))
    if selected_run not in latent_stats:
        st.error(f"No saved assets for {selected_run}")
        return
    model, norm = load_model(selected_run)

    st.sidebar.header("Task condition")
    sp = st.sidebar.selectbox(
        "Start/speed category",
        [1, 2, 3],
        index=1,
        format_func=lambda value: {1: "1: 120 / slow", 2: "2: 140 / medium", 3: "3: 160 / fast"}[value],
    )
    side_name = st.sidebar.radio("Starting side", ["Left", "Right"], horizontal=True)
    side = 1 if side_name == "Left" else 2
    speed_row = speed_ranges[speed_ranges.sp == sp].iloc[0]
    target_speed = st.sidebar.slider(
        "Executed target speed",
        round(float(speed_row.speed_min), 3),
        round(float(speed_row.speed_max), 3),
        round(float(speed_row.speed_median), 3),
        0.001,
        key=f"speed_{sp}",
    )
    st.sidebar.caption("Screen widths/s from the executed MAT target; 1920-pixel normalization pending confirmation.")
    st.sidebar.divider()
    st.sidebar.caption(f"Loaded n={latent_dim}, seed={seed} | timing withheld from encoder")

    tabs = st.tabs(["Generate", "Distribution check", "Model comparison", "Protocol and downloads"])
    with tabs[0]:
        generator_tab(
            model,
            norm,
            selected_run,
            int(latent_dim),
            fingerprints,
            latent_stats[selected_run],
            speed_ranges,
            sp,
            side,
            target_speed,
        )
    with tabs[1]:
        validation_tab(selected_run, int(latent_dim), empirical)
    with tabs[2]:
        comparison_tab(seed_results, generation_summary)
    with tabs[3]:
        protocol_tab(manifest, empirical, seed_results)


if __name__ == "__main__":
    main()
