"""Interactive explorer for the strategy-window CVAE comparison."""
from __future__ import annotations

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
from src.features import features_from_generated_window, movement_from_generated_window
from src.submovements import SubmovementConfig, decompose_normalized_trajectory, minimum_jerk_velocity
from src.vae_model import ConditionalVAE, NormStats, encode_trial_condition


RESULTS = config.RESULTS_DIR
ASSETS = RESULTS / "dashboard"
REPORT = config.STUDY_ROOT / "output" / "advisor_brief" / "Interception_Movement_Advisor_Brief.pdf"
COLORS = {"movement_only": "#2563EB", "go_to_arrival": "#DC2626"}
OUTPUTS = {
    "Movement time": "movement_time_s",
    "Initiation time": "initiation_time_s",
    "Peak speed": "peak_speed_tracker_units_s",
    "Path length": "path_length",
    "Curvature index": "curvature_index",
    "Maximum lateral deviation": "max_lateral_deviation",
    "Minimum-jerk fit error": "mj_fit_error",
    "Component count": "mj_n_components",
    "First component duration": "mj_first_duration_s",
    "First component amplitude": "mj_first_amplitude",
    "Secondary amplitude fraction": "mj_secondary_amplitude_fraction",
    "Mean component overlap": "mj_mean_overlap_pct",
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1420px; padding-top: 1.2rem; padding-bottom: 2rem;}
        h1 {font-size: 1.75rem !important; letter-spacing: 0 !important;}
        h2 {font-size: 1.25rem !important; letter-spacing: 0 !important;}
        h3 {font-size: 1.02rem !important; letter-spacing: 0 !important;}
        [data-testid="stMetric"] {background:#f7f9fb; border:1px solid #dce3e8; border-radius:6px; padding:.7rem;}
        [data-testid="stSidebar"] {background:#f3f6f8; border-right:1px solid #dce3e8;}
        .stTabs [data-baseweb="tab-list"] {gap:1.25rem; border-bottom:1px solid #dce3e8;}
        #MainMenu, footer, [data-testid="stToolbar"] {visibility:hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_name(window_mode: str, latent_dim: int, seed: int) -> str:
    return f"cvae_{window_mode}_z{latent_dim}_seed{seed}"


@st.cache_resource
def load_model(window_mode: str, name: str) -> tuple[ConditionalVAE, NormStats]:
    path = RESULTS / window_mode / "models" / name / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model_cfg = cfg.get("model", cfg)
    model = ConditionalVAE(
        input_dim=int(checkpoint.get("input_dim", len(checkpoint["train_mean"]))),
        condition_dim=int(checkpoint.get("condition_dim", 4)),
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        timing_dim=int(checkpoint["timing_dim"]),
        encoder_uses_timing=bool(checkpoint.get("encoder_uses_timing", False)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, NormStats.from_checkpoint(checkpoint)


def decode(
    model: ConditionalVAE,
    norm: NormStats,
    latent: np.ndarray,
    sp: int,
    side: int,
    target_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    latent = np.atleast_2d(np.asarray(latent, dtype=np.float32))
    metadata = {"sp": sp, "side": side, "target_speed_screen_s": target_speed}
    condition = encode_trial_condition(metadata, model.condition_dim)
    condition = np.repeat(condition[None, :], len(latent), axis=0)
    train_mean, train_std, _, _ = norm.torch("cpu")
    with torch.no_grad():
        trajectory_z, timing_z = model.decode(
            torch.as_tensor(latent), torch.as_tensor(condition, dtype=torch.float32)
        )
    channels = model.input_dim // config.NORMALISED_LENGTH
    trajectories = ((trajectory_z * train_std + train_mean).numpy()).reshape(
        len(latent), config.NORMALISED_LENGTH, channels
    )
    timing = norm.denormalise_timing(timing_z.numpy())
    timing[:, 0] = np.maximum(timing[:, 0], 1e-3)
    timing[:, 1] = np.maximum(timing[:, 1], 0.0)
    return trajectories, timing


def full_time_axis(window_mode: str, movement_time: float, initiation_time: float, n: int) -> np.ndarray:
    total = movement_time if window_mode == config.WINDOW_MOVEMENT_ONLY else movement_time + initiation_time
    return np.linspace(0, max(total, 1e-3), n)


def movement_start_index(window_mode: str, movement_time: float, initiation_time: float, n: int) -> int:
    if window_mode == config.WINDOW_MOVEMENT_ONLY:
        return 0
    total = movement_time + initiation_time
    return int(np.clip(round((initiation_time / max(total, 1e-9)) * (n - 1)), 0, n - 2))


def plot_trajectory(trajectory: np.ndarray, window_mode: str, move_time: float, init_time: float) -> plt.Figure:
    index = movement_start_index(window_mode, move_time, init_time, len(trajectory))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if index:
        ax.plot(trajectory[: index + 1, 0], trajectory[: index + 1, 1], color="#94A3B8", lw=2, label="waiting interval")
    ax.plot(trajectory[index:, 0], trajectory[index:, 1], color="#0F766E", lw=2.4, label="movement interval")
    ax.scatter(*trajectory[0], s=65, color="#2563EB", label="go signal")
    ax.scatter(*trajectory[index], s=65, color="#D97706", label="predicted movement onset")
    ax.scatter(*trajectory[-1], s=65, color="#B91C1C", label="arrival")
    ax.set_xlabel("Lateral x (tracker units)")
    ax.set_ylabel("Forward y (tracker units)")
    ax.set_title("Generated table-plane trajectory", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_speed(trajectory: np.ndarray, window_mode: str, move_time: float, init_time: float) -> plt.Figure:
    time = full_time_axis(window_mode, move_time, init_time, len(trajectory))
    velocity = np.gradient(trajectory, time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(time, speed, color="#17324D", lw=2.3)
    if window_mode == config.WINDOW_GO_TO_ARRIVAL:
        ax.axvline(init_time, color="#D97706", lw=1.5, ls="--", label="predicted movement onset")
        ax.legend(frameon=False)
    ax.set_xlabel("Time since target motion onset (s)" if window_mode == config.WINDOW_GO_TO_ARRIVAL else "Time since movement onset (s)")
    ax.set_ylabel("Speed (tracker units/s)")
    ax.set_title("Generated speed profile", loc="left", weight="bold")
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def fit_submovements(data: bytes, shape: tuple[int, int], movement_time: float) -> dict:
    movement = np.frombuffer(data, dtype=np.float64).reshape(shape)
    fit = decompose_normalized_trajectory(
        movement,
        movement_time,
        SubmovementConfig(restarts=1, max_nfev=300),
        "dashboard-generated",
    )
    return {
        "summary": fit.summary(),
        "parameters": fit.selected.parameters,
        "time": fit.selected.time,
        "reconstructed_velocity": fit.selected.reconstructed_velocity,
    }


def plot_submovements(movement: np.ndarray, movement_time: float, fit: dict) -> plt.Figure:
    time = np.linspace(0, movement_time, len(movement))
    observed = np.linalg.norm(np.gradient(movement, time, axis=0), axis=1)
    fit_time = np.asarray(fit["time"])
    reconstructed = np.linalg.norm(np.asarray(fit["reconstructed_velocity"]), axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(time, observed, color="#111827", lw=2.2, label="generated movement")
    ax.plot(fit_time, reconstructed, color="#DC2626", lw=1.8, label="component sum")
    for i, row in enumerate(np.asarray(fit["parameters"])):
        component = minimum_jerk_velocity(fit_time, row[0], row[1], row[2:4])
        ax.plot(fit_time, np.linalg.norm(component, axis=1), lw=1.1, ls="--", label=f"component {i + 1}")
    ax.set_xlabel("Time since movement onset (s)")
    ax.set_ylabel("Speed (tracker units/s)")
    ax.set_title("Minimum-jerk decomposition", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def latent_controls(name: str, latent_dim: int, stats_for_run: dict, fingerprints: pd.DataFrame) -> np.ndarray:
    options = ["Training-population center"] + sorted(fingerprints[fingerprints.run == name].subject.unique())
    source = st.selectbox("Fingerprint source", options)
    if source == "Training-population center":
        base = np.asarray(stats_for_run["training_center"], dtype=float)
    else:
        row = fingerprints[(fingerprints.run == name) & (fingerprints.subject == source)].iloc[0]
        base = row[[f"z{i + 1}" for i in range(latent_dim)]].to_numpy(float)
        st.caption(f"Held-out participant context fingerprint from {int(row.n_context)} trials.")
    scale = np.asarray(stats_for_run["training_scale"], dtype=float)
    columns = st.columns(2 if latent_dim <= 3 else 4)
    offsets = []
    for index in range(latent_dim):
        offsets.append(columns[index % len(columns)].slider(
            f"Latent {index + 1}", -2.5, 2.5, 0.0, 0.1,
            key=f"{name}_{source}_{index}",
            help="Offset in training-set latent standard deviations.",
        ))
    return base + np.asarray(offsets) * scale


def generator_tab(
    window_mode: str,
    name: str,
    latent_dim: int,
    model: ConditionalVAE,
    norm: NormStats,
    stats_for_run: dict,
    fingerprints: pd.DataFrame,
    sp: int,
    side: int,
    target_speed: float,
) -> None:
    latent = latent_controls(name, latent_dim, stats_for_run, fingerprints)
    trajectories, timing = decode(model, norm, latent, sp, side, target_speed)
    trajectory = trajectories[0]
    move_time, init_time = map(float, timing[0])
    features = features_from_generated_window(trajectory, move_time, init_time, window_mode)
    view = st.segmented_control("Output view", ["Trajectory", "Speed", "Submovements"], default="Trajectory") or "Trajectory"
    metrics = st.columns(5)
    metrics[0].metric("Initiation", f"{1000 * init_time:.0f} ms")
    metrics[1].metric("Movement", f"{1000 * move_time:.0f} ms")
    metrics[2].metric("Peak speed", f"{features['peak_speed_tracker_units_s']:.1f}")
    metrics[3].metric("Path length", f"{features['path_length']:.2f}")
    metrics[4].metric("Curvature", f"{features['curvature_index']:.3f}")
    if view == "Trajectory":
        st.pyplot(plot_trajectory(trajectory, window_mode, move_time, init_time), width="stretch")
    elif view == "Speed":
        st.pyplot(plot_speed(trajectory, window_mode, move_time, init_time), width="stretch")
    else:
        movement = movement_from_generated_window(trajectory, move_time, init_time, window_mode)
        with st.spinner("Fitting one to four minimum-jerk components..."):
            fit = fit_submovements(np.ascontiguousarray(movement, dtype=np.float64).tobytes(), movement.shape, move_time)
        st.pyplot(plot_submovements(movement, move_time, fit), width="stretch")
        st.caption(
            f"Selected {int(fit['summary']['mj_n_components'])} component(s); normalized fit error "
            f"{fit['summary']['mj_fit_error']:.3f}. Components are kinematic descriptions, not cognitive labels."
        )

    st.divider()
    left, middle, right = st.columns([1, 1.2, 1])
    n_samples = left.slider("Generated samples", 25, 500, 150, 25)
    output_label = middle.selectbox("Distribution output", list(OUTPUTS)[:6])
    random_seed = right.number_input("Sampling seed", 0, 100000, 2026, 1)
    rng = np.random.default_rng(int(random_seed))
    covariance = np.asarray(stats_for_run["shared_covariance"], dtype=float)
    z = rng.multivariate_normal(latent, covariance, size=n_samples)
    sample_trajectories, sample_timing = decode(model, norm, z, sp, side, target_speed)
    rows = [
        features_from_generated_window(sample_trajectories[i], sample_timing[i, 0], sample_timing[i, 1], window_mode)
        for i in range(n_samples)
    ]
    generated = pd.DataFrame(rows)
    column = OUTPUTS[output_label]
    values = generated[column].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(values, bins="auto", color="#0F766E", edgecolor="white")
    ax.axvline(values.mean(), color="#D97706", lw=2, label="mean")
    ax.set_xlabel(output_label)
    ax.set_ylabel("Generated samples")
    ax.set_title(f"Generated {output_label.lower()} distribution", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.download_button("Download generated samples", generated.to_csv(index=False).encode(), f"{name}_{column}.csv", "text/csv")


def validation_tab(name: str, window_mode: str, empirical: pd.DataFrame, model_row: pd.Series) -> None:
    metrics = st.columns(5)
    metrics[0].metric("Movement-time R2", f"{model_row.movement_time_s_r2:.3f}")
    metrics[1].metric("Initiation-time R2", f"{model_row.initiation_time_s_r2:.3f}")
    metrics[2].metric("Subject balanced accuracy", f"{100 * model_row.fingerprint_balanced_accuracy:.1f}%")
    metrics[3].metric("Chance", f"{100 * model_row.fingerprint_chance:.1f}%")
    metrics[4].metric("Mean KS", f"{model_row.mean_ks:.3f}")
    generated_path = RESULTS / window_mode / "generation" / f"{name}_generated.csv"
    if not generated_path.exists():
        st.info("Detailed generated minimum-jerk distributions are precomputed for the representative n=3 and n=8, seed=42 models. All model combinations retain the held-out metrics above.")
        return
    generated = read_csv(str(generated_path))
    generated = generated[generated.mj_fit_success == True].copy()  # noqa: E712
    subject_col, output_col = st.columns(2)
    subject = subject_col.selectbox("Held-out participant", sorted(empirical.subject.unique()))
    output_label = output_col.selectbox("Recorded versus generated output", list(OUTPUTS))
    column = OUTPUTS[output_label]
    real = empirical.loc[empirical.subject == subject, column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    simulated = generated.loc[generated.subject == subject, column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    if column == "mj_n_components":
        support = np.arange(1, 5)
        p = np.array([(real == value).mean() for value in support])
        q = np.array([(simulated == value).mean() for value in support])
        ax.bar(support - .18, p, .36, color="#17324D", label="recorded query")
        ax.bar(support + .18, q, .36, color="#D97706", label="generated")
        score = float(jensenshannon(p, q, base=2) ** 2)
        label = f"JSD={score:.3f} (lower is better)"
    else:
        bins = np.histogram_bin_edges(np.r_[real, simulated], bins="auto")
        ax.hist(real, bins=bins, density=True, alpha=.6, color="#17324D", label="recorded query")
        ax.hist(simulated, bins=bins, density=True, alpha=.6, color="#D97706", label="generated")
        score = stats.ks_2samp(real, simulated).statistic
        label = f"KS={score:.3f} (lower is better)"
    ax.set_xlabel(output_label)
    ax.set_ylabel("Probability" if column == "mj_n_components" else "Density")
    ax.set_title(f"Participant {subject}: {label}", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def comparison_tab(results: pd.DataFrame) -> None:
    metric_options = {
        "Initiation-time R2": "initiation_time_s_r2",
        "Movement-time R2": "movement_time_s_r2",
        "Subject balanced accuracy": "fingerprint_balanced_accuracy",
        "Generated-distribution KS": "mean_ks",
        "Movement-region MSE": "movement_reconstruction_mse_tracker_units2",
    }
    label = st.selectbox("Comparison metric", list(metric_options))
    metric = metric_options[label]
    summary = results.groupby(["window_mode", "latent_dim"])[metric].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for mode, group in summary.groupby("window_mode"):
        group = group.sort_values("latent_dim")
        ax.errorbar(group.latent_dim, group["mean"], yerr=group["std"], marker="o", lw=2.2, capsize=4, color=COLORS[mode], label=mode.replace("_", " "))
    if metric.endswith("_r2"):
        ax.axhline(0, color="#64748B", ls="--", lw=1)
    if metric == "fingerprint_balanced_accuracy":
        ax.axhline(1 / 7, color="#64748B", ls="--", lw=1, label="chance")
    ax.set_xticks([2, 3, 4, 8])
    ax.set_xlabel("Latent dimensions")
    ax.set_ylabel(label)
    ax.set_title(f"{label} across three training seeds", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.dataframe(summary.style.format({"mean": "{:.4f}", "std": "{:.4f}"}), hide_index=True, width="stretch")
    st.caption("Raw full-window reconstruction MSE is not compared across protocols because the modeled intervals differ.")


def association_tab(window_mode: str, latent_dim: int) -> None:
    level = st.segmented_control("Association level", ["Trial-level partial", "Subject context-to-query"], default="Trial-level partial") or "Trial-level partial"
    suffix = "trial_within_subject_partial" if level == "Trial-level partial" else "subject_context_to_query"
    path = RESULTS / "latent_associations" / "heatmaps" / f"{window_mode}_z{latent_dim}_{suffix}.png"
    if path.exists():
        st.image(str(path), width="stretch")
        st.caption("Seed 42. Asterisks survive within-model Benjamini-Hochberg FDR correction. Associations do not establish causal control, and axes may rotate across seeds.")
    else:
        st.info("Association heatmap is not available yet for this model.")


def protocol_tab(manifest: dict, results: pd.DataFrame) -> None:
    st.markdown("### Fixed study protocol")
    st.dataframe(pd.DataFrame({
        "Item": ["Cohort", "Representation", "Windows", "Split", "Latent widths", "Training repetitions", "Timing"],
        "Value": [
            f"{manifest['n_trials']:,} condition-2 trials from {manifest['n_subjects']} participants",
            "2-D table plane, x lateral and y forward; 100 phase samples",
            "movement onset to arrival; target-motion onset to arrival",
            "17 train / 4 validation / 7 held-out test participants",
            "2, 3, 4, 8",
            "seeds 42, 43, 44",
            "withheld from encoder; movement and initiation time decoded separately",
        ],
    }), hide_index=True, width="stretch")
    st.markdown("### Interpretation boundary")
    st.write("n=3 is the smallest stable strategy-inclusive model. n=8 is the capacity comparator and is not claimed to provide eight directly interpretable psychological variables. Minimum-jerk components describe velocity structure; they do not by themselves prove hesitation or a cognitive strategy.")
    st.download_button("Download all model metrics", results.to_csv(index=False).encode(), "model_seed_results.csv", "text/csv")
    if REPORT.exists():
        st.download_button("Download study PDF", REPORT.read_bytes(), REPORT.name, "application/pdf")


def main() -> None:
    st.set_page_config(page_title="Interception strategy explorer", layout="wide")
    apply_style()
    required = [ASSETS / "manifest.json", ASSETS / "latent_stats.json", ASSETS / "subject_fingerprints.csv", ASSETS / "empirical_query_features.csv"]
    missing = [path for path in required if not path.exists()]
    if missing:
        st.error("Dashboard assets are missing. Run `python scripts/build_strategy_dashboard_assets.py`.")
        return
    manifest = read_json(str(ASSETS / "manifest.json"))
    latent_stats = read_json(str(ASSETS / "latent_stats.json"))
    fingerprints = read_csv(str(ASSETS / "subject_fingerprints.csv"))
    empirical = read_csv(str(ASSETS / "empirical_query_features.csv"))
    speed_ranges = read_csv(str(ASSETS / "condition_speed_ranges.csv"))
    results = read_csv(str(RESULTS / "model_seed_results.csv"))

    st.title("Interception movement strategy explorer")
    st.caption("Conditional VAE comparison | condition 2 | held-out participant evaluation")
    st.sidebar.header("Representation")
    window_mode = st.sidebar.radio(
        "Temporal window",
        list(config.WINDOW_MODES),
        format_func=lambda value: "Movement onset -> arrival" if value == config.WINDOW_MOVEMENT_ONLY else "Target motion onset -> arrival",
    )
    latent_dimensions = [int(value) for value in manifest["latent_dimensions"]]
    model_seeds = [int(value) for value in manifest["model_seeds"]]
    default_dim = 3 if 3 in latent_dimensions else latent_dimensions[0]
    latent_dim = int(st.sidebar.segmented_control("Latent dimensions", latent_dimensions, default=default_dim) or default_dim)
    seed = int(st.sidebar.selectbox("Training seed", model_seeds))
    name = run_name(window_mode, latent_dim, seed)
    model, norm = load_model(window_mode, name)

    st.sidebar.header("Task condition")
    sp = int(st.sidebar.selectbox("Start/speed category", [1, 2, 3], index=1, format_func=lambda v: {1: "1: 120 / slow", 2: "2: 140 / medium", 3: "3: 160 / fast"}[v]))
    side = 1 if st.sidebar.radio("Starting side", ["Left", "Right"], horizontal=True) == "Left" else 2
    speed_row = speed_ranges[speed_ranges.sp == sp].iloc[0]
    target_speed = st.sidebar.slider("Executed target speed", float(speed_row.speed_min), float(speed_row.speed_max), float(speed_row.speed_median))
    st.sidebar.caption("Exact executed target speed is taken from the MAT stimulus trajectory.")

    model_row = results[(results.window_mode == window_mode) & (results.latent_dim == latent_dim) & (results.seed == seed)].iloc[0]
    tabs = st.tabs(["Generate", "Held-out validation", "Model comparison", "Latent associations", "Protocol"])
    with tabs[0]:
        generator_tab(window_mode, name, latent_dim, model, norm, latent_stats[name], fingerprints, sp, side, target_speed)
    with tabs[1]:
        validation_tab(name, window_mode, empirical, model_row)
    with tabs[2]:
        comparison_tab(results)
    with tabs[3]:
        association_tab(window_mode, latent_dim)
    with tabs[4]:
        protocol_tab(manifest, results)


if __name__ == "__main__":
    main()
