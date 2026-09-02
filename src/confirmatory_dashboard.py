"""Interactive dashboard for the frozen strategy-window confirmatory study."""
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
from src.submovements import (
    SubmovementConfig,
    decompose_normalized_trajectory,
    minimum_jerk_velocity,
)
from src.vae_model import ConditionalVAE, NormStats, encode_trial_condition


STUDY = ROOT / "studies" / "final_strategy_evaluation"
RUNS = STUDY / "runs" / "cvae" / "fold0"
ASSETS = ROOT / "studies" / "review_corrected_evaluation" / "results" / "dashboard"
REPORT = ROOT / "output" / "pdf" / "Interception_Movements_Final_Scientific_Report.pdf"
GUIDE = ROOT / "output" / "pdf" / "Interception_Movements_Results_Guide.pdf"
WINDOW_MODE = config.WINDOW_GO_TO_ARRIVAL
LIVE_SEED = 42
COLORS = {
    "navy": "#17324D",
    "blue": "#24739A",
    "teal": "#1D8F7A",
    "amber": "#C87519",
    "red": "#B6463A",
    "gray": "#6B7785",
    "light": "#E7EDF2",
}

OUTPUTS = {
    "Initiation time": "initiation_time_s",
    "Movement time": "movement_time_s",
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

MODEL_LABELS = {
    "condition_ridge": "Condition-only Ridge",
    "spline_pca": "Spline+PCA",
    "conditional_ae": "Conditional AE",
    "cvae": "CVAE",
    "unconditional_vae": "Unconditional VAE",
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1420px; padding-top: 1.1rem; padding-bottom: 2rem;}
        h1 {font-size: 1.75rem !important; letter-spacing: 0 !important;}
        h2 {font-size: 1.28rem !important; letter-spacing: 0 !important;}
        h3 {font-size: 1.03rem !important; letter-spacing: 0 !important;}
        [data-testid="stMetric"] {background:#f7f9fb; border:1px solid #d9e2e8; border-radius:6px; padding:.65rem;}
        [data-testid="stSidebar"] {background:#f3f6f8; border-right:1px solid #d9e2e8;}
        .stTabs [data-baseweb="tab-list"] {gap:1.05rem; border-bottom:1px solid #d9e2e8;}
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


def run_name(latent_dim: int) -> str:
    return f"cvae_z{latent_dim}_seed{LIVE_SEED}"


def checkpoint_path(latent_dim: int) -> Path:
    return RUNS / f"cvae_z{latent_dim}_seed{LIVE_SEED}" / "checkpoint.pt"


@st.cache_resource
def load_model(latent_dim: int) -> tuple[ConditionalVAE, NormStats]:
    checkpoint = torch.load(checkpoint_path(latent_dim), map_location="cpu", weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    model = ConditionalVAE(
        input_dim=int(checkpoint["input_dim"]),
        condition_dim=int(checkpoint["condition_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        timing_dim=int(checkpoint["timing_dim"]),
        encoder_uses_timing=bool(checkpoint["encoder_uses_timing"]),
        variational=True,
        use_condition=True,
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
    trajectories = ((trajectory_z * train_std + train_mean).numpy()).reshape(
        len(latent), 100, 2
    )
    timing = norm.denormalise_timing(timing_z.numpy())
    timing[:, 0] = np.maximum(timing[:, 0], 1e-3)
    timing[:, 1] = np.maximum(timing[:, 1], 0.0)
    return trajectories, timing


def movement_start_index(movement_time: float, initiation_time: float, n: int) -> int:
    total = movement_time + initiation_time
    return int(np.clip(round((initiation_time / max(total, 1e-9)) * (n - 1)), 0, n - 2))


def plot_trajectory(trajectory: np.ndarray, movement_time: float, initiation_time: float) -> plt.Figure:
    onset = movement_start_index(movement_time, initiation_time, len(trajectory))
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    if onset:
        ax.plot(
            trajectory[: onset + 1, 0], trajectory[: onset + 1, 1],
            color="#9AA6B2", lw=2.2, label="target-moving / finger waiting",
        )
    ax.plot(
        trajectory[onset:, 0], trajectory[onset:, 1],
        color=COLORS["teal"], lw=2.5, label="finger movement",
    )
    ax.scatter(*trajectory[0], s=58, color=COLORS["blue"], label="target motion onset")
    ax.scatter(*trajectory[onset], s=58, color=COLORS["amber"], label="predicted movement onset")
    ax.scatter(*trajectory[-1], s=58, color=COLORS["red"], label="arrival")
    ax.set_xlabel("Lateral x (tracker units)")
    ax.set_ylabel("Forward y (tracker units)")
    ax.set_title("Generated table-plane trajectory", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_speed(trajectory: np.ndarray, movement_time: float, initiation_time: float) -> plt.Figure:
    total = max(movement_time + initiation_time, 1e-3)
    time = np.linspace(0, total, len(trajectory))
    speed = np.linalg.norm(np.gradient(trajectory, time, axis=0), axis=1)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(time, speed, color=COLORS["navy"], lw=2.3)
    ax.axvline(initiation_time, color=COLORS["amber"], lw=1.6, ls="--", label="predicted movement onset")
    ax.set_xlabel("Time since target motion onset (s)")
    ax.set_ylabel("Speed (tracker units/s)")
    ax.set_title("Generated speed profile", loc="left", weight="bold")
    ax.legend(frameon=False)
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
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.plot(time, observed, color=COLORS["navy"], lw=2.2, label="generated movement")
    ax.plot(fit_time, reconstructed, color=COLORS["red"], lw=1.9, label="component sum")
    for index, row in enumerate(np.asarray(fit["parameters"])):
        component = minimum_jerk_velocity(fit_time, row[0], row[1], row[2:4])
        ax.plot(
            fit_time, np.linalg.norm(component, axis=1), lw=1.1, ls="--",
            label=f"component {index + 1}",
        )
    ax.set_xlabel("Time since movement onset (s)")
    ax.set_ylabel("Speed (tracker units/s)")
    ax.set_title("Minimum-jerk decomposition", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def latent_controls(
    name: str,
    latent_dim: int,
    stats_for_run: dict,
    fingerprints: pd.DataFrame,
) -> np.ndarray:
    available = fingerprints[fingerprints.run == name]
    options = ["Training-population center"] + sorted(available.subject.unique())
    source = st.selectbox("Fingerprint source", options)
    if source == "Training-population center":
        base = np.asarray(stats_for_run["training_center"], dtype=float)
    else:
        row = available[available.subject == source].iloc[0]
        base = row[[f"z{i + 1}" for i in range(latent_dim)]].to_numpy(float)
        st.caption(f"Context fingerprint estimated from {int(row.n_context)} held-out-participant trials.")

    scale = np.asarray(stats_for_run["training_scale"], dtype=float)
    columns = st.columns(2 if latent_dim <= 3 else 4)
    offsets = [
        columns[index % len(columns)].slider(
            f"z{index + 1} offset (SD)", -2.5, 2.5, 0.0, 0.1,
            key=f"{name}_{source}_{index}",
        )
        for index in range(latent_dim)
    ]
    return base + np.asarray(offsets) * scale


def generate_feature_frame(
    model: ConditionalVAE,
    norm: NormStats,
    latent: np.ndarray,
    sp: int,
    side: int,
    target_speed: float,
) -> pd.DataFrame:
    trajectories, timing = decode(model, norm, latent, sp, side, target_speed)
    return pd.DataFrame([
        features_from_generated_window(
            trajectories[index], timing[index, 0], timing[index, 1], WINDOW_MODE
        )
        for index in range(len(trajectories))
    ])


def generator_tab(
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
    movement_time, initiation_time = map(float, timing[0])
    features = features_from_generated_window(
        trajectory, movement_time, initiation_time, WINDOW_MODE
    )

    view = st.segmented_control(
        "Output view", ["Trajectory", "Speed", "Minimum-jerk"], default="Trajectory"
    ) or "Trajectory"
    metrics = st.columns(5)
    metrics[0].metric("Initiation", f"{1000 * initiation_time:.0f} ms")
    metrics[1].metric("Movement", f"{1000 * movement_time:.0f} ms")
    metrics[2].metric("Peak speed", f"{features['peak_speed_tracker_units_s']:.1f}")
    metrics[3].metric("Path length", f"{features['path_length']:.2f}")
    metrics[4].metric("Curvature", f"{features['curvature_index']:.3f}")

    if view == "Trajectory":
        st.pyplot(plot_trajectory(trajectory, movement_time, initiation_time), width="stretch")
    elif view == "Speed":
        st.pyplot(plot_speed(trajectory, movement_time, initiation_time), width="stretch")
    else:
        movement = movement_from_generated_window(
            trajectory, movement_time, initiation_time, WINDOW_MODE
        )
        with st.spinner("Fitting minimum-jerk components..."):
            fit = fit_submovements(
                np.ascontiguousarray(movement, dtype=np.float64).tobytes(),
                movement.shape,
                movement_time,
            )
        st.pyplot(plot_submovements(movement, movement_time, fit), width="stretch")
        st.caption(
            f"Selected {int(fit['summary']['mj_n_components'])} component(s); "
            f"normalized fit error {fit['summary']['mj_fit_error']:.3f}. "
            "Component count is model-order sensitive and is not a cognitive-strategy label."
        )

    st.divider()
    controls = st.columns([1, 1.2, 1])
    n_samples = controls[0].slider("Generated samples", 25, 500, 100, 25)
    output_label = controls[1].selectbox("Generated distribution", list(OUTPUTS)[:6])
    random_seed = controls[2].number_input("Sampling seed", 0, 100000, 2026, 1)
    rng = np.random.default_rng(int(random_seed))
    covariance = np.asarray(stats_for_run["shared_covariance"], dtype=float)
    z = rng.multivariate_normal(latent, covariance, size=n_samples)
    generated = generate_feature_frame(model, norm, z, sp, side, target_speed)
    column = OUTPUTS[output_label]
    values = generated[column].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(values, bins="auto", color=COLORS["teal"], edgecolor="white")
    ax.axvline(values.mean(), color=COLORS["amber"], lw=2, label="mean")
    ax.set_xlabel(output_label)
    ax.set_ylabel("Generated samples")
    ax.set_title(f"Generated {output_label.lower()} distribution", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.download_button(
        "Download generated samples",
        generated.to_csv(index=False).encode(),
        f"{name}_{column}.csv",
        "text/csv",
    )


def validation_tab(
    latent_dim: int,
    empirical: pd.DataFrame,
    generated: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if latent_dim not in (3, 8):
        st.info("All-fold generated validation is precomputed for the compact n=3 and capacity n=8 models.")
        return
    cvae = comparison[
        (comparison.model_family == "cvae") & (comparison.latent_dim == latent_dim)
    ].iloc[0]
    cards = st.columns(5)
    cards[0].metric("Trajectory MSE", f"{cvae.trajectory_mse_subject_balanced_mean:.3f}")
    cards[1].metric("Initiation MAE", f"{cvae.initiation_time_mae_ms_subject_balanced_mean:.1f} ms")
    cards[2].metric("Movement MAE", f"{cvae.movement_time_mae_ms_subject_balanced_mean:.1f} ms")
    cards[3].metric("Mean KS", f"{cvae.mean_ks_mean:.3f}")
    cards[4].metric("MMD", f"{cvae.mean_mmd_rbf_mean:.3f}")

    generated_n = generated[generated.latent_dim == latent_dim]
    subjects = sorted(set(empirical.subject) & set(generated_n.subject))
    pickers = st.columns(2)
    subject = pickers[0].selectbox("Held-out participant", subjects)
    output_label = pickers[1].selectbox("Recorded versus generated", list(OUTPUTS))
    column = OUTPUTS[output_label]
    real = empirical.loc[empirical.subject == subject, column].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().to_numpy()
    simulated = generated_n.loc[generated_n.subject == subject, column].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().to_numpy()

    if len(real) == 0 or len(simulated) == 0:
        st.warning("No finite values are available for this participant/output pair.")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    if column == "mj_n_components":
        support = np.arange(1, 5)
        p = np.array([(real == value).mean() for value in support])
        q = np.array([(simulated == value).mean() for value in support])
        ax.bar(support - .18, p, .36, color=COLORS["navy"], label="recorded query")
        ax.bar(support + .18, q, .36, color=COLORS["amber"], label="generated")
        score = float(jensenshannon(p, q, base=2) ** 2)
        title = f"{subject}: JSD={score:.3f}"
        ax.set_xticks(support)
        ax.set_ylabel("Probability")
    else:
        bins = np.histogram_bin_edges(np.r_[real, simulated], bins="auto")
        ax.hist(real, bins=bins, density=True, alpha=.62, color=COLORS["navy"], label="recorded query")
        ax.hist(simulated, bins=bins, density=True, alpha=.62, color=COLORS["amber"], label="generated")
        score = stats.ks_2samp(real, simulated).statistic
        title = f"{subject}: KS={score:.3f}"
        ax.set_ylabel("Density")
    ax.set_xlabel(output_label)
    ax.set_title(title, loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.caption(
        "Recorded values are disjoint query trials. Generated values use the context fingerprint. "
        "Lower KS/JSD indicates closer distributions."
    )


def benchmark_tab(comparison: pd.DataFrame) -> None:
    metrics = {
        "Trajectory MSE": "trajectory_mse_subject_balanced_mean",
        "Initiation MAE (ms)": "initiation_time_mae_ms_subject_balanced_mean",
        "Movement MAE (ms)": "movement_time_mae_ms_subject_balanced_mean",
        "Mean KS": "mean_ks_mean",
        "MMD": "mean_mmd_rbf_mean",
        "Enrollment accuracy": "fingerprint_balanced_accuracy_mean",
    }
    metric_label = st.selectbox("Benchmark endpoint", list(metrics))
    metric = metrics[metric_label]
    frame = comparison.dropna(subset=[metric]).copy()
    frame["model"] = frame.model_family.map(MODEL_LABELS)
    frame["label"] = [
        model if int(n) == 0 else f"{model} n={int(n)}"
        for model, n in zip(frame.model, frame.latent_dim)
    ]
    frame = frame.sort_values(metric, ascending=metric_label == "Enrollment accuracy")
    fig, ax = plt.subplots(figsize=(8.6, max(4.5, .33 * len(frame))))
    colors = [COLORS["teal"] if family == "cvae" else COLORS["navy"] for family in frame.model_family]
    ax.barh(frame.label, frame[metric], color=colors)
    if metric_label == "Enrollment accuracy":
        ax.axvline(1 / 7, color=COLORS["gray"], ls="--", lw=1.3, label="7-way nominal chance")
        ax.legend(frameon=False)
    ax.set_xlabel(metric_label)
    ax.set_title("Out-of-fold participant-balanced benchmark", loc="left", weight="bold")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    table = frame[["model", "latent_dim", metric]].copy()
    table.columns = ["Model", "n", metric_label]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Lower is better for MSE, MAE, KS, and MMD; higher is better for enrollment. "
        "Spline+PCA is strongest for deterministic reconstruction, while generative endpoints favor higher-capacity latent models."
    )


def association_tab(
    latent_dim: int,
    model: ConditionalVAE,
    norm: NormStats,
    stats_for_run: dict,
    sp: int,
    side: int,
    target_speed: float,
) -> None:
    sample_count = st.slider("Association samples", 250, 1000, 250, 250)
    rng = np.random.default_rng(1147)
    center = np.asarray(stats_for_run["training_center"], dtype=float)
    covariance = np.asarray(stats_for_run["shared_covariance"], dtype=float)
    z = rng.multivariate_normal(center, covariance, size=sample_count)
    generated = generate_feature_frame(model, norm, z, sp, side, target_speed)
    feature_columns = [
        "initiation_time_s",
        "movement_time_s",
        "peak_speed_tracker_units_s",
        "path_length",
        "curvature_index",
        "max_lateral_deviation",
    ]
    correlations = np.empty((latent_dim, len(feature_columns)))
    for i in range(latent_dim):
        for j, feature in enumerate(feature_columns):
            correlations[i, j] = stats.spearmanr(z[:, i], generated[feature]).statistic

    labels = ["initiation", "movement", "peak speed", "path", "curvature", "lateral dev."]
    fig, ax = plt.subplots(figsize=(9.0, max(3.1, .55 * latent_dim + 1.6)))
    image = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(latent_dim), [f"z{i + 1}" for i in range(latent_dim)])
    for i in range(latent_dim):
        for j in range(len(labels)):
            color = "white" if abs(correlations[i, j]) > .55 else "#17212B"
            ax.text(j, i, f"{correlations[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Run-specific decoder associations (Spearman rho)", loc="left", weight="bold")
    fig.colorbar(image, ax=ax, fraction=.025, pad=.025)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.caption(
        "Seed 42, fold 0, current task condition. VAE axes can rotate, permute, or change sign across fits; "
        "these associations describe this decoder coordinate system and do not establish causal control."
    )


def diagnostics_tab(
    condition_summary: pd.DataFrame,
    timing: pd.DataFrame,
    min_jerk: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    st.markdown("### Conditioning diagnostic")
    display = condition_summary[[
        "latent_dim", "cvae_median", "unconditional_vae_median",
        "cvae_better_participants", "wilcoxon_p_holm",
    ]].copy()
    display.columns = ["n", "CVAE median", "U-VAE median", "CVAE better / 28", "Holm p"]
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("No trajectory diagnostic or feature-level condition comparison survived corrected inference; task sliders remain exploratory.")

    st.markdown("### Timing-tail audit")
    cvae = timing[(timing.model_family == "cvae") & (timing.latent_dim.isin([3, 8]))]
    timing_display = cvae[[
        "latent_dim", "training_seed",
        "initiation_time_median_abs_error_ms", "initiation_time_p95_abs_error_ms",
        "initiation_time_max_abs_error_ms",
    ]].copy()
    timing_display.columns = ["n", "seed", "Median abs. error (ms)", "95th percentile (ms)", "Maximum (ms)"]
    st.dataframe(timing_display, hide_index=True, width="stretch")
    st.caption("Participant-balanced MAE is primary. Rare log-timing extrapolations are reported without post-hoc clipping.")

    st.markdown("### Minimum-jerk fidelity and sensitivity")
    mj_display = min_jerk[[
        "latent_dim", "mean_count_total_variation_across_seeds",
        "mean_count_jsd_across_seeds", "mean_ks_mj_first_duration_s_across_seeds",
        "mean_ks_mj_first_amplitude_across_seeds",
    ]].copy()
    mj_display.columns = ["n", "Count TV", "Count JSD", "First-duration KS", "First-amplitude KS"]
    st.dataframe(mj_display, hide_index=True, width="stretch")
    st.dataframe(sensitivity, hide_index=True, width="stretch")
    st.caption("The fitted component count changes materially under alternative filter and model-order assumptions; it is a secondary kinematic descriptor.")

    st.markdown("### Participant behavioural probes")
    probes = read_csv(str(ASSETS / "behavioral_probe_summary.csv"))
    family = st.selectbox("Probe model", sorted(probes.model_family.unique()),
                          format_func=lambda value: MODEL_LABELS[value])
    dim = st.selectbox("Probe dimension", sorted(probes[probes.model_family == family].latent_dim.unique()))
    selected = probes[(probes.model_family == family) & (probes.latent_dim == dim)]
    st.dataframe(selected[["fingerprint", "target", "r2_oof_mean", "r2_fold_mean",
                           "mae_model_mean", "mae_baseline_mean"]], hide_index=True, width="stretch")
    st.caption("Pooled R-squared uses 28 out-of-fold participant summaries per seed. Fold R-squared averages seven-participant scores. Negative values are retained.")
    st.markdown("### Timing-head sensitivity")
    st.dataframe(read_csv(str(ASSETS / "timing_fairness_paired.csv")), hide_index=True, width="stretch")
    st.caption("Validation-only calibration and a common MLP on frozen codes; these sensitivity predictions do not replace the live generator's original timing head.")
    st.markdown("### Finite-sample component reference")
    sampling = read_csv(str(ASSETS / "submovement_sampling_reference.csv"))
    st.dataframe(sampling[sampling.n_generated == 10], hide_index=True, width="stretch")
    st.caption("Matched-size empirical resampling reference, not a formal p-value or a hard KS floor. Generated component fits still use 10 samples per participant and seed.")
    st.markdown("### Event audit")
    st.json(read_json(str(ASSETS / "event_audit.json")), expanded=False)


def protocol_tab(manifest: dict, comparison: pd.DataFrame) -> None:
    rows = [
        ("Cohort", f"{manifest['n_trials']:,} retained condition-2 trials; {manifest['n_subjects']} participants"),
        ("Window", "MAT target motion to recording end (arrival proxy); marker 5 is appearance"),
        ("Evaluation version", manifest.get("evaluation_version", "unversioned")),
        ("Representation", "2-D table plane; 10 Hz low-pass; 100 phase points"),
        ("Timing", "Withheld from the encoder; initiation and movement time decoded separately"),
        ("Outer evaluation", "Four deterministic 17/4/7 participant folds; every participant tested once"),
        ("Optimization repetitions", "Seeds 42, 43, and 44 within each fold/model cell"),
        ("Live generator", "Fold 0, seed 42 reference checkpoints; n=2,3,4,8"),
        ("Generated validation", "All 28 held-out participants; n=3 and n=8"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Protocol item", "Value"]), hide_index=True, width="stretch")
    st.download_button(
        "Download model comparison CSV",
        comparison.to_csv(index=False).encode(),
        "confirmatory_model_comparison.csv",
        "text/csv",
    )
    downloads = st.columns(2)
    st.download_button("Download behavioural probe CSV",
        (ASSETS / "behavioral_probe_summary.csv").read_bytes(),"behavioral_probe_summary.csv","text/csv")
    if REPORT.exists():
        downloads[0].download_button(
            "Download scientific report", REPORT.read_bytes(), REPORT.name, "application/pdf"
        )
    if GUIDE.exists():
        downloads[1].download_button(
            "Download results guide", GUIDE.read_bytes(), GUIDE.name, "application/pdf"
        )


def main() -> None:
    st.set_page_config(page_title="Interception movement explorer", layout="wide")
    apply_style()
    required = [
        ASSETS / "manifest.json",
        ASSETS / "latent_stats.json",
        ASSETS / "subject_fingerprints.csv",
        ASSETS / "empirical_query_features.csv",
        ASSETS / "generated_validation.csv",
        ASSETS / "model_comparison.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        st.error("Confirmatory dashboard assets are missing. Run `python scripts/build_confirmatory_dashboard_assets.py`.")
        return

    manifest = read_json(str(ASSETS / "manifest.json"))
    latent_stats = read_json(str(ASSETS / "latent_stats.json"))
    fingerprints = read_csv(str(ASSETS / "subject_fingerprints.csv"))
    empirical = read_csv(str(ASSETS / "empirical_query_features.csv"))
    generated = read_csv(str(ASSETS / "generated_validation.csv"))
    comparison = read_csv(str(ASSETS / "model_comparison.csv"))
    speed_ranges = read_csv(str(ASSETS / "condition_speed_ranges.csv"))
    condition_summary = read_csv(str(ASSETS / "condition_trajectory_summary.csv"))
    timing = read_csv(str(ASSETS / "timing_outlier_audit.csv"))
    min_jerk = read_csv(str(ASSETS / "minimum_jerk_summary.csv"))
    sensitivity = read_csv(str(ASSETS / "minimum_jerk_sensitivity.csv"))

    st.title("Interception movement strategy explorer")
    st.caption("Frozen strategy-window confirmatory study | out-of-fold participant evaluation")

    st.sidebar.header("Reference model")
    latent_dimensions = [int(value) for value in manifest["latent_dimensions"]]
    latent_dim = int(
        st.sidebar.segmented_control("Latent dimension", latent_dimensions, default=3) or 3
    )
    name = run_name(latent_dim)
    model, norm = load_model(latent_dim)

    st.sidebar.header("Task condition")
    sp = int(st.sidebar.selectbox(
        "Start/speed category", [1, 2, 3], index=1,
        format_func=lambda value: {
            1: "1: 120 / slow", 2: "2: 140 / medium", 3: "3: 160 / fast"
        }[value],
    ))
    side = 1 if st.sidebar.radio("Starting side", ["Left", "Right"], horizontal=True) == "Left" else 2
    speed_row = speed_ranges[speed_ranges.sp == sp].iloc[0]
    target_speed = st.sidebar.slider(
        "Executed target speed",
        float(speed_row.speed_min), float(speed_row.speed_max), float(speed_row.speed_median),
    )
    st.sidebar.warning("Task-condition controls are exploratory; participant-specific conditioning benefit was not validated.")

    sections = [
        "Generate", "Held-out validation", "Benchmarks", "Latent associations",
        "Diagnostics", "Protocol & downloads",
    ]
    section = st.segmented_control(
        "Dashboard section", sections, default="Generate", label_visibility="collapsed"
    ) or "Generate"
    st.divider()
    if section == "Generate":
        generator_tab(
            name, latent_dim, model, norm, latent_stats[name], fingerprints,
            sp, side, target_speed,
        )
    elif section == "Held-out validation":
        validation_tab(latent_dim, empirical, generated, comparison)
    elif section == "Benchmarks":
        benchmark_tab(comparison)
    elif section == "Latent associations":
        association_tab(latent_dim, model, norm, latent_stats[name], sp, side, target_speed)
    elif section == "Diagnostics":
        diagnostics_tab(condition_summary, timing, min_jerk, sensitivity)
    else:
        protocol_tab(manifest, comparison)


if __name__ == "__main__":
    main()
