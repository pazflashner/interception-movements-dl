"""Build the concise advisor-facing story plus a complete figure appendix."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.run_corrected_study import (
    load_per_trial_checkpoint,
    training_latent_noise_covariance,
)
from src.context_query import split_context_query
from src.evaluate import encode_trials, reconstruct
from src.features import movement_from_generated_window
from src.preprocessing import lowpass_filter
from src.submovements import (
    SubmovementConfig,
    decompose_normalized_trajectory,
    minimum_jerk_velocity,
    reconstruct_velocity,
)
from src.train import split_subjects
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import encode_trial_condition
from reports.build_strategy_comparison_pdf import (
    BLUE,
    FIG_DIR as TECHNICAL_FIG_DIR,
    LIGHT,
    NAVY,
    ORANGE,
    RED,
    WINDOW_LABEL,
    build as _unused_build,
    load_data,
    make_figures,
    report_table,
    styles,
)


OUT_DIR = config.STUDY_ROOT / "output" / "advisor_brief"
FIG_DIR = OUT_DIR / "figures"
OUT = OUT_DIR / "Interception_Movement_Advisor_Brief.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def build_extra_figures(data: dict) -> dict[str, Path]:
    results = data["results"]
    generation = data["generation"]
    figures: dict[str, Path] = {}

    # Main decision figure: only the measures needed to select a primary model.
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.15))
    metrics = [
        ("initiation_time_s_r2", "Initiation-time R2", False),
        ("fingerprint_balanced_accuracy", "Enrollment accuracy", False),
        ("mean_ks", "Mean distribution KS", True),
    ]
    for ax, (metric, title, lower_better) in zip(axes, metrics):
        for mode, color in (("movement_only", BLUE), ("go_to_arrival", RED)):
            group = results[results.window_mode == mode].groupby("latent_dim")[metric].agg(["mean", "std"])
            ax.errorbar(
                group.index, group["mean"], yerr=group["std"], marker="o",
                linewidth=2, capsize=3, color=color, label=WINDOW_LABEL[mode],
            )
        if metric.endswith("_r2"):
            ax.axhline(0, color="#64748B", linewidth=1, linestyle="--")
        if metric == "fingerprint_balanced_accuracy":
            ax.axhline(1 / 7, color="#64748B", linewidth=1, linestyle="--", label="chance")
        ax.set_xticks([2, 3, 4, 8])
        ax.set_xlabel("Latent dimensions")
        ax.set_title(title + (" (lower better)" if lower_better else ""), fontsize=10)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=6.5)
    figures["decision"] = savefig("primary_model_decision.png")

    # Representative generated-distribution tradeoff.
    rows = []
    for mode in config.WINDOW_MODES:
        frame = generation[mode].copy()
        frame["latent_dim"] = frame.run.str.extract(r"_z(\d+)_").astype(int)
        rows.append(frame)
    generated = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.15))
    x = np.arange(2)
    width = .18
    offsets = [-1.5, -.5, .5, 1.5]
    palette = ["#93C5FD", BLUE, "#FCA5A5", RED]
    labels = []
    for offset, color, (_, row) in zip(offsets, palette, generated.iterrows()):
        mode = "Strategy" if "go_to_arrival" in row.run else "Execution"
        label = f"{mode} n={int(row.latent_dim)}"
        labels.append(label)
        axes[0].bar(x[0] + offset * width, row.mean_ks_initiation_time_s, width, color=color, label=label)
        axes[1].bar(x[1] + offset * width, row.mean_count_jsd, width, color=color, label=label)
    axes[0].set_xticks([0], ["Initiation time"])
    axes[0].set_ylabel("KS statistic")
    axes[0].set_title("Timing fidelity (lower better)")
    axes[1].set_xticks([1], ["Component count"])
    axes[1].set_ylabel("Jensen-Shannon divergence")
    axes[1].set_title("Execution-structure fidelity (lower better)")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    figures["generation"] = savefig("generated_distribution_tradeoff.png")

    # Dashboard map, kept conceptual so it remains valid across UI revisions.
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.axis("off")
    blocks = [
        (.01, "Protocol\nexecution / strategy"),
        (.205, "Model\nn + seed"),
        (.40, "Task condition\nsp + side + speed"),
        (.595, "Latent controls\ncenter / participant"),
        (.79, "Inspect\ntrajectory / validation"),
    ]
    for x0, label in blocks:
        ax.add_patch(plt.Rectangle((x0, .30), .17, .42, facecolor="#EEF3F6", edgecolor="#176B87", lw=1.5))
        ax.text(x0 + .085, .51, label, ha="center", va="center", fontsize=8.2)
    for index, (x0, _) in enumerate(blocks[:-1]):
        ax.annotate("", xy=(blocks[index + 1][0], .51), xytext=(x0 + .17, .51), arrowprops={"arrowstyle": "->", "color": "#17324D"})
    ax.text(.5, .12, "One dashboard contains both temporal protocols and every tested latent width", ha="center", fontsize=9, color="#176B87")
    figures["dashboard"] = savefig("dashboard_workflow.png")
    return figures


def _decode_batch(model, norm, latent: np.ndarray, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    latent = np.atleast_2d(np.asarray(latent, dtype=np.float32))
    condition = encode_trial_condition(metadata, model.condition_dim)
    conditions = np.repeat(condition[None, :], len(latent), axis=0).astype(np.float32)
    train_mean, train_std, _, _ = norm.torch("cpu")
    with torch.no_grad():
        trajectory_z, timing_z = model.decode(
            torch.as_tensor(latent), torch.as_tensor(conditions)
        )
    channels = model.input_dim // config.NORMALISED_LENGTH
    trajectories = ((trajectory_z * train_std + train_mean).numpy()).reshape(
        len(latent), config.NORMALISED_LENGTH, channels
    )
    timing = norm.denormalise_timing(timing_z.numpy())
    timing[:, 0] = np.maximum(timing[:, 0], 1e-3)
    timing[:, 1] = np.maximum(timing[:, 1], 0.0)
    return trajectories, timing


def _speed(trajectory: np.ndarray, total_time_s: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, max(float(total_time_s), 1e-3), len(trajectory))
    return time, np.linalg.norm(np.gradient(trajectory, time, axis=0), axis=1)


def build_prediction_figures(data: dict) -> dict:
    """Build deterministic held-out examples for reconstruction and generation."""
    window_mode = config.WINDOW_GO_TO_ARRIVAL
    run_name = "cvae_go_to_arrival_z3_seed42"
    trials = project_trials_to_table_plane(
        select_trials_window(data["trials"], window_mode)
    )
    train_trials, _, test_trials = split_subjects(trials, 17, 4, 7, 42)
    model, norm = load_per_trial_checkpoint(
        config.RESULTS_DIR / window_mode / "models" / run_name / "checkpoint.pt",
        "cpu",
    )
    reconstructed, recorded, predicted_timing, recorded_timing = reconstruct(
        model, test_trials, norm, "cpu"
    )
    recorded = recorded.reshape(len(test_trials), config.NORMALISED_LENGTH, 2)
    reconstructed = reconstructed.reshape(len(test_trials), config.NORMALISED_LENGTH, 2)

    subjects = [trial["metadata"]["subject"] for trial in test_trials]
    sp = [trial["metadata"]["sp"] for trial in test_trials]
    side = [trial["metadata"]["side"] for trial in test_trials]
    splits = split_context_query(subjects, sp, side, seed=config.CONTEXT_QUERY_SEED)
    query_indices = np.concatenate([split.query_indices for split in splits])
    trial_mse = np.mean((reconstructed - recorded) ** 2, axis=(1, 2))
    ordered_query = query_indices[np.argsort(trial_mse[query_indices])]
    selected_index = int(ordered_query[len(ordered_query) // 2])
    selected_trial = test_trials[selected_index]
    selected_subject = selected_trial["metadata"]["subject"]
    selected_split = next(split for split in splits if split.subject == selected_subject)

    actual = recorded[selected_index]
    posterior = reconstructed[selected_index]
    actual_timing = recorded_timing[selected_index]
    posterior_timing = predicted_timing[selected_index]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    axes[0].plot(actual[:, 0], actual[:, 1], color="#111827", lw=2.2, label="recorded")
    axes[0].plot(posterior[:, 0], posterior[:, 1], color=RED, lw=1.9, label="posterior reconstruction")
    axes[0].set_xlabel("Lateral x")
    axes[0].set_ylabel("Forward y")
    axes[0].set_title("Table-plane path")
    phase = np.linspace(0, 100, len(actual))
    axes[1].plot(phase, actual[:, 1], color="#111827", lw=2.2, label="recorded")
    axes[1].plot(phase, posterior[:, 1], color=RED, lw=1.9, label="reconstructed")
    axes[1].set_xlabel("Normalized phase (%)")
    axes[1].set_ylabel("Forward y")
    axes[1].set_title("Forward position")
    actual_total = float(actual_timing.sum())
    posterior_total = float(posterior_timing.sum())
    actual_time, actual_speed = _speed(actual, actual_total)
    posterior_time, posterior_speed = _speed(posterior, posterior_total)
    axes[2].plot(actual_time, actual_speed, color="#111827", lw=2.2, label="recorded")
    axes[2].plot(posterior_time, posterior_speed, color=RED, lw=1.9, label="reconstructed")
    axes[2].axvline(float(actual_timing[1]), color="#111827", ls=":", lw=1)
    axes[2].axvline(float(posterior_timing[1]), color=RED, ls=":", lw=1)
    axes[2].set_xlabel("Time since target motion onset (s)")
    axes[2].set_ylabel("Speed")
    axes[2].set_title("Speed; dotted = movement onset")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=6.5)
    reconstruction_path = savefig("heldout_posterior_reconstruction.png")

    mu, _, _, _ = encode_trials(model, test_trials, norm, "cpu")
    context_center = mu[selected_split.context_indices].mean(axis=0)
    covariance = training_latent_noise_covariance(model, train_trials, norm, "cpu")
    rng = np.random.default_rng(2026)
    sampled_latent = rng.multivariate_normal(context_center, covariance, size=30)
    sampled_trajectories, sampled_timing = _decode_batch(
        model, norm, sampled_latent, selected_trial["metadata"]
    )
    center_trajectory, center_timing = _decode_batch(
        model, norm, context_center, selected_trial["metadata"]
    )
    center_trajectory = center_trajectory[0]
    center_timing = center_timing[0]

    timing_median = np.median(sampled_timing, axis=0)
    timing_scale = np.std(sampled_timing, axis=0)
    timing_scale = np.where(timing_scale > 1e-6, timing_scale, 1.0)
    typical_index = int(
        np.argmin(np.sum(((sampled_timing - timing_median) / timing_scale) ** 2, axis=1))
    )
    typical_trajectory = sampled_trajectories[typical_index]
    typical_timing = sampled_timing[typical_index]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for trajectory in sampled_trajectories:
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], color="#FCA5A5", lw=.7, alpha=.35)
    axes[0].plot(actual[:, 0], actual[:, 1], color="#111827", lw=2.2, label="recorded query")
    axes[0].plot(center_trajectory[:, 0], center_trajectory[:, 1], color=RED, lw=2.0, label="context-mean generation")
    axes[0].set_xlabel("Lateral x")
    axes[0].set_ylabel("Forward y")
    axes[0].set_title("Condition-matched paths")
    for trajectory, timing in zip(sampled_trajectories, sampled_timing):
        time = np.linspace(0, float(timing.sum()), len(trajectory))
        axes[1].plot(time, trajectory[:, 1], color="#FCA5A5", lw=.7, alpha=.30)
    axes[1].plot(
        np.linspace(0, actual_total, len(actual)), actual[:, 1],
        color="#111827", lw=2.2, label="recorded query",
    )
    axes[1].plot(
        np.linspace(0, float(center_timing.sum()), len(center_trajectory)),
        center_trajectory[:, 1], color=RED, lw=2.0, label="context mean",
    )
    axes[1].set_xlabel("Time since target motion onset (s)")
    axes[1].set_ylabel("Forward y")
    axes[1].set_title("Forward position and timing")
    axes[2].scatter(sampled_timing[:, 1], sampled_timing[:, 0], color="#FCA5A5", s=20, alpha=.7, label="generated samples")
    axes[2].scatter(actual_timing[1], actual_timing[0], marker="*", s=90, color="#111827", label="recorded query")
    axes[2].scatter(center_timing[1], center_timing[0], marker="D", s=45, color=RED, label="context mean")
    axes[2].set_xlabel("Initiation time (s)")
    axes[2].set_ylabel("Movement time (s)")
    axes[2].set_title("Generated timing spread")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=6.3)
    axes[2].legend(frameon=False, fontsize=5.8)
    generation_path = savefig("fingerprint_conditioned_generation.png")

    canonical_trial = next(
        trial
        for trial in data["trials"]
        if trial["metadata"]["trial_id"] == selected_trial["metadata"]["trial_id"]
    )
    raw_movement = np.asarray(canonical_trial["pos_filtered"], dtype=float)[
        canonical_trial["move_start_idx"] : canonical_trial["move_end_idx"] + 1, :2
    ]
    recorded_row = data["sub"].set_index("trial_id").loc[
        selected_trial["metadata"]["trial_id"]
    ]
    recorded_parameters = np.asarray(json.loads(recorded_row.mj_parameters_json), dtype=float)
    recorded_time = np.arange(len(raw_movement), dtype=float) / config.RECORDING_HZ
    recorded_speed = np.linalg.norm(
        np.gradient(raw_movement, 1.0 / config.RECORDING_HZ, axis=0), axis=1
    )
    recorded_fit_time = np.arange(
        0.0,
        max(recorded_time[-1], np.max(recorded_parameters[:, 0] + recorded_parameters[:, 1]))
        + 0.5 / config.RECORDING_HZ,
        1.0 / config.RECORDING_HZ,
    )
    recorded_sum = np.linalg.norm(
        reconstruct_velocity(recorded_fit_time, recorded_parameters), axis=1
    )

    generated_movement = movement_from_generated_window(
        typical_trajectory,
        float(typical_timing[0]),
        float(typical_timing[1]),
        window_mode,
    )
    generated_fit = decompose_normalized_trajectory(
        generated_movement,
        float(typical_timing[0]),
        SubmovementConfig(restarts=1, max_nfev=300),
        f"advisor-{selected_trial['metadata']['trial_id']}",
    ).selected
    generated_sample_hz = max(
        (len(generated_movement) - 1) / max(float(typical_timing[0]), 1e-3), 20.0
    )
    generated_filtered = lowpass_filter(
        generated_movement,
        cutoff=min(config.LOWPASS_CUTOFF_HZ, 0.45 * generated_sample_hz),
        fs=generated_sample_hz,
    )
    generated_time = np.arange(len(generated_filtered), dtype=float) / generated_sample_hz
    generated_speed = np.linalg.norm(
        np.gradient(generated_filtered, 1.0 / generated_sample_hz, axis=0), axis=1
    )
    generated_sum = np.linalg.norm(generated_fit.reconstructed_velocity, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15))
    axes[0].plot(recorded_time, recorded_speed, color="#111827", lw=2.0, label="recorded movement")
    axes[0].plot(recorded_fit_time, recorded_sum, color=RED, lw=1.7, label="component sum")
    for index, row in enumerate(recorded_parameters):
        component = minimum_jerk_velocity(recorded_fit_time, row[0], row[1], row[2:4])
        axes[0].plot(recorded_fit_time, np.linalg.norm(component, axis=1), ls="--", lw=1.0, label=f"component {index + 1}")
    recorded_label = "component" if len(recorded_parameters) == 1 else "components"
    axes[0].set_title(f"Recorded query: {len(recorded_parameters)} {recorded_label}")
    axes[1].plot(generated_time, generated_speed, color="#111827", lw=2.0, label="generated movement")
    axes[1].plot(generated_fit.time, generated_sum, color=RED, lw=1.7, label="component sum")
    for index, row in enumerate(generated_fit.parameters):
        component = minimum_jerk_velocity(generated_fit.time, row[0], row[1], row[2:4])
        axes[1].plot(generated_fit.time, np.linalg.norm(component, axis=1), ls="--", lw=1.0, label=f"component {index + 1}")
    generated_label = "component" if generated_fit.n_components == 1 else "components"
    axes[1].set_title(f"Typical generated sample: {generated_fit.n_components} {generated_label}")
    recorded_end = max(
        float(recorded_time[-1]),
        float(np.max(recorded_parameters[:, 0] + recorded_parameters[:, 1])),
    )
    generated_end = max(
        float(generated_time[-1]),
        float(np.max(generated_fit.parameters[:, 0] + generated_fit.parameters[:, 1])),
    )
    axes[0].set_xlim(0, recorded_end + .02)
    axes[1].set_xlim(0, generated_end + .02)
    for ax in axes:
        ax.set_xlabel("Time since movement onset (s)")
        ax.set_ylabel("Speed (tracker units/s)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=6.2, ncol=2)
    submovement_path = savefig("recorded_vs_generated_submovements.png")

    return {
        "reconstruction": reconstruction_path,
        "generation": generation_path,
        "submovements": submovement_path,
        "trial_id": selected_trial["metadata"]["trial_id"],
        "subject": selected_subject,
        "response": selected_trial["metadata"].get("responseText", ""),
        "sp": int(selected_trial["metadata"]["sp"]),
        "side": int(selected_trial["metadata"]["side"]),
        "target_speed": float(selected_trial["metadata"]["target_speed_screen_s"]),
        "reconstruction_mse": float(trial_mse[selected_index]),
        "actual_movement_s": float(actual_timing[0]),
        "actual_initiation_s": float(actual_timing[1]),
        "predicted_movement_s": float(posterior_timing[0]),
        "predicted_initiation_s": float(posterior_timing[1]),
        "n_context": int(len(selected_split.context_indices)),
        "n_query": int(len(selected_split.query_indices)),
        "n_generated": int(len(sampled_trajectories)),
        "recorded_components": int(len(recorded_parameters)),
        "recorded_fit_error": float(recorded_row.mj_fit_error),
        "generated_components": int(generated_fit.n_components),
        "generated_fit_error": float(generated_fit.normalized_error),
    }


def image_grid(paths: list[Path], width: float, height: float, columns: int = 2) -> Table:
    cells = [Image(str(path), width=width, height=height) for path in paths]
    rows = [cells[index:index + columns] for index in range(0, len(cells), columns)]
    while len(rows[-1]) < columns:
        rows[-1].append("")
    return Table(rows, colWidths=[width + 3 * mm] * columns)


def build(
    data: dict,
    technical: dict[str, Path],
    extra: dict[str, Path],
    prediction: dict,
) -> None:
    s = styles()
    P = lambda text, style="body": Paragraph(text, s[style])
    results = data["results"]
    grouped = results.groupby(["window_mode", "latent_dim"]).mean(numeric_only=True)
    go3 = grouped.loc[("go_to_arrival", 3)]
    go8 = grouped.loc[("go_to_arrival", 8)]
    movement8 = grouped.loc[("movement_only", 8)]
    movement_generation = data["generation"]["movement_only"].set_index("run")
    strategy_generation = data["generation"]["go_to_arrival"].set_index("run")
    associations = data["associations"]
    go_n3_seed42 = associations[
        (associations.window_mode == "go_to_arrival")
        & (associations.latent_dim == 3)
        & (associations.seed == 42)
    ]
    go_trial = go_n3_seed42[go_n3_seed42.level == "trial_within_subject_partial"]
    go_subject = go_n3_seed42[go_n3_seed42.level == "subject_context_to_query"]

    story: list = []
    story += [
        Spacer(1, 13 * mm),
        P("Interception-Movement Fingerprints", "title"),
        P("Advisor brief: temporal representation, CVAE findings, latent interpretation, and interactive review", "subtitle"),
        P("Seman Libbiss and Paz Flashner | Deep Learning Workshop | Prof. Jason Friedman | Advisor: Moni", "subtitle"),
        Spacer(1, 3 * mm),
        P("Question addressed after the submitted proposal", "h1"),
        P("Can a low-dimensional conditional VAE preserve participant-specific interception behavior while generating held-out distributions of movement shape, initiation, duration, and submovement structure?", "callout"),
        P("Recommendation", "h2"),
        P("Use the target-motion-onset -> arrival CVAE with n=3 as the primary low-dimensional model. It is the smallest tested representation with stable held-out initiation and movement-time prediction, and it retains the waiting interval required for strategy analysis. Use n=8 as a capacity reference. Keep movement-onset -> arrival as an execution-focused control, not as a second equal final model. Representative held-out reconstruction and fingerprint-conditioned generation are shown in Appendix A4-A5."),
        report_table([
            ["Primary conclusion", "Evidence"],
            ["n=3 is the interpretable strategy model", f"Initiation R2={go3.initiation_time_s_r2:.2f}; movement-time R2={go3.movement_time_s_r2:.2f}; all values are held-out participant means across three seeds."],
            ["n=8 is the capacity bound", f"Strategy-window mean KS improves to {go8.mean_ks:.3f}; enrollment reaches {100*go8.fingerprint_balanced_accuracy:.1f}%."],
            ["Execution-only remains informative", f"Its n=8 enrollment reaches {100*movement8.fingerprint_balanced_accuracy:.1f}% versus 14.3% chance and its generated component-count distribution is more faithful."],
            ["The full goal is not solved", "No tested low-dimensional model exactly reproduces every held-out behavioral distribution."],
        ], [48 * mm, 122 * mm]),
        PageBreak(),
    ]

    story += [
        P("1. Why two temporal protocols were compared", "h1"),
        P("Phase resampling is needed to provide a fixed-size neural input, but it removes physical duration. Following Jason's concern, physical timing is withheld from the encoder and predicted separately by the decoder. A second issue is the start event: beginning at finger movement onset removes the participant's decision to wait."),
        Image(str(technical["windows"]), width=165 * mm, height=65 * mm),
        report_table([
            ["Protocol", "Scientific content", "Role in final study"],
            ["Movement onset -> arrival", "Physical execution and corrections", "Execution control"],
            ["Target motion onset -> arrival", "Waiting, initiation, execution, corrections", "Primary strategy model"],
        ], [47 * mm, 73 * mm, 50 * mm]),
        P("Both protocols use the same 4,732 retained condition-2 trials, x-y table plane, 10 Hz filter, task conditions, participant split, dimensions, seeds, and evaluation. Movement time and initiation time are withheld from the encoder and decoded separately. Appendix A1 contains the full cohort audit and temporal-window illustration."),
        P("Trial audit", "h2"),
        P("No recording ended before target motion began. All 48 Too early labels contain a completed arrival after the target started and are retained. Four no-arrival timeouts and 27 arrivals more than one second after the target window are excluded under the current rerunnable rule."),
        PageBreak(),
    ]

    story += [
        P("2. Model and evaluation", "h1"),
        P("The CVAE encodes 100 x-y phase samples plus task condition (start/speed category, starting side, and exact executed target speed) into a Gaussian latent distribution. The decoder reconstructs trajectory and predicts positive initiation and movement times. KL regularization supports continuous latent interpolation and participant-level averaging within one trained model."),
        P("Appendix A4 shows posterior reconstruction, where the encoder has seen the held-out query trajectory. Appendix A5 separately shows fingerprint-conditioned generation, where the query trajectory is not supplied to the model. This distinction is essential when interpreting reconstruction error versus predictive generation."),
        report_table([
            ["Stage", "Protocol"],
            ["Training", "17 participants; subject-balanced batches"],
            ["Hyperparameter selection", "4 separate validation participants"],
            ["Final evaluation", "7 untouched participants"],
            ["Fingerprint enrollment", "Context-half latent mean for each held-out participant"],
            ["Fingerprint query", "Disjoint, condition-stratified trial half"],
            ["Model sweep", "Both temporal protocols x n=2/3/4/8 x seeds 42/43/44 = 24 runs"],
        ], [52 * mm, 118 * mm]),
        P("The fingerprint-identification result is closed-set enrollment: context movements are available for each test participant. It is not zero-shot identification of a person with no recorded movement."),
        P("Baselines", "h2"),
        P("K-Means with k fixed to 28 is above a 200-permutation null but weak in absolute terms (trajectory ARI about 0.05-0.06). A spline fitted directly to each test trial is descriptive, not predictive. The fair learned compression baseline is spline coefficients followed by PCA fitted only on training participants. Appendix A2 contains the baseline and full repeated-seed panels."),
        PageBreak(),
    ]

    story += [
        P("3. Main comparison and model choice", "h1"),
        Image(str(extra["decision"]), width=165 * mm, height=64 * mm),
        report_table([
            ["Model", "Initiation R2", "Movement R2", "Enrollment", "Mean KS"],
            ["Strategy n=3", f"{go3.initiation_time_s_r2:.2f}", f"{go3.movement_time_s_r2:.2f}", f"{100*go3.fingerprint_balanced_accuracy:.1f}%", f"{go3.mean_ks:.3f}"],
            ["Strategy n=8", f"{go8.initiation_time_s_r2:.2f}", f"{go8.movement_time_s_r2:.2f}", f"{100*go8.fingerprint_balanced_accuracy:.1f}%", f"{go8.mean_ks:.3f}"],
            ["Execution n=8", f"{movement8.initiation_time_s_r2:.2f}", f"{movement8.movement_time_s_r2:.2f}", f"{100*movement8.fingerprint_balanced_accuracy:.1f}%", f"{movement8.mean_ks:.3f}"],
        ], [42 * mm, 30 * mm, 31 * mm, 34 * mm, 33 * mm]),
        P("n=2 is too restrictive: initiation-time R2 is unstable and strongly negative in both protocols. n=4 is a useful transition point but adds little to the scientific story. n=3 is therefore the low-dimensional recommendation, while n=8 shows the attainable capacity with this architecture and dataset."),
        P("There is room for both temporal protocols only because they answer different questions. The strategy protocol is primary. The execution protocol remains a control that reveals what is gained or lost by including the waiting interval. Appendix A4 provides a held-out trajectory-level example behind the aggregate reconstruction metrics."),
        PageBreak(),
    ]

    story += [
        P("4. Generated distributions: the two protocols are complementary", "h1"),
        Image(str(extra["generation"]), width=160 * mm, height=64 * mm),
        report_table([
            ["Representative seed-42 model", "Initiation KS", "Component-count JSD"],
            ["Execution n=3", f"{movement_generation.loc['cvae_movement_only_z3_seed42'].mean_ks_initiation_time_s:.3f}", f"{movement_generation.loc['cvae_movement_only_z3_seed42'].mean_count_jsd:.3f}"],
            ["Execution n=8", f"{movement_generation.loc['cvae_movement_only_z8_seed42'].mean_ks_initiation_time_s:.3f}", f"{movement_generation.loc['cvae_movement_only_z8_seed42'].mean_count_jsd:.3f}"],
            ["Strategy n=3", f"{strategy_generation.loc['cvae_go_to_arrival_z3_seed42'].mean_ks_initiation_time_s:.3f}", f"{strategy_generation.loc['cvae_go_to_arrival_z3_seed42'].mean_count_jsd:.3f}"],
            ["Strategy n=8", f"{strategy_generation.loc['cvae_go_to_arrival_z8_seed42'].mean_ks_initiation_time_s:.3f}", f"{strategy_generation.loc['cvae_go_to_arrival_z8_seed42'].mean_count_jsd:.3f}"],
        ], [76 * mm, 47 * mm, 47 * mm]),
        P("The strategy window is substantially better for initiation-time distributions because waiting is represented. The execution window is better for generated component-count distributions because all model capacity is focused on physical movement. Continuous outputs use KS/Wasserstein; discrete component count uses JSD/total variation. Appendix A5 shows the actual fingerprint-conditioned trajectory and timing spread for a disjoint held-out query condition."),
        P("Minimum-jerk components", "h2"),
        Spacer(1, 2 * mm),
        P("Each component is a fitted smooth velocity primitive with onset, duration, and 2-D displacement. Component count, secondary amplitude fraction, and overlap describe kinematic organization. They are compatible with single or corrective movement organization but do not by themselves prove feedforward, feedback, hesitation, or regret. Components are fitted downstream of the decoder; they are not direct neural-network outputs. Appendix A6 shows this decomposition for recorded and generated movement."),
        PageBreak(),
    ]

    trial_heatmap = config.RESULTS_DIR / "latent_associations" / "heatmaps" / "go_to_arrival_z3_trial_within_subject_partial.png"
    subject_heatmap = config.RESULTS_DIR / "latent_associations" / "heatmaps" / "go_to_arrival_z3_subject_context_to_query.png"
    story += [
        P("5. What the n=3 strategy latent represents", "h1"),
        P("Trial-level correlations residualize participant, start category, side, and exact speed. Subject-level correlations compare context fingerprints with means from disjoint query trials. The heatmaps below use the pre-specified seed-42 n=3 strategy model."),
        image_grid([trial_heatmap, subject_heatmap], 78 * mm, 59 * mm),
        report_table([
            ["Strong association", "Spearman rho", "Interpretation boundary"],
            ["Trial z3 vs initiation time", "+0.57", "Within-participant, condition-adjusted"],
            ["Trial z1 vs initiation time", "+0.55", "Within-participant, condition-adjusted"],
            ["Trial z1 vs movement time", "-0.49", "Within-participant, condition-adjusted"],
            ["Subject z3 vs query initiation mean", "+0.61", "N=28 context-to-query association"],
        ], [67 * mm, 34 * mm, 69 * mm]),
        P(f"Within this model, {int(go_trial.fdr_reject_0_05.sum())}/24 trial-level and {int(go_subject.fdr_reject_0_05.sum())}/24 subject-level associations survive Benjamini-Hochberg correction. With thousands of trials, effect size matters more than significance count. Latent axes can rotate, reflect, or swap across seeds, so z1/z2/z3 are model-specific coordinates, not universal psychological variables. Appendix A8-A11 contains every seed-42 heatmap for both protocols and all latent widths."),
        PageBreak(),
    ]

    story += [
        P("6. One dashboard contains both protocols", "h1"),
        Image(str(extra["dashboard"]), width=165 * mm, height=56 * mm),
        report_table([
            ["Control", "Available choices"],
            ["Temporal protocol", "Movement onset -> arrival; target motion onset -> arrival"],
            ["Latent width", "n=2, 3, 4, 8"],
            ["Training seed", "42, 43, 44 in the full dashboard"],
            ["Task condition", "Start/speed category, side, exact executed speed"],
            ["Fingerprint source", "Training-population center or an enrolled held-out participant"],
            ["Generated output", "Trajectory, speed, minimum-jerk decomposition, sampled distributions"],
            ["Evidence tabs", "Held-out validation, repeated-seed comparison, latent heatmaps, protocol"],
        ], [48 * mm, 122 * mm]),
        P("Use", "h2"),
        P("1. Extract the dashboard ZIP and run setup_and_launch.bat once.<br/>2. Select the strategy protocol and n=3 for the primary scientific view.<br/>3. Choose a participant context fingerprint or the population center.<br/>4. Change one latent slider while holding task condition fixed.<br/>5. Inspect trajectory and timing, then use Held-out validation before interpreting the generated distribution.<br/>6. Use n=8 to inspect the capacity bound, not as an eight-trait explanation."),
        P("The full dashboard includes every seed. The email-sized dashboard uses seed 42 for live generation but retains all-seed aggregate comparison tables."),
        PageBreak(),
    ]

    story += [
        P("7. Conclusions, assumptions, and requested feedback", "h1"),
        P("What the current evidence supports", "h2"),
        P("- A low-dimensional strategy-inclusive CVAE can jointly represent trajectory shape and predict withheld timing.<br/>- n=3 is the smallest stable candidate for the requested controllable fingerprint.<br/>- Participant information transfers after context enrollment.<br/>- Waiting is informative for initiation strategy; execution-only remains useful for movement organization.<br/>- Generated distributions are partially, not exactly, reproduced."),
        P("Assumptions made for the current analysis", "h2"),
        report_table([
            ["Item", "Current decision to confirm"],
            ["Completion event", "A non-empty MAT pressedTime indicates arrival at the interception site"],
            ["Too early", "Retain: all 48 arrivals occur after target motion begins"],
            ["Late threshold", "Exclude arrival more than 1 s after the target window"],
            ["Go signal", "Executed MAT dotArray target-motion onset defines time zero"],
            ["Task plane", "x-y table plane; z treated as off-plane variation"],
            ["Submovement constraints", "100 ms minimum duration; 50 ms minimum onset spacing"],
            ["Component order", "Smallest model reaching normalized error <=0.05, then <=0.10; BIC retained as sensitivity output"],
            ["Units and target scaling", "Report tracker units until physical conversion and screen scaling are confirmed"],
            ["Not-fixating label", "Keep separate in condition 2 until its interpretation is confirmed"],
        ], [51 * mm, 119 * mm], 7.6),
        P("Feedback requested", "h2"),
        P("We would appreciate Jason's confirmation or correction of the task-specific assumptions above, and comments from both Jason and Moni on the temporal comparison, CVAE design, evaluation protocol, and interpretation boundaries. If an event or component rule changes, the corresponding preprocessing or submovement stage can be rerun without changing the overall study design."),
        PageBreak(),
    ]

    # Appendix cover and complete secondary figures.
    story += [
        P("Appendix", "title"),
        P("Complete secondary figures and numerical sweeps", "subtitle"),
        P("The main text deliberately reports only the models needed for the scientific decision. The appendix preserves the complete audit trail, baseline comparison, all latent widths, and every seed-42 association heatmap."),
        PageBreak(),
        P("A1. Data audit and temporal windows", "h1"),
        Image(str(technical["audit"]), width=165 * mm, height=67 * mm),
        Image(str(technical["windows"]), width=165 * mm, height=65 * mm),
        PageBreak(),
        P("A2. Baselines and full repeated-seed result panels", "h1"),
        Image(str(technical["baselines"]), width=165 * mm, height=69 * mm),
        Image(str(technical["models"]), width=165 * mm, height=122 * mm),
        PageBreak(),
        P("A3. Recorded minimum-jerk structure", "h1"),
        Image(str(technical["sub_examples"]), width=165 * mm, height=64 * mm),
        Image(str(technical["submovements"]), width=165 * mm, height=67 * mm),
        PageBreak(),
        P("A4. Held-out posterior trajectory reconstruction", "h1"),
        P("This example is selected deterministically as the query trial nearest the median n=3 strategy-model reconstruction error. The encoder receives this held-out trajectory, and the decoder reconstructs it from the posterior mean. This is a compression/generalization check, not fingerprint-only generation."),
        Image(str(prediction["reconstruction"]), width=165 * mm, height=59 * mm),
        report_table([
            ["Example", "Recorded", "Predicted"],
            ["Participant / trial", f"{prediction['subject']} / {prediction['trial_id']}", "n=3 strategy, seed 42"],
            ["Initiation time", f"{prediction['actual_initiation_s']:.3f} s", f"{prediction['predicted_initiation_s']:.3f} s"],
            ["Movement time", f"{prediction['actual_movement_s']:.3f} s", f"{prediction['predicted_movement_s']:.3f} s"],
            ["Trajectory MSE", "-", f"{prediction['reconstruction_mse']:.4f} tracker units squared"],
        ], [46 * mm, 63 * mm, 61 * mm]),
        P("The dotted speed-profile lines mark recorded and predicted movement onset. Because posterior reconstruction uses the query trajectory in the encoder, it should not be confused with the context-fingerprint generation below.", "small"),
        PageBreak(),
        P("A5. Fingerprint-conditioned held-out generation", "h1"),
        P("The participant fingerprint is the mean latent code from context trials only. Thirty latent samples are drawn using covariance estimated from training participants and decoded under the selected query trial's task condition. The query trajectory is shown only for comparison and is never passed to this generation step."),
        Image(str(prediction["generation"]), width=165 * mm, height=59 * mm),
        report_table([
            ["Generation context", "Value"],
            ["Held-out participant", prediction["subject"]],
            ["Disjoint context / query trials", f"{prediction['n_context']} / {prediction['n_query']}"],
            ["Condition", f"sp={prediction['sp']}, side={prediction['side']}, executed target speed={prediction['target_speed']:.3f}"],
            ["Generated samples", prediction["n_generated"]],
            ["Selection rule", "Same median-reconstruction query as A4; no best-case visual selection"],
        ], [60 * mm, 110 * mm]),
        P("Thin red lines and dots are generated samples; the solid red line/diamond is the context-mean decode; black is the disjoint recorded query. Distribution-level conclusions still rely on all held-out query trials and the KS/JSD results, not this single visual example.", "small"),
        PageBreak(),
        P("A6. Minimum-jerk analysis after generation", "h1"),
        P("The CVAE decoder outputs trajectory and timing. Minimum-jerk components are then fitted to the physical movement interval recovered from that output. The generated example is selected only by proximity to the median generated initiation and movement times, not by similarity to the recorded trial."),
        Image(str(prediction["submovements"]), width=165 * mm, height=64 * mm),
        report_table([
            ["Downstream fit", "Components", "Normalized fit error"],
            ["Recorded held-out query", prediction["recorded_components"], f"{prediction['recorded_fit_error']:.3f}"],
            ["Typical fingerprint-generated sample", prediction["generated_components"], f"{prediction['generated_fit_error']:.3f}"],
        ], [80 * mm, 40 * mm, 50 * mm]),
        P("Component count describes the number of smooth kinematic primitives required by the selected error rule. It does not identify a cognitive strategy by itself. Population-level recorded examples and distributions are shown in A3; full generated-versus-recorded component-count results are summarized in the main text.", "small"),
        PageBreak(),
        P("A7. Complete repeated-seed numerical sweep", "h1"),
        report_table([
            ["Protocol", "n", "Move MSE", "Move R2", "Initiation R2", "Enrollment", "Mean KS"],
            *[
                [
                    "Strategy" if mode == "go_to_arrival" else "Execution",
                    n,
                    f"{grouped.loc[(mode, n)].movement_reconstruction_mse_tracker_units2:.3f}",
                    f"{grouped.loc[(mode, n)].movement_time_s_r2:.2f}",
                    f"{grouped.loc[(mode, n)].initiation_time_s_r2:.2f}",
                    f"{100*grouped.loc[(mode, n)].fingerprint_balanced_accuracy:.1f}%",
                    f"{grouped.loc[(mode, n)].mean_ks:.3f}",
                ]
                for mode in config.WINDOW_MODES for n in [2, 3, 4, 8]
            ],
        ], [34 * mm, 13 * mm, 27 * mm, 25 * mm, 29 * mm, 23 * mm, 19 * mm], 7.2),
        P("Values are means across seeds 42, 43, and 44 on held-out participants. Raw full-window MSE is not compared across temporal protocols because the target intervals differ; the table reports movement-region MSE."),
        PageBreak(),
    ]

    heatmap_dir = config.RESULTS_DIR / "latent_associations" / "heatmaps"
    for page_index, (mode, level, title) in enumerate([
        ("go_to_arrival", "trial_within_subject_partial", "A8. Strategy protocol: trial-level partial associations"),
        ("go_to_arrival", "subject_context_to_query", "A9. Strategy protocol: subject context-to-query associations"),
        ("movement_only", "trial_within_subject_partial", "A10. Execution protocol: trial-level partial associations"),
        ("movement_only", "subject_context_to_query", "A11. Execution protocol: subject context-to-query associations"),
    ]):
        paths = [heatmap_dir / f"{mode}_z{n}_{level}.png" for n in [2, 3, 4, 8]]
        story += [
            P(title, "h1"),
            image_grid(paths, 80 * mm, 57 * mm),
            P("Seed 42. Asterisks survive within-model Benjamini-Hochberg correction. Compare association patterns, not latent-axis labels, across independently trained models.", "small"),
            PageBreak(),
        ]

    story += [
        P("A12. References and reproducibility", "h1"),
        P("Brenner, E. and Smeets, J. B. J Neurophysiol (2018), doi:10.1152/jn.00517.2018.<br/>Flash, T. and Hogan, N. J Neurosci (1985), minimum-jerk movement model.<br/>Friedman, J. submovements repository, commit 9c2f40c, github.com/JasonFriedman/submovements.<br/>Kingma, D. P. and Welling, M. Auto-Encoding Variational Bayes, ICLR (2014).<br/>Sohn, K., Lee, H. and Yan, X. Conditional deep generative models, NeurIPS (2015).<br/>Slowinski, P. et al. Dynamic similarity and individual motor signatures, J R Soc Interface (2016), doi:10.1098/rsif.2015.1093."),
        P("All numerical values and figures are generated from the current study artifacts. The code, split, seeds, retained-trial list, checkpoints, and validation tables are preserved in the project directory. No raw Dropbox trajectory or MAT file is included in the advisor package.", "small"),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#607582"))
        canvas.drawString(20 * mm, 11 * mm, "Interception-movement advisor brief")
        canvas.drawRightString(190 * mm, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Interception-Movement Fingerprints - Advisor Brief",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    data = load_data()
    technical = make_figures(data)
    extra = build_extra_figures(data)
    prediction = build_prediction_figures(data)
    build(data, technical, extra, prediction)
    print(OUT)


if __name__ == "__main__":
    main()
