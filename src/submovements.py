"""Minimum-jerk submovement decomposition for the interception study.

The parameterization and normalized velocity error follow Prof. Jason
Friedman's GPL-3.0 ``submovements`` repository (commit 9c2f40c, inspected
2026-08-10). This implementation keeps the same scientific model while making
the temporal constraints configurable for the shorter interception reaches.

Each 2-D component is ``[onset_s, duration_s, lateral_displacement,
forward_displacement]``. The measured velocity is modeled as the sum of the
component minimum-jerk velocity profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import least_squares

import config
from src.preprocessing import lowpass_filter


@dataclass(frozen=True)
class SubmovementConfig:
    max_components: int = 4
    min_duration_s: float = 0.100
    min_onset_spacing_s: float = 0.050
    max_duration_s: float = 1.000
    error_threshold: float = 0.050
    fallback_error_threshold: float = 0.100
    restarts: int = 4
    max_nfev: int = 700
    lateral_bounds: tuple[float, float] = (-5.0, 5.0)
    forward_bounds: tuple[float, float] = (0.0, 20.0)
    sample_hz: float = config.RECORDING_HZ
    cutoff_hz: float = config.LOWPASS_CUTOFF_HZ


@dataclass
class ComponentFit:
    n_components: int
    parameters: np.ndarray
    normalized_error: float
    bic: float
    success: bool
    nfev: int
    reconstructed_velocity: np.ndarray
    time: np.ndarray


@dataclass
class SubmovementResult:
    selected: ComponentFit
    selected_bic: ComponentFit
    fits: dict[int, ComponentFit]

    def summary(self) -> dict[str, float | str]:
        p = self.selected.parameters
        amplitudes = np.linalg.norm(p[:, 2:4], axis=1)
        if len(p) > 1:
            relative = 100.0 * np.diff(p[:, 0]) / np.maximum(p[:-1, 1], 1e-8)
            overlaps = []
            for first, second in zip(p[:-1], p[1:]):
                t1, d1 = first[:2]
                t2, d2 = second[:2]
                overlap = max(0.0, min(100.0, 100.0 * (t1 + d1 - t2) / max(t2 + d2 - t1, 1e-8)))
                overlaps.append(overlap)
            mean_relative = float(np.mean(relative))
            mean_overlap = float(np.mean(overlaps))
            second_onset = float(p[1, 0])
        else:
            mean_relative = np.nan
            mean_overlap = 0.0
            second_onset = np.nan

        if len(p) == 1:
            pattern = "single_component"
        elif mean_relative < 100.0:
            pattern = "overlapping_components"
        else:
            pattern = "sequential_components"

        return {
            "mj_n_components": float(len(p)),
            "mj_n_components_bic": float(self.selected_bic.n_components),
            "mj_fit_error": float(self.selected.normalized_error),
            "mj_bic": float(self.selected.bic),
            "mj_first_onset_s": float(p[0, 0]),
            "mj_second_onset_s": second_onset,
            "mj_first_duration_s": float(p[0, 1]),
            "mj_mean_duration_s": float(np.mean(p[:, 1])),
            "mj_first_amplitude": float(amplitudes[0]),
            "mj_total_amplitude": float(np.sum(amplitudes)),
            "mj_secondary_amplitude_fraction": float(np.sum(amplitudes[1:]) / max(np.sum(amplitudes), 1e-8)),
            "mj_mean_overlap_pct": mean_overlap,
            "mj_mean_relative_onset_pct": mean_relative,
            "mj_pattern": pattern,
        }


def minimum_jerk_velocity(
    time: np.ndarray,
    onset_s: float,
    duration_s: float,
    displacement: np.ndarray,
) -> np.ndarray:
    """Evaluate a 2-D minimum-jerk velocity component."""
    time = np.asarray(time, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    u = (time - onset_s) / max(duration_s, 1e-8)
    active = (u > 0.0) & (u < 1.0)
    basis = np.zeros_like(time)
    ua = u[active]
    basis[active] = (30.0 * ua**2 - 60.0 * ua**3 + 30.0 * ua**4) / duration_s
    return basis[:, None] * displacement[None, :]


def reconstruct_velocity(time: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    out = np.zeros((len(time), 2), dtype=float)
    for onset, duration, lateral, forward in np.asarray(parameters).reshape(-1, 4):
        out += minimum_jerk_velocity(time, onset, duration, np.array([lateral, forward]))
    return out


def _seed_from_text(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def _prepare_velocity(position_xy: np.ndarray, sample_hz: float, cutoff_hz: float) -> tuple[np.ndarray, np.ndarray]:
    position_xy = np.asarray(position_xy, dtype=float)
    cutoff = min(cutoff_hz, 0.45 * sample_hz)
    filtered = lowpass_filter(position_xy, cutoff=cutoff, fs=sample_hz)
    velocity = np.gradient(filtered, 1.0 / sample_hz, axis=0)
    time = np.arange(len(filtered), dtype=float) / sample_hz
    return time, velocity


def fit_component_count(
    time: np.ndarray,
    velocity: np.ndarray,
    n_components: int,
    cfg: SubmovementConfig,
    seed: int,
) -> ComponentFit | None:
    """Fit one candidate component count with deterministic random restarts."""
    movement_end = float(time[-1])
    onset_upper = max(movement_end - cfg.min_onset_spacing_s, cfg.min_onset_spacing_s)
    onset_lower = np.arange(n_components) * cfg.min_onset_spacing_s
    if onset_lower[-1] > onset_upper:
        return None

    eval_end = onset_upper + cfg.max_duration_s
    eval_time = np.arange(0.0, eval_end + 0.5 / cfg.sample_hz, 1.0 / cfg.sample_hz)
    observed = np.zeros((len(eval_time), 2), dtype=float)
    observed[: len(velocity)] = velocity
    observed_speed = np.linalg.norm(observed, axis=1)
    denominator = max(float(np.sum(observed[:, 0] ** 2 + observed[:, 1] ** 2 + observed_speed**2)), 1e-12)

    lower, upper = [], []
    for i in range(n_components):
        lower.extend([onset_lower[i], cfg.min_duration_s, cfg.lateral_bounds[0], cfg.forward_bounds[0]])
        upper.extend([onset_upper, cfg.max_duration_s, cfg.lateral_bounds[1], cfg.forward_bounds[1]])
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    net_displacement = np.trapz(velocity, time, axis=0)
    rng = np.random.default_rng(seed + 1009 * n_components)
    best = None

    def residual(params: np.ndarray) -> np.ndarray:
        predicted = reconstruct_velocity(eval_time, params.reshape(n_components, 4))
        predicted_speed = np.linalg.norm(predicted, axis=1)
        raw = np.column_stack((predicted - observed, predicted_speed - observed_speed))
        return raw.ravel() / np.sqrt(denominator)

    for restart in range(cfg.restarts):
        base_onsets = np.linspace(0.0, min(onset_upper, movement_end * 0.75), n_components)
        base_duration = np.clip(max(movement_end * 0.75, cfg.min_duration_s), cfg.min_duration_s, cfg.max_duration_s)
        x0 = np.empty((n_components, 4), dtype=float)
        x0[:, 0] = base_onsets
        x0[:, 1] = base_duration
        x0[:, 2:4] = net_displacement[None, :] / n_components
        if restart:
            x0[:, 0] += rng.normal(0, cfg.min_onset_spacing_s * 0.35, n_components)
            x0[:, 1] *= rng.uniform(0.75, 1.25, n_components)
            x0[:, 2:4] *= rng.uniform(0.65, 1.35, (n_components, 2))
        x0 = np.clip(x0.ravel(), lower + 1e-7, upper - 1e-7)
        try:
            result = least_squares(
                residual,
                x0,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                max_nfev=cfg.max_nfev,
            )
        except (ValueError, FloatingPointError):
            continue
        error = float(np.sum(residual(result.x) ** 2))
        if best is None or error < best[0]:
            best = (error, result)

    if best is None:
        return None
    error, result = best
    params = result.x.reshape(n_components, 4)
    params = params[np.argsort(params[:, 0])]
    reconstructed = reconstruct_velocity(eval_time, params)
    rss = max(error * denominator, 1e-12)
    n_observations = 3 * len(eval_time)
    bic = n_observations * np.log(rss / n_observations) + 4 * n_components * np.log(n_observations)
    return ComponentFit(n_components, params, error, float(bic), bool(result.success), int(result.nfev), reconstructed, eval_time)


def decompose_position(
    position_xy: np.ndarray,
    cfg: SubmovementConfig | None = None,
    trial_id: str = "trial",
) -> SubmovementResult:
    cfg = cfg or SubmovementConfig()
    time, velocity = _prepare_velocity(position_xy, cfg.sample_hz, cfg.cutoff_hz)
    fits = {}
    base_seed = _seed_from_text(trial_id)
    for n in range(1, cfg.max_components + 1):
        fit = fit_component_count(time, velocity, n, cfg, base_seed)
        if fit is not None:
            fits[n] = fit
    if not fits:
        raise RuntimeError(f"no feasible submovement fit for {trial_id}")

    ordered = [fits[n] for n in sorted(fits)]
    selected = next((fit for fit in ordered if fit.normalized_error <= cfg.error_threshold), None)
    if selected is None:
        selected = next((fit for fit in ordered if fit.normalized_error < cfg.fallback_error_threshold), None)
    if selected is None:
        selected = min(ordered, key=lambda fit: fit.normalized_error)
    selected_bic = min(ordered, key=lambda fit: fit.bic)
    return SubmovementResult(selected, selected_bic, fits)


def decompose_trial(trial: dict, cfg: SubmovementConfig | None = None) -> SubmovementResult:
    start = int(trial["move_start_idx"])
    end = int(trial["move_end_idx"])
    position_xy = np.asarray(trial["pos_filtered"])[start : end + 1, :2]
    return decompose_position(position_xy, cfg, trial["metadata"].get("trial_id", "trial"))


def decompose_normalized_trajectory(
    trajectory: np.ndarray,
    movement_time_s: float,
    cfg: SubmovementConfig | None = None,
    trial_id: str = "generated",
) -> SubmovementResult:
    """Decompose generated phase-normalized x/y after restoring physical time."""
    cfg = cfg or SubmovementConfig()
    trajectory = np.asarray(trajectory, dtype=float)
    sample_hz = max((len(trajectory) - 1) / max(float(movement_time_s), 1e-3), 20.0)
    generated_cfg = SubmovementConfig(**{**cfg.__dict__, "sample_hz": sample_hz})
    return decompose_position(trajectory[:, :2], generated_cfg, trial_id)
