"""
Synthetic trials for smoke-testing the pipeline without any real data.

The generated dicts use exactly the schema ``preprocessing.preprocess_trial``
returns, so ``TrajectoryDataset``, ``train_vae`` and the evaluation code all run
on them unchanged. That is the point: it lets the model, loss and training loop
be validated end-to-end before — or independently of — the data stage, so a
problem in one is never mistaken for a problem in the other.

Each synthetic subject is generated from two latent traits — *vigour* (how far
and how fast they reach) and *style* (how much they bow the path) — with
per-trial jitter around them. Vigour drives amplitude, movement time and
reaction time together, so timing is genuinely predictable from a
low-dimensional code rather than being independent noise bolted on. That mirrors
the project's own hypothesis (an individual signature is a coupled
spatiotemporal pattern) and gives the smoke test the properties it needs: enough
structure that a working VAE visibly reduces its loss, and a real spread in
movement time for the timing head to recover.
"""
from __future__ import annotations

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _minimum_jerk(s: np.ndarray) -> np.ndarray:
    """Normalised minimum-jerk position profile, 0 → 1 over phase *s*."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def make_dummy_trial(
    rng: np.random.Generator,
    subject: str,
    trial_index: int,
    amplitude: float,
    curvature: float,
    movement_time_s: float,
    initiation_time_s: float,
    T: int = config.NORMALISED_LENGTH,
    fs: float = config.RECORDING_HZ,
) -> dict:
    """One synthetic trial in the ``preprocess_trial`` output schema."""
    s = np.linspace(0.0, 1.0, T)
    profile = _minimum_jerk(s)

    pos = np.empty((T, 3), dtype=np.float64)
    pos[:, 0] = curvature * np.sin(np.pi * s)             # lateral bow
    pos[:, 1] = amplitude * profile                        # forward reach
    pos[:, 2] = 0.15 * curvature * np.sin(2 * np.pi * s)   # slight lift
    pos += rng.normal(0.0, 0.5, size=pos.shape)            # sensor noise
    pos -= pos[0]                                          # spatial normalisation

    vel = np.gradient(pos, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Frame indices consistent with the requested timing, so encode_timing and
    # compute_trial_features recover the values used to build the trial.
    stim_onset = 100
    # Go-signal == stimulus onset for synthetic trials (no foreperiod modelled),
    # so initiation_time_s is recovered as (move_start - go_signal) unchanged.
    go_signal = stim_onset
    move_start = go_signal + int(round(initiation_time_s * fs))
    move_end = move_start + int(round(movement_time_s * fs))
    sp = 1 + trial_index % 3
    side = 1 + trial_index % 2

    return {
        "pos_raw": pos,
        "pos_filtered": pos,
        "pos_norm": pos,
        "vel_norm": vel,
        "speed_norm": speed,
        "stim_onset_idx": stim_onset,
        "go_signal_idx": go_signal,
        "move_start_idx": move_start,
        "move_end_idx": move_end,
        "metadata": {
            "subject": subject,
            "condition": config.CONDITION_FREE_EYE,
            "sp": sp,
            "side": side,
            "rep": trial_index,
            "starting_position_mm": config.STARTING_POSITIONS.get(sp),
            "starting_side": "left" if side == 1 else "right",
            "trial_id": f"{subject}_dummy_{trial_index}",
        },
    }


def make_dummy_trials(
    n_subjects: int = 12,
    n_trials_per_subject: int = 30,
    T: int = config.NORMALISED_LENGTH,
    seed: int = 0,
) -> list[dict]:
    """
    A synthetic dataset of ``n_subjects × n_trials_per_subject`` trials.

    Every subject is drawn from the same two-trait distribution, so a held-out
    subject falls inside the range spanned by the others and the model is being
    asked to interpolate rather than extrapolate — the same thing the real
    leave-N-subjects-out protocol assumes.
    """
    rng = np.random.default_rng(seed)
    trials: list[dict] = []

    for i in range(n_subjects):
        subject = f"dummy{i:02d}"
        # Subject signature: two traits, coupled into the observable parameters.
        vigour = rng.uniform(0.0, 1.0)   # far + fast at 1, near + slow at 0
        style = rng.uniform(-1.0, 1.0)   # signed path curvature

        base_amplitude = 150.0 + 100.0 * vigour
        base_curvature = 40.0 * style
        base_move_time = 0.60 - 0.25 * vigour   # seconds; ballistic reaches
        base_init_time = 0.35 - 0.15 * vigour

        for j in range(n_trials_per_subject):
            trials.append(
                make_dummy_trial(
                    rng=rng,
                    subject=subject,
                    trial_index=j,
                    amplitude=base_amplitude * rng.normal(1.0, 0.05),
                    curvature=base_curvature + rng.normal(0.0, 4.0),
                    movement_time_s=max(0.05, base_move_time * rng.normal(1.0, 0.08)),
                    initiation_time_s=max(0.01, base_init_time * rng.normal(1.0, 0.12)),
                    T=T,
                )
            )

    return trials


def make_dummy_batch(
    batch_size: int = 32,
    T: int = config.NORMALISED_LENGTH,
    latent_structure: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Raw tensors for testing the model in isolation: (trajectories, timing, conditions).

    Trajectories are flattened to ``(batch_size, T*3)`` and standardised, timing
    to ``(batch_size, TIMING_DIM)``, matching what the training loop feeds the
    model after normalisation. Generated from a low-dimensional factor so the
    batch is genuinely compressible — a VAE that cannot reduce its loss here is
    broken, not under-parameterised.
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 1.0, T)
    basis = np.stack(
        [_minimum_jerk(s), np.sin(np.pi * s), np.sin(2 * np.pi * s)][:latent_structure]
        + [np.ones(T)],
        axis=0,
    )  # (k, T)

    coeffs = rng.normal(size=(batch_size, 3, basis.shape[0]))
    traj = np.einsum("bdk,kt->btd", coeffs, basis)
    traj += rng.normal(0.0, 0.05, size=traj.shape)
    traj = traj.reshape(batch_size, T * 3).astype(np.float32)

    timing = rng.normal(size=(batch_size, config.TIMING_DIM)).astype(np.float32)

    conditions = np.zeros((batch_size, 4), dtype=np.float32)
    conditions[np.arange(batch_size), rng.integers(0, 3, size=batch_size)] = 1.0
    conditions[:, 3] = rng.integers(0, 2, size=batch_size)

    return traj, timing, conditions
