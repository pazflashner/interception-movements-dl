"""
Smoke tests for the CVAE — all on dummy tensors, no real data required.

The point is to separate two failure modes that are easy to confuse: a broken
model/loss/training loop, and a broken data stage. Everything here runs on
synthetic trials from ``src/dummy_data.py`` built to the same schema as
preprocessed real trials, so if these pass, anything that then goes wrong on the
real dataset is a data problem.

Run either way:

    python tests/test_vae_smoke.py     # standalone, no pytest needed
    pytest tests/test_vae_smoke.py -v
"""
from __future__ import annotations

import shutil
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dummy_data import make_dummy_batch, make_dummy_trials
from src.run_config import RunConfig, set_seed
from src.train import split_subject_ids, split_subjects, train_vae
from src.vae_model import (
    CONDITION_DIM,
    TIMING_DIM,
    ConditionalVAE,
    NormStats,
    TrajectoryDataset,
    encode_condition,
    encode_timing,
    kl_weight_at,
    vae_loss,
)

T = config.NORMALISED_LENGTH
INPUT_DIM = T * 3


# ── Model plumbing ────────────────────────────────────────────────────────────
def test_forward_shapes():
    """A forward pass returns the shapes the training loop expects."""
    set_seed(0)
    batch = 16
    model = ConditionalVAE(latent_dim=3, hidden_dim=32)
    traj, timing, cond = make_dummy_batch(batch_size=batch, seed=0)

    recon, recon_timing, mu, logvar, z = model(
        torch.from_numpy(traj), torch.from_numpy(cond), torch.from_numpy(timing)
    )

    assert recon.shape == (batch, INPUT_DIM), recon.shape
    assert recon_timing.shape == (batch, TIMING_DIM), recon_timing.shape
    assert mu.shape == logvar.shape == z.shape == (batch, 3)
    assert torch.isfinite(recon).all() and torch.isfinite(recon_timing).all()


def test_encoder_sees_timing():
    """
    Timing must reach the encoder, or the latent cannot carry it.

    Guards the fix for the reviewer's concern: if timing were silently dropped
    from the encoder input, changing it would leave mu untouched and the timing
    head could only ever regress the condition-wise mean.
    """
    set_seed(0)
    model = ConditionalVAE(latent_dim=3, hidden_dim=32)
    traj, timing, cond = make_dummy_batch(batch_size=8, seed=1)
    traj_t, cond_t = torch.from_numpy(traj), torch.from_numpy(cond)

    mu_a, _ = model.encode(traj_t, cond_t, torch.from_numpy(timing))
    mu_b, _ = model.encode(traj_t, cond_t, torch.from_numpy(timing) + 1.0)

    assert not torch.allclose(mu_a, mu_b), "encoder output is independent of timing"

    # Same trajectory + condition, same timing -> deterministic encoder output.
    mu_c, _ = model.encode(traj_t, cond_t, torch.from_numpy(timing))
    assert torch.allclose(mu_a, mu_c)


def test_condition_reaches_encoder_and_decoder():
    """
    The condition vector must change both halves of the model.

    If it reached only the encoder, the decoder could not be told which task it
    is generating for, and z would have to absorb task variance — the opposite
    of what conditioning is for.
    """
    set_seed(0)
    model = ConditionalVAE(latent_dim=3, hidden_dim=32)
    traj, timing, cond = make_dummy_batch(batch_size=8, seed=5)
    traj_t, timing_t = torch.from_numpy(traj), torch.from_numpy(timing)

    # Two different task conditions: sp=1/left vs sp=3/right.
    c_a = torch.tensor(np.tile(encode_condition(1, 1), (8, 1)))
    c_b = torch.tensor(np.tile(encode_condition(3, 2), (8, 1)))

    mu_a, _ = model.encode(traj_t, c_a, timing_t)
    mu_b, _ = model.encode(traj_t, c_b, timing_t)
    assert not torch.allclose(mu_a, mu_b), "encoder ignores the condition vector"

    z = torch.zeros(8, 3)
    out_a, tim_a = model.decode(z, c_a)
    out_b, tim_b = model.decode(z, c_b)
    assert not torch.allclose(out_a, out_b), "decoder ignores the condition vector"
    assert not torch.allclose(tim_a, tim_b), "timing head ignores the condition vector"


def test_condition_encoding_is_correct():
    """sp one-hot and side binary land in the documented slots."""
    assert encode_condition(1, 1).tolist() == [1, 0, 0, 0]
    assert encode_condition(2, 1).tolist() == [0, 1, 0, 0]
    assert encode_condition(3, 2).tolist() == [0, 0, 1, 1]
    # sp indexes the (start position, speed range) pair, so all three are distinct
    assert len({tuple(encode_condition(sp, 1)) for sp in (1, 2, 3)}) == 3
    assert len(encode_condition(1, 1)) == CONDITION_DIM


def test_kl_annealing_schedules():
    """β ramps from 0 to the target and never exceeds it."""
    target = 1.0

    # linear: starts at 0, reaches target exactly at the end of the ramp, holds
    assert kl_weight_at(1, target, "linear", 10) == 0.0
    assert kl_weight_at(6, target, "linear", 10) == pytest_approx(0.5)
    assert kl_weight_at(11, target, "linear", 10) == target
    assert kl_weight_at(500, target, "linear", 10) == target

    linear = [kl_weight_at(e, target, "linear", 10) for e in range(1, 30)]
    assert linear == sorted(linear), "linear schedule must be monotone"
    assert max(linear) <= target

    # cyclical: repeatedly returns to 0, then settles at target after the cycles
    cyc = [kl_weight_at(e, target, "cyclical", 10, cycles=3) for e in range(1, 60)]
    assert min(cyc) == 0.0 and max(cyc) <= target
    assert cyc[:12].count(0.0) >= 2, "cyclical schedule should restart at 0"
    assert kl_weight_at(55, target, "cyclical", 10, cycles=3) == target

    # none: constant
    assert all(kl_weight_at(e, target, "none", 10) == target for e in (1, 5, 100))


def pytest_approx(x, tol=1e-9):
    """Tiny local helper so the file runs without pytest installed."""
    class _Approx(float):
        def __eq__(self, other):
            return abs(float(self) - other) < tol
    return _Approx(x)


def test_shape_only_model_still_works():
    """``timing_dim=0`` recovers the original shape-only VAE."""
    set_seed(0)
    model = ConditionalVAE(latent_dim=2, hidden_dim=32, timing_dim=0)
    traj, _, cond = make_dummy_batch(batch_size=8, seed=2)

    recon, recon_timing, mu, logvar, _ = model(torch.from_numpy(traj), torch.from_numpy(cond))
    assert recon.shape == (8, INPUT_DIM)
    assert recon_timing is None

    total, rl, kl, tl = vae_loss(recon, torch.from_numpy(traj), mu, logvar)
    assert float(tl) == 0.0
    assert torch.isfinite(total)


def test_loss_terms_are_sane():
    """KL is non-negative, every term is finite, and total is their weighted sum."""
    set_seed(0)
    model = ConditionalVAE(latent_dim=3, hidden_dim=32)
    traj, timing, cond = make_dummy_batch(batch_size=16, seed=3)
    traj_t, timing_t, cond_t = map(torch.from_numpy, (traj, timing, cond))

    with torch.no_grad():
        recon, recon_timing, mu, logvar, _ = model(traj_t, cond_t, timing_t)
        total, rl, kl, tl = vae_loss(
            recon, traj_t, mu, logvar,
            kl_weight=1.0, recon_timing=recon_timing, target_timing=timing_t, timing_weight=1.0,
        )

    for name, term in (("total", total), ("recon", rl), ("kl", kl), ("timing", tl)):
        assert torch.isfinite(term), f"{name} is not finite"
    assert kl.item() >= 0.0, "KL divergence must be non-negative"
    assert rl.item() > 0.0 and tl.item() > 0.0
    assert np.isclose(total.item(), rl.item() + tl.item() + kl.item(), rtol=1e-5)


def test_kl_is_zero_at_the_prior():
    """The KL term vanishes exactly when q(z|x) equals the N(0, I) prior."""
    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)
    recon = torch.zeros(4, 5)
    _, _, kl, _ = vae_loss(recon, recon, mu, logvar)
    assert abs(float(kl)) < 1e-6, float(kl)


def test_gradients_reach_every_parameter():
    """No dead branch: one backward pass produces a gradient for every weight."""
    set_seed(0)
    model = ConditionalVAE(latent_dim=3, hidden_dim=32)
    traj, timing, cond = make_dummy_batch(batch_size=8, seed=4)
    traj_t, timing_t, cond_t = map(torch.from_numpy, (traj, timing, cond))

    recon, recon_timing, mu, logvar, _ = model(traj_t, cond_t, timing_t)
    total, *_ = vae_loss(
        recon, traj_t, mu, logvar,
        recon_timing=recon_timing, target_timing=timing_t,
    )
    total.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


# ── Learning ──────────────────────────────────────────────────────────────────
def _overfit_one_batch(steps: int = 300, timing_weight: float = 1.0, seed: int = 0):
    """Train on a single fixed batch; returns the per-step loss components."""
    set_seed(seed)
    model = ConditionalVAE(latent_dim=4, hidden_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    traj, timing, cond = make_dummy_batch(batch_size=32, seed=seed)
    traj_t, timing_t, cond_t = map(torch.from_numpy, (traj, timing, cond))

    history = {"total": [], "recon": [], "kl": [], "timing": []}
    for _ in range(steps):
        recon, recon_timing, mu, logvar, _ = model(traj_t, cond_t, timing_t)
        total, rl, kl, tl = vae_loss(
            recon, traj_t, mu, logvar,
            kl_weight=config.KL_WEIGHT,
            recon_timing=recon_timing, target_timing=timing_t,
            timing_weight=timing_weight,
        )
        opt.zero_grad()
        total.backward()
        opt.step()

        history["total"].append(total.item())
        history["recon"].append(rl.item())
        history["kl"].append(kl.item())
        history["timing"].append(tl.item())

    return history


def test_loss_decreases_on_dummy_data():
    """
    The core sanity check: the loss goes down, and stays finite doing it.

    Compares the mean of the first and last 10 steps rather than single values,
    since the stochastic sampling of z makes any individual step noisy.
    """
    history = _overfit_one_batch(steps=300)

    assert all(np.isfinite(history["total"])), "loss went non-finite during training"

    first = float(np.mean(history["total"][:10]))
    last = float(np.mean(history["total"][-10:]))
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
    assert last < 0.7 * first, (
        f"loss barely moved: {first:.4f} -> {last:.4f} "
        f"({100 * (1 - last / first):.1f}% reduction, expected > 30%)"
    )

    recon_first = float(np.mean(history["recon"][:10]))
    recon_last = float(np.mean(history["recon"][-10:]))
    assert recon_last < recon_first, "reconstruction term did not improve"


def test_timing_head_learns():
    """
    The timing head must actually fit its target, not just ride along.

    This is the check that the reviewer's concern is addressed in substance: if
    movement time were unlearnable through this architecture, the timing MSE
    would sit flat near the variance of the standardised target (~1.0).
    """
    history = _overfit_one_batch(steps=300)

    first = float(np.mean(history["timing"][:10]))
    last = float(np.mean(history["timing"][-10:]))
    assert last < first, f"timing loss did not decrease: {first:.4f} -> {last:.4f}"
    # vae_loss sums over the timing dims, so the target variance of the
    # standardised targets is TIMING_DIM, not 1. Compare per dimension.
    per_dim = last / TIMING_DIM
    assert per_dim < 0.5, (
        f"timing MSE stayed near the target variance: {per_dim:.4f} per dim "
        f"({last:.4f} summed over {TIMING_DIM})"
    )


# ── Dataset & training loop ───────────────────────────────────────────────────
def test_dummy_trials_match_the_real_schema():
    """Dummy trials carry every key the dataset and feature code reads."""
    from src.features import compute_trial_features

    trials = make_dummy_trials(n_subjects=3, n_trials_per_subject=5, seed=0)
    assert len(trials) == 15

    required = {"pos_norm", "speed_norm", "stim_onset_idx", "move_start_idx",
                "move_end_idx", "metadata"}
    for t in trials:
        assert required <= set(t), f"missing keys: {required - set(t)}"
        assert t["pos_norm"].shape == (T, 3)

    # Timing must round-trip: what the generator was asked for is what the
    # feature extractor and the model's timing encoder read back.
    feats = compute_trial_features(trials[0])
    timing = encode_timing(trials[0])
    assert np.isclose(timing[0], feats["movement_time_s"])
    assert np.isclose(timing[1], feats["initiation_time_s"])
    assert 0.0 < feats["movement_time_s"] < 2.0
    assert feats["peak_speed_mm_s"] > 0


def test_dataset_batches_correctly():
    """``TrajectoryDataset`` yields the 4-tuple the training loop unpacks."""
    trials = make_dummy_trials(n_subjects=2, n_trials_per_subject=4, seed=1)
    ds = TrajectoryDataset(trials)

    assert len(ds) == 8
    assert ds.trajectories.shape == (8, INPUT_DIM)
    assert ds.timings.shape == (8, TIMING_DIM)
    assert ds.conditions.shape == (8, CONDITION_DIM)

    traj, timing, cond, idx = ds[0]
    assert traj.shape == (INPUT_DIM,) and traj.dtype == torch.float32
    assert timing.shape == (TIMING_DIM,) and timing.dtype == torch.float32
    assert cond.shape == (CONDITION_DIM,)
    assert idx == 0

    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    btraj, btiming, bcond, bidx = next(iter(loader))
    assert btraj.shape == (4, INPUT_DIM)
    assert btiming.shape == (4, TIMING_DIM)


def test_subject_split_is_disjoint_and_seed_stable():
    """Splits never leak a subject across sets and depend only on the seed."""
    subjects = [f"s{i:02d}" for i in range(10)]
    a = split_subject_ids(subjects, n_train=6, n_val=2, n_test=2, seed=0)
    b = split_subject_ids(list(reversed(subjects)), n_train=6, n_val=2, n_test=2, seed=0)
    c = split_subject_ids(subjects, n_train=6, n_val=2, n_test=2, seed=1)

    assert a == b, "split depends on input order, not just the seed"
    assert a != c, "different seeds produced identical splits"

    train, val, test = a
    assert len(train) == 6 and len(val) == 2 and len(test) == 2
    assert not (set(train) & set(val)) and not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_train_vae_end_to_end_on_dummy_data():
    """
    The full training loop runs and improves, writing the expected artefacts.

    This is the milestone check: model, loss, loaders, normalisation, early
    stopping and checkpointing all work together before real data is involved.
    """
    trials = make_dummy_trials(n_subjects=16, n_trials_per_subject=25, seed=2)
    train_trials, val_trials, test_trials = split_subjects(
        trials, n_train=10, n_val=4, n_test=2, seed=0
    )
    assert train_trials and val_trials and test_trials

    # Keep the temporary run under the repository. Some managed Windows
    # environments create system-temp directories with an owner ACL that the
    # test process cannot subsequently write into.
    tmp = Path(".tmp") / f"vae_smoke_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cfg = RunConfig(
            seed=0, latent_dim=3, hidden_dim=64,
            epochs=40, batch_size=32, lr=1e-3, patience=40,
        )
        model, history, run_dir = train_vae(
            train_trials, val_trials, cfg=cfg, run_dir=tmp / "run", device="cpu"
        )

        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "checkpoint.pt").exists()
        assert (run_dir / "history.json").exists()

        # Only the training loss is asserted to fall. Validation here is on
        # held-out *subjects*, and with a handful of synthetic subjects the model
        # overfits them — as it does on the real 17/4/7 split. That is a
        # modelling result to be measured, not something a smoke test should
        # gate on; what is being checked is that the loop optimises what it
        # claims to, on both output heads.
        for key in ("train_loss", "train_recon", "train_timing"):
            first = float(np.mean(history[key][:3]))
            last = float(np.mean(history[key][-3:]))
            assert last < first, f"{key} did not decrease: {first:.4f} -> {last:.4f}"
        assert all(np.isfinite(history["val_loss"])), "validation loss went non-finite"
        assert len(history["val_loss"]) == len(history["train_loss"])

        # The checkpoint round-trips into a working model with its own stats.
        ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
        assert ckpt["timing_dim"] == TIMING_DIM
        norm = NormStats.from_checkpoint(ckpt)
        assert norm.timing_mean.shape == (TIMING_DIM,)

        reloaded = ConditionalVAE(
            latent_dim=ckpt["latent_dim"], hidden_dim=cfg.hidden_dim,
            timing_dim=ckpt["timing_dim"],
        )
        reloaded.load_state_dict(ckpt["model_state"])

        # And evaluation runs on held-out dummy subjects.
        from src.evaluate import encode_trials, timing_reconstruction_error

        mus, logvars, zs, subjects = encode_trials(reloaded, test_trials, norm, device="cpu")
        assert mus.shape == (len(test_trials), cfg.latent_dim)
        assert np.isfinite(mus).all()

        timing_df = timing_reconstruction_error(reloaded, test_trials, norm, device="cpu")
        assert len(timing_df) == TIMING_DIM
        assert set(timing_df["timing_feature"]) == set(config.TIMING_FEATURES)
        # Predictions land in a physically plausible range, in seconds.
        assert (timing_df["mae_ms"] > 0).all()
        assert (timing_df["mae_ms"] < 1000).all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Standalone runner ─────────────────────────────────────────────────────────
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []

    print(f"Running {len(tests)} smoke tests on dummy data (torch {torch.__version__})\n")
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a test runner reports, it does not raise
            failures.append((fn.__name__, exc))
            print(f"  FAIL  {fn.__name__}: {exc}")
        else:
            print(f"  ok    {fn.__name__}")

    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED")
        return 1
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
