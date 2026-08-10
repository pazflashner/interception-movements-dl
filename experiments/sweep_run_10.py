"""Full 10-seed x latent-dim sweep. Writes results/sweep_summary_10seed.csv."""
import sys, os, time, pickle
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.path.insert(0, REPO)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch

import config
from src.run_config import RunConfig
from src.train import train_vae, split_subjects_for
from src.vae_model import NormStats
from src.evaluate import (
    compute_reconstruction_mse, timing_reconstruction_error,
    encode_trials, behavioural_probing,
)

with open(config.DATA_PROCESSED_DIR / "trials.pkl", "rb") as f:
    trials = pickle.load(f)
print(f"Loaded {len(trials)} trials", flush=True)

DIMS = [2, 3, 4, 8, 16]
SEEDS = list(range(10))          # full 10-seed run
EPOCHS = 120

rows = []
t0 = time.time()
for seed in SEEDS:
    for d in DIMS:
        cfg = RunConfig(seed=seed, latent_dim=d, epochs=EPOCHS)
        train_t, val_t, test_t = split_subjects_for(trials, cfg)
        model, hist, run_dir = train_vae(train_t, val_t, cfg=cfg, device="cpu")
        ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
        norm = NormStats.from_checkpoint(ckpt)
        mse = compute_reconstruction_mse(model, test_t, norm)
        tdf = timing_reconstruction_error(model, test_t, norm)
        def tr2(n):
            m = tdf.loc[tdf.timing_feature == n, "r2"]; return float(m.iloc[0]) if len(m) else float("nan")
        mus, _, _, subs = encode_trials(model, test_t, norm)
        probe = behavioural_probing(mus, test_t, subs)
        def pr(n):
            m = probe.loc[probe.target == n, "r2_linear_loso"]; return float(m.iloc[0]) if len(m) else float("nan")
        n_pos = int((probe["r2_linear_loso"] > 0).sum())
        rows.append(dict(seed=seed, latent=d, recon_mse=mse,
                         r2_move_time=tr2("movement_time_s"), r2_init_time=tr2("initiation_time_s"),
                         probe_move=pr("movement_time_s"), probe_move_sd=pr("movement_time_s_sd"),
                         probe_init=pr("initiation_time_s"), n_feat_pos=n_pos, n_feat=len(probe)))
        print(f"[{time.time()-t0:5.0f}s] seed{seed} z{d:2d}: mse={mse:.3f} "
              f"R2_move={tr2('movement_time_s'):.3f} R2_init={tr2('initiation_time_s'):.3f} "
              f"feat_pos={n_pos}/{len(probe)}", flush=True)

df = pd.DataFrame(rows)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = config.RESULTS_DIR / "sweep_summary_10seed.csv"
df.to_csv(out, index=False)
print("\n" + "=" * 70, flush=True)
print("FULL 10-SEED SWEEP SUMMARY (mean ± sd over 10 seeds)", flush=True)
print("=" * 70, flush=True)
agg = df.groupby("latent").agg(
    recon_mse=("recon_mse", "mean"), recon_sd=("recon_mse", "std"),
    R2_move=("r2_move_time", "mean"), R2_move_sd=("r2_move_time", "std"),
    R2_init=("r2_init_time", "mean"),
    feat_pos=("n_feat_pos", "mean"), feat_pos_sd=("n_feat_pos", "std"),
).round(3)
print(agg.to_string(), flush=True)
print(f"\nSaved {out}   ({time.time()-t0:.0f}s total)", flush=True)
