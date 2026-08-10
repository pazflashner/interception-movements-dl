import sys, os, pickle
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.path.insert(0, REPO); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config
from src.vae_model import ConditionalVAE, NormStats
from src.run_config import find_runs
from src.evaluate import encode_trials, latent_feature_correlations

# original complete z=3 run
run = ckpt = None
for r in find_runs():
    if r.name.startswith("z3_seed0_2026"):
        try:
            c = torch.load(r / "checkpoint.pt", weights_only=False)
            if c["latent_dim"] == 3 and "epoch" in c:
                run, ckpt = r, c; break
        except Exception:
            continue
model = ConditionalVAE(latent_dim=3, timing_dim=ckpt["timing_dim"])
model.load_state_dict(ckpt["model_state"]); model.eval()
norm = NormStats.from_checkpoint(ckpt)

with open(config.DATA_PROCESSED_DIR / "trials.pkl", "rb") as f:
    trials = pickle.load(f)

mus, _, _, subs = encode_trials(model, trials, norm)
corr = latent_feature_correlations(mus, trials)   # latent_dim, feature, spearman_rho, p_value

nice = {
    "initiation_time_s": "reaction time", "movement_time_s": "movement time",
    "peak_speed_mm_s": "peak speed", "time_to_peak_speed": "time to peak speed",
    "path_length": "path length", "straight_line_dist": "reach distance",
    "curvature_index": "curvature", "max_lateral_deviation": "lateral deviation",
    "end_x": "end x", "end_y": "end y", "end_z": "end z",
}
corr["feat"] = corr["feature"].map(lambda f: nice.get(f, f))
pivot = corr.pivot(index="latent_dim", columns="feat", values="spearman_rho")
order = [nice[f] for f in ["reaction time".replace("reaction time","initiation_time_s")]] if False else None
# order columns by |corr| importance
cols = pivot.abs().max(axis=0).sort_values(ascending=False).index
pivot = pivot[cols]

print("Spearman correlation of each latent with each kinematic feature (z=3):\n")
print(pivot.round(2).to_string())
print("\nDominant feature per latent:")
for zi in pivot.index:
    row = pivot.loc[zi]
    top = row.abs().sort_values(ascending=False).head(3)
    desc = ", ".join(f"{f} ({row[f]:+.2f})" for f in top.index)
    print(f"  z{zi}: {desc}")

# heatmap figure
fig, ax = plt.subplots(figsize=(11, 3.2))
im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(i) for i in pivot.index])
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.4 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="Spearman ρ")
ax.set_title("What each latent variable controls (z=3): correlation with movement features")
fig.tight_layout()
out = os.path.join(REPO, "figures", "latent_interpretation.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out, "  (run:", run.name, ")")
