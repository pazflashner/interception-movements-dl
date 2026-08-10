import sys, os
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(REPO, "results", "experiment_full.csv"))
dims = sorted(df["latent"].unique()); x = np.arange(len(dims))
variants = ["baseline_mlp", "cnn", "mlp+speed", "cnn+speed"]
labels = {"baseline_mlp": "MLP (baseline)", "cnn": "CNN", "mlp+speed": "MLP + speed", "cnn+speed": "CNN + speed"}
colors = {"baseline_mlp": "#1f77b4", "cnn": "#ff7f0e", "mlp+speed": "#2ca02c", "cnn+speed": "#d62728"}
marks = {"baseline_mlp": "o", "cnn": "s", "mlp+speed": "^", "cnn+speed": "D"}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

def series(v, col):
    g = df[df.variant == v].groupby("latent")[col]
    return g.mean().reindex(dims).values, g.std(ddof=1).reindex(dims).values

for v in variants:
    m, s = series(v, "recon_mse")
    ax[0].errorbar(x, m, yerr=s, marker=marks[v], color=colors[v], capsize=3, lw=1.8, label=labels[v])
    m, s = series(v, "r2_move")
    ax[1].errorbar(x, m, yerr=s, marker=marks[v], color=colors[v], capsize=3, lw=1.8, label=labels[v])
    m, s = series(v, "feat_pos")
    ax[2].errorbar(x, m, yerr=s, marker=marks[v], color=colors[v], capsize=3, lw=1.8, label=labels[v])

ax[0].set_title("Reconstruction error"); ax[0].set_ylabel("Trajectory MSE (lower = better)")
ax[1].set_title("Timing prediction"); ax[1].set_ylabel("Movement-time R\u00b2 (higher = better)"); ax[1].set_ylim(0.65, 1.01)
ax[2].set_title("Fingerprint"); ax[2].set_ylabel("Behavioural features > chance (of 12)"); ax[2].set_ylim(0, 8)
for a in ax:
    a.set_xticks(x); a.set_xticklabels([str(d) for d in dims]); a.set_xlabel("Latent dimension  n")
ax[1].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Architecture & feature comparison  (mean \u00b1 sd over 10 seeds, 7 held-out subjects)", fontsize=13, y=1.02)
fig.tight_layout()
out = os.path.join(REPO, "figures", "variant_comparison.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)

# print the deltas that matter
print("\nTiming R2 (move) — MLP vs CNN:")
for d in dims:
    mlp = df[(df.variant=="baseline_mlp")&(df.latent==d)].r2_move.mean()
    cnn = df[(df.variant=="cnn")&(df.latent==d)].r2_move.mean()
    print(f"  n={d:2d}: MLP {mlp:.3f}  CNN {cnn:.3f}  (delta {cnn-mlp:+.3f})")
