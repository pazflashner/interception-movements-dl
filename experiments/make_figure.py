import sys, os
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(REPO, "results", "sweep_summary_10seed.csv"))
g = df.groupby("latent")
dims = sorted(df["latent"].unique())
x = np.arange(len(dims))
labels = [str(d) for d in dims]

def mean_sd(col):
    return g[col].mean().reindex(dims).values, g[col].std(ddof=1).reindex(dims).values

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN = "#1f77b4", "#ff7f0e", "#2ca02c"
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# Panel 1 — reconstruction MSE
m, s = mean_sd("recon_mse")
ax[0].errorbar(x, m, yerr=s, marker="o", color=BLUE, capsize=3, lw=2)
ax[0].set_title("Reconstruction error")
ax[0].set_ylabel("Trajectory MSE  (lower = better)")

# Panel 2 — timing R²
mm, sm = mean_sd("r2_move_time")
mi, si = mean_sd("r2_init_time")
ax[1].errorbar(x, mm, yerr=sm, marker="o", color=BLUE, capsize=3, lw=2, label="movement time")
ax[1].errorbar(x, mi, yerr=si, marker="s", color=ORANGE, capsize=3, lw=2, label="reaction time")
ax[1].axvline(3, ls="--", color="gray", lw=1)               # n=8 elbow
ax[1].text(3.02, 0.86, "elbow (n=8)", color="gray", fontsize=9, rotation=90, va="bottom")
ax[1].set_title("Timing reconstruction")
ax[1].set_ylabel("R² on held-out subjects  (higher = better)")
ax[1].set_ylim(0.75, 1.01)
ax[1].legend(frameon=False, loc="lower right")

# Panel 3 — behavioural features above chance
mf, sf = mean_sd("n_feat_pos")
ax[2].errorbar(x, mf, yerr=sf, marker="o", color=GREEN, capsize=3, lw=2)
ax[2].axhline(11, ls=":", color="gray", lw=1)
ax[2].text(0, 10.4, "all 11 features", color="gray", fontsize=9)
ax[2].set_title("Behavioural fingerprint")
ax[2].set_ylabel("Features predicted > chance  (of 11)")
ax[2].set_ylim(0, 11.5)

for a in ax:
    a.set_xticks(x); a.set_xticklabels(labels); a.set_xlabel("Latent dimension  n")
# headline marker on panel 1
ax[0].annotate("headline\n(n=3)", xy=(1, mean_sd("recon_mse")[0][1]), xytext=(1.4, 0.33),
               fontsize=9, color="#444", arrowprops=dict(arrowstyle="->", color="#888"))

fig.suptitle("CVAE latent-dimension sweep  (mean ± sd over 10 seeds, 7 held-out test subjects)",
             fontsize=13, y=1.02)
fig.tight_layout()
out_dir = os.path.join(REPO, "figures")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "latent_sweep.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
