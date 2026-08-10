import sys, os
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv(os.path.join(REPO, "results", "loss_experiment.csv"))
d = df[df.latent == 8]
order = ["baseline", "beta4", "discriminative"]; nice = ["baseline", "β-VAE (4×KL)", "discriminative"]
colors = ["#1f77b4", "#2ca02c", "#d62728"]
def ms(col): g = d.groupby("variant")[col]; return g.mean().reindex(order).values, g.std(ddof=1).reindex(order).values
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(15, 3.9)); x = np.arange(3)
specs = [("fp_id_acc", "Held-out fingerprint ID (higher=better)", 0.143, "chance"),
         ("sep_ratio", "Separation ratio (training objective)", None, None),
         ("recon_mse", "Reconstruction MSE (lower=better)", None, None),
         ("r2_move", "Timing R² (higher=better)", None, None)]
for a, (col, title, hline, hlabel) in zip(ax, specs):
    m, s = ms(col); a.bar(x, m, yerr=s, capsize=4, color=colors)
    if hline is not None:
        a.axhline(hline, ls="--", color="gray", lw=1); a.text(2.4, hline + 0.01, hlabel, color="gray", fontsize=8, ha="right")
    a.set_xticks(x); a.set_xticklabels(nice, rotation=20, ha="right"); a.set_title(title, fontsize=10.5)
fig.suptitle("Loss-function comparison at n=8 (mean ± sd, 10 seeds, 7 held-out subjects)", fontsize=13, y=1.03)
fig.tight_layout()
out = os.path.join(REPO, "figures", "loss_comparison.png"); fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
