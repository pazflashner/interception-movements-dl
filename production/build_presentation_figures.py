"""Draw the styled deck's own figures from the study's saved numbers.

Run:  python production/build_presentation_figures.py

Figures that the course deck renders as native PowerPoint charts have to be
pictures here, so they are drawn once into production/presentation_assets/ and
referenced from production/styled_slides.py. Values are read from the audit
CSVs rather than typed in, so a figure can never drift from the study.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MEANS = ROOT / "production" / "audit_2026_09_08" / "all_model_means.csv"
OUT = ROOT / "production" / "presentation_assets"

# The deck's palette, matching build_presentation_styled_pptx.py.
GREY, TEAL, INK, RULE = "#8395A3", "#087C83", "#161E2E", "#E1E6EC"
FAMILIES = [("spline_pca", "Spline + PCA"), ("conditional_ae", "CAE"),
            ("cvae", "CVAE"), ("unconditional_vae", "VAE")]


def _means(metric: str) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(MEANS.open(encoding="utf-8")))
    out: dict[str, dict[str, float]] = {}
    for family, _ in FAMILIES:
        out[family] = {r["latent_dim"]: float(r[metric])
                       for r in rows if r["model_family"] == family}
    return out


def mean_ks_chart() -> Path:
    """Participant-balanced mean feature KS, four models at n = 3 and n = 8."""
    data = _means("mean_ks")
    labels = [label for _, label in FAMILIES]
    n3 = [data[f]["3"] for f, _ in FAMILIES]
    n8 = [data[f]["8"] for f, _ in FAMILIES]

    fig, ax = plt.subplots(figsize=(7.4, 3.3), layout="constrained")
    x = range(len(labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], n3, width, label="n = 3", color=GREY)
    ax.bar([i + width / 2 for i in x], n8, width, label="n = 8", color=TEAL)

    for i, (a, b) in enumerate(zip(n3, n8)):
        ax.text(i - width / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=9, color=INK)
        ax.text(i + width / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=9, color=INK)

    ax.set_xticks(list(x), labels, fontsize=11, color=INK)
    ax.set_ylabel("Mean feature KS  (lower = better)", fontsize=11, color=INK)
    ax.set_ylim(0, max(n3 + n8) * 1.18)
    ax.tick_params(axis="y", labelsize=9, colors=INK)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.yaxis.grid(True, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=10)

    path = OUT / "mean_ks_chart.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = mean_ks_chart()
    print(f"{path}\n  values read from {MEANS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
