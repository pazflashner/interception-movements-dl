"""Build the 10-minute project presentation (16:9 PDF slides).

Run:  python production/build_presentation.py

TIMING
    Nine slides, ~7 minutes, leaving ~3 minutes for the live dashboard.
    Per-slide budgets are in SLIDE_SECONDS and printed when the script runs.
    Present the PDF full-screen; it is 16:9.

EDITING
    Each slide is one function call in build(). Text lives inline. Figures come
    from production/figures_extracted/, extracted automatically by
    build_submission_report.py (run that first if the folder is missing).

DESIGN
    Palette matches the paper's own figures (reports/build_final_reports.py),
    so slides and figures read as one system. Helvetica for projection.
"""
from __future__ import annotations

from pathlib import Path
import sys

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGURES = ROOT / "production" / "figures_extracted"
OUTPUT = ROOT / "production" / "10_min_presentation.pdf"

W, H = 338.7 * mm, 190.5 * mm          # 16:9
INK = (0.094, 0.129, 0.165)            # #18212A
BLUE = (0.141, 0.416, 0.553)           # #246A8D
TEAL = (0.165, 0.616, 0.561)           # #2A9D8F
GRAY = (0.365, 0.408, 0.447)           # #5D6872
PAPER = (0.992, 0.992, 0.988)

SLIDE_SECONDS = {
    "Title": 10,
    "The question": 50,
    "Five models, one protocol": 50,
    "Can it redraw a movement?": 60,
    "What does matching a person mean?": 65,
    "Can it invent new movements?": 50,
    "Does the right fingerprint matter?": 90,
    "What holds, what does not": 45,
    "Demo": 180,
}



def ensure_cropped_panel() -> None:
    """Crop draft Figure 11 to its VAE n=8 panel.

    The published figure places CVAE n=3 beside VAE n=8. Jason asked that
    comparisons hold the latent size fixed, so the slide shows one panel only.
    Derived here rather than committed, since figures_extracted/ is git-ignored.
    """
    source, target = FIGURES / "p15_0.png", FIGURES / "p15_0_vae.png"
    if target.exists() or not source.exists():
        return
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(source) as im:
        width, height = im.size
        im.crop((int(width * 0.545), 0, width, height)).save(target)


class Deck:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.n = 0

    # ── page furniture ──────────────────────────────────────────────────────
    def slide(self, title: str, eyebrow: str = "") -> None:
        if self.n:
            self.c.showPage()
        self.n += 1
        self.c.setFillColorRGB(*PAPER)
        self.c.rect(0, 0, W, H, stroke=0, fill=1)
        if eyebrow:
            self.c.setFillColorRGB(*TEAL)
            self.c.setFont("Helvetica-Bold", 12)
            self.c.drawString(24 * mm, H - 22 * mm, eyebrow.upper())
        self.c.setFillColorRGB(*INK)
        self.c.setFont("Helvetica-Bold", 27)
        self.c.drawString(24 * mm, H - 34 * mm, title)
        self.c.setStrokeColorRGB(*BLUE)
        self.c.setLineWidth(1.6)
        self.c.line(24 * mm, H - 39 * mm, 70 * mm, H - 39 * mm)
        self.c.setFillColorRGB(*GRAY)
        self.c.setFont("Helvetica", 9)
        self.c.drawRightString(W - 24 * mm, 10 * mm, str(self.n))

    def bullets(self, items: list[str], x=24 * mm, y=H - 56 * mm, size=15, gap=11 * mm) -> None:
        for item in items:
            self.c.setFillColorRGB(*BLUE)
            self.c.circle(x + 1.6 * mm, y + 1.6 * mm, 1.4 * mm, stroke=0, fill=1)
            self.c.setFillColorRGB(*INK)
            self.c.setFont("Helvetica", size)
            self.c.drawString(x + 7 * mm, y, item)
            y -= gap

    def note(self, text: str, x=24 * mm, y=18 * mm, size=11, color=GRAY) -> None:
        self.c.setFillColorRGB(*color)
        self.c.setFont("Helvetica-Oblique", size)
        self.c.drawString(x, y, text)

    def figure(self, name: str, x, y, max_w, max_h) -> None:
        path = FIGURES / name
        if not path.exists():
            self.c.setFillColorRGB(*GRAY)
            self.c.setFont("Helvetica-Oblique", 11)
            self.c.drawString(x, y + max_h / 2, f"[missing figure: {name}]")
            return
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih)
        self.c.drawImage(img, x + (max_w - iw * scale) / 2, y,
                         iw * scale, ih * scale, mask="auto")

    def headline(self, value: str, label: str, x, y, color=BLUE) -> None:
        """One large number with a caption - use where a figure would be noise."""
        self.c.setFillColorRGB(*color)
        self.c.setFont("Helvetica-Bold", 40)
        self.c.drawString(x, y, value)
        self.c.setFillColorRGB(*GRAY)
        self.c.setFont("Helvetica", 11)
        self.c.drawString(x, y - 8 * mm, label)

    def rows(self, data: list[list[str]], x, y, widths, size=12, gap=9 * mm,
             emphasis: int | None = None) -> None:
        for r, row in enumerate(data):
            cx = x
            for value, width in zip(row, widths):
                if r == 0:
                    self.c.setFillColorRGB(*GRAY)
                    self.c.setFont("Helvetica-Bold", size - 2)
                elif emphasis is not None and r == emphasis:
                    self.c.setFillColorRGB(*BLUE)
                    self.c.setFont("Helvetica-Bold", size)
                else:
                    self.c.setFillColorRGB(*INK)
                    self.c.setFont("Helvetica", size)
                self.c.drawString(cx, y, value)
                cx += width
            if r == 0:
                self.c.setStrokeColorRGB(*GRAY)
                self.c.setLineWidth(0.5)
                self.c.line(x, y - 2.5 * mm, cx - 4 * mm, y - 2.5 * mm)
            y -= gap

    def save(self) -> None:
        self.c.showPage()
        self.c.save()


def build() -> Path:
    d = Deck(OUTPUT)

    # 1 ─────────────────────────────────────────────────────────────────────
    d.slide("Compact Representations of", "0368-3538 Workshop on Deep Learning")
    d.c.setFillColorRGB(*INK)
    d.c.setFont("Helvetica-Bold", 27)
    d.c.drawString(24 * mm, H - 48 * mm, "Human Interception Movements")
    d.bullets(["Seman Libbiss  |  Paz Flashner",
               "Supervision: Prof. Jason Friedman  |  Advisor: Moni Shahar"],
              y=H - 78 * mm, size=14)
    d.figure("p02_1.png", 150 * mm, 22 * mm, 165 * mm, 78 * mm)

    # 2 ─────────────────────────────────────────────────────────────────────
    d.slide("Everyone moves differently. Can we compress that?", "The question")
    d.bullets([
        "Task: intercept a moving target. Fast, ballistic, under one second.",
        "28 participants, 4,732 trials, finger position at 240 Hz.",
        "Everyone solves it, but no two people move the same way.",
    ], y=H - 58 * mm)
    d.headline("3", "numbers per person?", 24 * mm, 62 * mm)
    d.note("Target is a distribution of movements, not one trajectory - perception and movement are stochastic.")
    d.figure("p03_0.png", 150 * mm, 30 * mm, 165 * mm, 90 * mm)

    # 3 ─────────────────────────────────────────────────────────────────────
    d.slide("Five models, one protocol", "Approach")
    d.bullets([
        "Every trial compressed to n numbers, then decoded back.",
        "Spline + PCA (linear, ~10² parameters)  vs  VAE (neural, ~10⁵).",
        "Plus a conditional VAE, a conditional autoencoder, and a task-only floor.",
        "4 folds x 17 train / 4 validation / 7 test participants.",
        "Each participant held out exactly once. Nothing is tested on its own training data.",
    ], y=H - 56 * mm, size=14, gap=10 * mm)
    d.figure("p04_0.png", 24 * mm, 22 * mm, 290 * mm, 62 * mm)

    # 4 ─────────────────────────────────────────────────────────────────────
    d.slide("Can it redraw a movement it just saw?", "Task 1  |  Reconstruction")
    d.rows([["Model", "n=3", "n=8"],
            ["Spline + PCA", "0.097", "0.023"],
            ["VAE", "0.148", "0.033"],
            ["Condition-only Ridge", "1.234", "1.234"]],
           24 * mm, H - 58 * mm, [62 * mm, 30 * mm, 30 * mm], emphasis=1)
    d.note("Participant-balanced MSE, tracker units squared. Lower is better.", y=H - 92 * mm, size=10)
    d.bullets(["The linear baseline wins - and that is expected.",
               "PCA is optimal for squared error; the VAE trades accuracy for a samplable space."],
              y=H - 108 * mm, size=13, gap=9 * mm)
    d.figure("p07_0.png", 168 * mm, 26 * mm, 148 * mm, 100 * mm)

    # 5 ─────────────────────────────────────────────────────────────────────
    d.slide("What does 'matching a person' actually mean?", "How we measure it")
    d.bullets([
        "Not trajectory-by-trajectory. We compare distributions.",
        "Every trial becomes 11 kinematic features:",
    ], y=H - 56 * mm, size=14, gap=10 * mm)
    d.c.setFillColorRGB(*GRAY)
    d.c.setFont("Helvetica", 12)
    for i, line in enumerate([
        "initiation time  .  movement time  .  peak speed  .  time to peak",
        "path length  .  straight-line distance  .  curvature  .  lateral deviation",
        "speed-peak count  .  endpoint x  .  endpoint y",
    ]):
        d.c.drawString(31 * mm, H - 84 * mm - i * 7 * mm, line)
    d.bullets(["Then: KS per feature, plus energy distance and MMD across all 11 jointly."],
              y=H - 115 * mm, size=13)
    d.figure("p13_0.png", 168 * mm, 20 * mm, 148 * mm, 112 * mm)

    # 6 ─────────────────────────────────────────────────────────────────────
    d.slide("Can it invent new movements for a person?", "Task 2  |  Generation")
    d.bullets([
        "Give the model ~85 of a person's trials. Never show it the rest.",
        "It generates 120 new trajectories. Compare against the held-out half.",
    ], y=H - 56 * mm, size=14, gap=10 * mm)
    d.rows([["Model (n=8)", "Mean KS", "Energy", "MMD2"],
            ["VAE", "0.215", "0.411", "0.064"],
            ["Spline + PCA", "0.255", "0.735", "0.115"]],
           24 * mm, H - 86 * mm, [48 * mm, 28 * mm, 28 * mm, 28 * mm], emphasis=1)
    d.note("The VAE generates closer distributions - and it never sees the task conditions at all.",
           y=H - 120 * mm, size=11)
    d.figure("p11_0.png", 168 * mm, 30 * mm, 148 * mm, 92 * mm)

    # 7 ─────────────────────────────────────────────────────────────────────
    d.slide("Does the RIGHT person's fingerprint matter?", "The control that matters")
    d.bullets([
        "Freeze everything: decoder, task draws, noise, covariance, distance reference.",
        "Change one thing only - whose fingerprint we supply.",
    ], y=H - 56 * mm, size=14, gap=10 * mm)
    d.rows([["VAE n=8, fingerprint supplied", "Mean KS", "Energy", "MMD2"],
            ["Own participant", "0.215", "0.411", "0.064"],
            ["Population average", "0.284", "0.879", "0.142"],
            ["Another participant", "0.314", "1.168", "0.195"]],
           24 * mm, H - 86 * mm, [60 * mm, 26 * mm, 26 * mm, 26 * mm], emphasis=1)
    d.headline("27 / 28", "participants improve with their own fingerprint;  all 6 contrasts p < 1.5e-06",
               24 * mm, 30 * mm, TEAL)
    d.figure("p15_0_vae.png", 186 * mm, 26 * mm, 126 * mm, 96 * mm)

    # 8 ─────────────────────────────────────────────────────────────────────
    d.slide("What holds, and what does not", "Honest summary")
    d.c.setFillColorRGB(*TEAL)
    d.c.setFont("Helvetica-Bold", 15)
    d.c.drawString(24 * mm, H - 58 * mm, "Holds")
    d.bullets([
        "Personal information really is in the fingerprint - under a strict control.",
        "The VAE generates closer feature distributions than the baseline.",
    ], y=H - 70 * mm, size=13, gap=9 * mm)
    d.c.setFillColorRGB(*BLUE)
    d.c.setFont("Helvetica-Bold", 15)
    d.c.drawString(176 * mm, H - 58 * mm, "Does not")
    d.bullets([
        "Spline + PCA reconstructs better.",
        "The best generator ignores the task entirely.",
        "A simple average beats it on all 14 targets.",
    ], x=176 * mm, y=H - 70 * mm, size=13, gap=9 * mm)
    d.c.setStrokeColorRGB(*GRAY)
    d.c.setLineWidth(0.6)
    d.c.line(24 * mm, 44 * mm, W - 24 * mm, 44 * mm)
    d.c.setFillColorRGB(*BLUE)
    d.c.setFont("Helvetica-Bold", 15)
    d.c.drawString(24 * mm, 33 * mm, "If you are building this today:")
    d.c.setFillColorRGB(*INK)
    d.c.setFont("Helvetica", 14)
    d.c.drawString(24 * mm, 24 * mm,
                   "reconstruct with Spline + PCA;  generate with the VAE.  Use n=8 - n=3 is worse "
                   "on almost every endpoint.")
    d.c.setFillColorRGB(*GRAY)
    d.c.setFont("Helvetica-Oblique", 10)
    d.c.drawString(24 * mm, 16 * mm,
                   "Caveat: n was matched for comparability, not chosen by a variance or elbow "
                   "criterion. That criterion is the next step.")

    # 9 ─────────────────────────────────────────────────────────────────────
    d.slide("See it work", "Demo")
    d.bullets([
        "Pick a model and capacity, pick a participant.",
        "Move the latent sliders - watch the trajectory change.",
        "Compare generated against that person's real held-out trials.",
    ], y=H - 58 * mm, size=15, gap=11 * mm)
    d.figure("p29_0.png", 24 * mm, 24 * mm, 290 * mm, 78 * mm)

    d.save()
    return OUTPUT


def main() -> None:
    if not FIGURES.exists():
        print(f"warning: {FIGURES} missing - run build_submission_report.py first.")
    ensure_cropped_panel()
    path = build()
    total = sum(SLIDE_SECONDS.values())
    print(f"{path}\n  {len(SLIDE_SECONDS)} slides\n")
    print("  timing plan")
    for name, seconds in SLIDE_SECONDS.items():
        bar = "#" * max(1, round(seconds / 10))
        print(f"    {seconds:>4}s  {bar:<19} {name}")
    print(f"    {'-' * 4}")
    print(f"    {total:>4}s  = {total // 60}m {total % 60:02d}s total "
          f"({(total - SLIDE_SECONDS['Demo']) // 60}m {(total - SLIDE_SECONDS['Demo']) % 60:02d}s slides "
          f"+ {SLIDE_SECONDS['Demo'] // 60}m demo)")


if __name__ == "__main__":
    main()
