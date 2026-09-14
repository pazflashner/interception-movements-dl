"""Render the styled course deck: dark title slide, light content slides.

Run:  python production/build_presentation_styled.py

The visual language follows the Hebrew project deck - gradient rule, timing pill,
dark-header tables with a green winning row, cards, an amber caveat callout and a
dark terminal strip. Content lives in production/styled_slides.py, which is that
deck's text translated to English; edit there, re-run, get a new PDF.

Only reportlab is required, so the deck rebuilds on a machine without Node.
"""
from __future__ import annotations

from pathlib import Path
import sys

from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production.styled_slides import BACKUP, SLIDES, TITLE

OUTPUT = ROOT / "production" / "presentation_styled.pdf"
W, H = 338.7 * mm, 190.5 * mm

NAVY = (0.055, 0.090, 0.161)
INK = (0.086, 0.118, 0.180)
BODY = (0.239, 0.278, 0.337)
MUTED = (0.451, 0.494, 0.545)
RULE = (0.882, 0.902, 0.925)
CARD = (0.973, 0.980, 0.988)
CARD_EDGE = (0.898, 0.914, 0.937)
BLUE = (0.106, 0.435, 0.918)
CYAN = (0.243, 0.694, 0.925)
TEAL = (0.090, 0.635, 0.635)
VIOLET = (0.478, 0.361, 0.941)
GREEN = (0.114, 0.478, 0.243)
GREEN_BG = (0.918, 0.969, 0.933)
AMBER = (0.878, 0.631, 0.106)
AMBER_BG = (0.996, 0.973, 0.890)
AMBER_INK = (0.541, 0.384, 0.086)
INFO_BG = (0.937, 0.961, 0.988)
PILL_BG = (0.890, 0.937, 0.988)

M = 22 * mm                      # page margin
COL_GAP = 10 * mm


def wrap(c, text, font, size, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.n = 0

    # ── furniture ───────────────────────────────────────────────────────────
    def _gradient(self):
        steps = 260
        for i in range(steps):
            t = i / (steps - 1)
            a, b, u = (BLUE, TEAL, t * 2) if t < 0.5 else (TEAL, VIOLET, (t - 0.5) * 2)
            self.c.setFillColorRGB(*[a[k] + (b[k] - a[k]) * u for k in range(3)])
            self.c.rect(W * t, H - 3.4 * mm, W / steps + 0.7, 3.4 * mm, stroke=0, fill=1)

    def _pill(self, text, x, y, fg, bg):
        w = self.c.stringWidth(text, "Helvetica-Bold", 9.5) + 8 * mm
        self.c.setFillColorRGB(*bg)
        self.c.roundRect(x, y, w, 7.6 * mm, 3.8 * mm, stroke=0, fill=1)
        self.c.setFillColorRGB(*fg)
        self.c.setFont("Helvetica-Bold", 9.5)
        self.c.drawString(x + 4 * mm, y + 2.6 * mm, text)
        return w

    def _page(self, dark=False):
        if self.n:
            self.c.showPage()
        self.n += 1
        self.c.setFillColorRGB(*(NAVY if dark else (1, 1, 1)))
        self.c.rect(0, 0, W, H, stroke=0, fill=1)
        self._gradient()

    # ── slide kinds ─────────────────────────────────────────────────────────
    def title_slide(self, spec):
        self._page(dark=True)
        y = H / 2 + 30 * mm
        self.c.setFillColorRGB(1, 1, 1)
        for line in wrap(self.c, spec["title"], "Helvetica-Bold", 38, W - 80 * mm):
            self.c.setFont("Helvetica-Bold", 38)
            self.c.drawCentredString(W / 2, y, line)
            y -= 15 * mm
        self.c.setFillColorRGB(0.60, 0.66, 0.74)
        self.c.setFont("Helvetica", 14)
        self.c.drawCentredString(W / 2, y - 2 * mm, spec["subtitle"])
        self.c.setFillColorRGB(*CYAN)
        self.c.setFont("Helvetica-Bold", 15)
        self.c.drawCentredString(W / 2, y - 24 * mm, spec["authors"])
        self.c.setFillColorRGB(0.55, 0.60, 0.68)
        self.c.setFont("Helvetica", 10.5)
        for i, line in enumerate(spec["affiliation"]):
            self.c.drawCentredString(W / 2, y - 34 * mm - i * 6 * mm, line)
        badge = spec["badge"]
        w = self.c.stringWidth(badge, "Helvetica-Bold", 9.5) + 8 * mm
        self._pill(badge, (W - w) / 2, 32 * mm, (0.78, 0.84, 0.91), (0.129, 0.165, 0.231))

    def header(self, title, seconds=None, index=None, total=None, backup=False):
        self._page()
        self.c.setFillColorRGB(*INK)
        self.c.setFont("Helvetica-Bold", 23)
        for line in wrap(self.c, title, "Helvetica-Bold", 23, W - 2 * M - 46 * mm)[:1]:
            self.c.drawString(M, H - 21 * mm, line)
        if backup:
            self._pill("BACKUP", M if False else W - M - 24 * mm, H - 23 * mm,
                       MUTED, (0.937, 0.945, 0.953))
        else:
            label = f"{seconds} s   |   {index}/{total}"
            w = self.c.stringWidth(label, "Helvetica-Bold", 9.5) + 8 * mm
            self._pill(label, W - M - w, H - 23 * mm, BLUE, PILL_BG)
        self.c.setStrokeColorRGB(*RULE)
        self.c.setLineWidth(0.8)
        self.c.line(M, H - 29 * mm, W - M, H - 29 * mm)

    # ── blocks ──────────────────────────────────────────────────────────────
    def heading(self, text, x, y, w, alt=False):
        self.c.setFillColorRGB(*(VIOLET if alt else INK))
        self.c.setFont("Helvetica-Bold", 12.5)
        for line in wrap(self.c, text, "Helvetica-Bold", 12.5, w):
            self.c.drawString(x, y, line)
            y -= 6.4 * mm
        return y - 1.5 * mm

    def text(self, text, x, y, w, size=11):
        self.c.setFillColorRGB(*BODY)
        for line in wrap(self.c, text, "Helvetica", size, w):
            self.c.setFont("Helvetica", size)
            self.c.drawString(x, y, line)
            y -= size * 1.42
        return y - 2 * mm

    def bullet(self, lead, rest, x, y, w, size=11):
        self.c.setFillColorRGB(*BLUE)
        self.c.circle(x + 1.3 * mm, y + 1.4 * mm, 1.1 * mm, stroke=0, fill=1)
        lead_w = self.c.stringWidth(lead + " ", "Helvetica-Bold", size)
        self.c.setFillColorRGB(*INK)
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(x + 5.5 * mm, y, lead)
        first = wrap(self.c, rest, "Helvetica", size, w - 5.5 * mm - lead_w)
        self.c.setFillColorRGB(*BODY)
        self.c.setFont("Helvetica", size)
        if first:
            self.c.drawString(x + 5.5 * mm + lead_w, y, first[0])
        y -= size * 1.42
        for line in wrap(self.c, " ".join(rest.split()[len(first[0].split()):]) if first else rest,
                         "Helvetica", size, w - 5.5 * mm):
            self.c.drawString(x + 5.5 * mm, y, line)
            y -= size * 1.42
        return y - 1.5 * mm

    def table(self, spec, x, y, w, size=10):
        rows, mark = spec["rows"], spec.get("mark")
        weights = [max(self.c.stringWidth(str(r[i]), "Helvetica", size) for r in rows)
                   for i in range(len(rows[0]))]
        total = sum(weights) or 1
        widths = [w * wt / total for wt in weights]
        head_h = 10 * mm
        self.c.setFillColorRGB(*NAVY)
        self.c.rect(x, y - head_h, w, head_h, stroke=0, fill=1)
        cx = x
        for value, cw in zip(rows[0], widths):
            self.c.setFillColorRGB(1, 1, 1)
            self.c.setFont("Helvetica-Bold", size - 0.5)
            for line in wrap(self.c, str(value), "Helvetica-Bold", size - 0.5, cw - 6 * mm)[:1]:
                self.c.drawString(cx + 3 * mm, y - head_h + 3.6 * mm, line)
            cx += cw
        y -= head_h
        for r, row in enumerate(rows[1:], start=1):
            cells = [wrap(self.c, str(v), "Helvetica", size, cw - 6 * mm)
                     for v, cw in zip(row, widths)]
            rh = max(8 * mm, max(len(cl) for cl in cells) * 4.9 * mm + 3.4 * mm)
            if r == mark:
                self.c.setFillColorRGB(*GREEN_BG)
                self.c.rect(x, y - rh, w, rh, stroke=0, fill=1)
            elif r % 2 == 0:
                self.c.setFillColorRGB(*CARD)
                self.c.rect(x, y - rh, w, rh, stroke=0, fill=1)
            cx = x
            for ci, (lines, cw) in enumerate(zip(cells, widths)):
                ty = y - 5 * mm
                for line in lines:
                    if r == mark:
                        self.c.setFillColorRGB(*GREEN)
                        self.c.setFont("Helvetica-Bold", size)
                    else:
                        self.c.setFillColorRGB(*(INK if ci == 0 else BODY))
                        self.c.setFont("Helvetica-Bold" if ci == 0 else "Helvetica", size)
                    self.c.drawString(cx + 3 * mm, ty, line)
                    ty -= 4.9 * mm
                cx += cw
            y -= rh
        self.c.setStrokeColorRGB(*RULE)
        self.c.setLineWidth(0.6)
        self.c.line(x, y, x + w, y)
        return y - 3 * mm

    def callout(self, kind, title, text, x, y, w):
        """`text` may be a list of (paragraph, bold); a falsy `title` is omitted."""
        warn = kind == "warn"
        paragraphs = [(text, False)] if isinstance(text, str) else list(text)
        wrapped = [
            (wrap(self.c, t, "Helvetica-Bold" if b else "Helvetica", 10.5, w - 14 * mm), b)
            for t, b in paragraphs
        ]
        n_lines = sum(len(lines) for lines, _ in wrapped)
        head_h = 6.5 * mm if title else 0
        h = 6.5 * mm + head_h + n_lines * 5.2 * mm + (len(wrapped) - 1) * 2 * mm
        self.c.setFillColorRGB(*(AMBER_BG if warn else INFO_BG))
        self.c.rect(x, y - h, w, h, stroke=0, fill=1)
        self.c.setFillColorRGB(*(AMBER if warn else BLUE))
        self.c.rect(x + w - 1.6 * mm, y - h, 1.6 * mm, h, stroke=0, fill=1)
        ty = y - 7.5 * mm
        if title:
            self.c.setFillColorRGB(*(AMBER_INK if warn else BLUE))
            self.c.setFont("Helvetica-Bold", 11.5)
            self.c.drawString(x + 6 * mm, ty, ("! " if warn else "") + title)
            ty -= head_h
        for lines, bold in wrapped:
            self.c.setFillColorRGB(*(INK if bold else BODY))
            for line in lines:
                self.c.setFont("Helvetica-Bold" if bold else "Helvetica", 10.5)
                self.c.drawString(x + 6 * mm, ty, line)
                ty -= 5.2 * mm
            ty -= 2 * mm
        return y - h - 3 * mm

    def card(self, spec, x, y, w, h):
        self.c.setFillColorRGB(*CARD)
        self.c.setStrokeColorRGB(*CARD_EDGE)
        self.c.setLineWidth(0.7)
        self.c.roundRect(x, y - h, w, h, 3 * mm, stroke=1, fill=1)
        ty = y - 9 * mm
        if spec.get("eyebrow"):
            self.c.setFillColorRGB(*(GREEN if spec.get("accent") == "green" else BLUE))
            self.c.setFont("Helvetica-Bold", 10)
            self.c.drawString(x + 6 * mm, ty, spec["eyebrow"])
            ty -= 8 * mm
        self.c.setFillColorRGB(*INK)
        self.c.setFont("Helvetica-Bold", 17)
        self.c.drawString(x + 6 * mm, ty, spec["title"])
        ty -= 9 * mm
        self.c.setFillColorRGB(*BODY)
        for line in wrap(self.c, spec["text"], "Helvetica", 10.5, w - 12 * mm):
            self.c.setFont("Helvetica", 10.5)
            self.c.drawString(x + 6 * mm, ty, line)
            ty -= 5.2 * mm

    def stat(self, value, title, text, x, y, w, h):
        self.c.setFillColorRGB(*GREEN_BG)
        self.c.roundRect(x, y - h, w, h, 3 * mm, stroke=0, fill=1)
        self.c.setFillColorRGB(*GREEN)
        self.c.setFont("Helvetica-Bold", 46)
        self.c.drawCentredString(x + w / 2, y - h / 2 + 6 * mm, value)
        self.c.setFillColorRGB(*INK)
        self.c.setFont("Helvetica-Bold", 13)
        self.c.drawCentredString(x + w / 2, y - h / 2 - 6 * mm, title)
        self.c.setFillColorRGB(*BODY)
        ty = y - h / 2 - 15 * mm
        for line in wrap(self.c, text, "Helvetica", 10, w - 14 * mm):
            self.c.setFont("Helvetica", 10)
            self.c.drawCentredString(x + w / 2, ty, line)
            ty -= 5 * mm

    def equation(self, text, x, y, w):
        lines = wrap(self.c, text, "Helvetica", 12, w - 12 * mm)
        h = 8 * mm + len(lines) * 6.4 * mm
        self.c.setFillColorRGB(0.980, 0.984, 0.992)
        self.c.setStrokeColorRGB(*CARD_EDGE)
        self.c.setLineWidth(0.7)
        self.c.roundRect(x, y - h, w, h, 2.5 * mm, stroke=1, fill=1)
        self.c.setFillColorRGB(*INK)
        ty = y - 8 * mm
        for line in lines:
            self.c.setFont("Helvetica", 12)
            self.c.drawCentredString(x + w / 2, ty, line)
            ty -= 6.4 * mm
        return y - h - 3 * mm

    def terminal(self, spec, x, y, w):
        h = 20 * mm
        self.c.setFillColorRGB(*NAVY)
        self.c.roundRect(x, y - h, w, h, 2.5 * mm, stroke=0, fill=1)
        self.c.setFillColorRGB(*CYAN)
        self.c.setFont("Helvetica", 9)
        self.c.drawRightString(x + w - 6 * mm, y - 6.5 * mm, spec["label"])
        self.c.setFillColorRGB(*BLUE)
        bw = self.c.stringWidth(spec["badge"], "Helvetica-Bold", 10) + 8 * mm
        self.c.roundRect(x + 6 * mm, y - 15 * mm, bw, 8 * mm, 2 * mm, stroke=0, fill=1)
        self.c.setFillColorRGB(1, 1, 1)
        self.c.setFont("Helvetica-Bold", 10)
        self.c.drawString(x + 10 * mm, y - 12.4 * mm, spec["badge"])
        self.c.setFillColorRGB(0.88, 0.91, 0.95)
        self.c.setFont("Courier-Bold", 11)
        self.c.drawRightString(x + w - 6 * mm, y - 12.4 * mm, spec["command"])
        return y - h - 3 * mm

    def figure(self, name, x, y, w, max_h):
        path = ROOT / name
        if not path.exists():
            self.c.setFillColorRGB(*MUTED)
            self.c.setFont("Helvetica-Oblique", 9)
            self.c.drawString(x, y - 6 * mm, f"[missing: {name}]")
            return y - 10 * mm
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        s = min(w / iw, max_h / ih)
        self.c.drawImage(img, x + (w - iw * s) / 2, y - ih * s, iw * s, ih * s, mask="auto")
        return y - ih * s - 3 * mm

    def column(self, blocks, x, y, w):
        for block in blocks:
            kind = block[0]
            if kind == "heading":
                y = self.heading(block[1], x, y, w)
            elif kind == "heading-alt":
                y = self.heading(block[1], x, y, w, alt=True)
            elif kind == "text":
                y = self.text(block[1], x, y, w)
            elif kind == "bullet":
                y = self.bullet(block[1], block[2], x, y, w)
            elif kind == "table":
                y = self.table(block[1], x, y, w)
            elif kind == "equation":
                y = self.equation(block[1], x, y, w)
            elif kind == "card":
                spec = {"title": block[1], "text": block[2]}
                if len(block) > 3:
                    spec["eyebrow"] = block[3]
                if len(block) > 4:
                    spec["accent"] = block[4]
                self.card(spec, x, y, w, 40 * mm)
                y -= 44 * mm
            elif kind == "stat":
                self.stat(block[1], block[2], block[3], x, y, w, 62 * mm)
                y -= 66 * mm
            elif kind == "callout-info":
                y = self.callout("info", block[1], block[2], x, y, w)
            elif kind == "figure":
                cap = (block[2] * 25.4 * mm) if len(block) > 2 else 76 * mm
                y = self.figure(block[1], x, y, w, cap)
        return y

    def render(self, slide, index=None, total=None, backup=False):
        self.header(slide["title"], slide.get("seconds"), index, total, backup)
        y = H - 40 * mm
        inner = W - 2 * M
        if slide.get("body"):
            lead = slide["body"]
            for paragraph in ([lead] if isinstance(lead, str) else lead):
                y = self.text(paragraph, M, y, inner, size=12)
        if slide.get("columns"):
            cw = (inner - COL_GAP) / 2
            left = self.column(slide["columns"]["left"], M, y, cw)
            right = self.column(slide["columns"]["right"], M + cw + COL_GAP, y, cw)
            y = min(left, right)
        if slide.get("table"):
            y = self.table(slide["table"], M, y, inner)
        if slide.get("cards"):
            cards = slide["cards"]
            cw = (inner - COL_GAP * (len(cards) - 1)) / len(cards)
            for i, spec in enumerate(cards):
                self.card(spec, M + i * (cw + COL_GAP), y, cw, 46 * mm)
            y -= 50 * mm
        if slide.get("equation"):
            y = self.equation(slide["equation"], M, y, inner)
        if slide.get("terminal"):
            y = self.terminal(slide["terminal"], M, y, inner)
        if slide.get("figure"):
            name = slide["figure"]
            cap = 60 * mm if isinstance(name, str) else name[1] * 25.4 * mm
            y = self.figure(name if isinstance(name, str) else name[0], M, y, inner, cap)
        if slide.get("callout"):
            c = slide["callout"]
            self.callout(c["kind"], c["title"], c["text"], M, max(y, 30 * mm), inner)

    def save(self):
        self.c.showPage()
        self.c.save()


def build() -> Path:
    d = Deck(OUTPUT)
    d.title_slide(TITLE)
    total = len(SLIDES) + 1
    for i, slide in enumerate(SLIDES, start=2):
        d.render(slide, i, total)
    for slide in BACKUP:
        d.render(slide, backup=True)
    d.save()
    return OUTPUT


def main() -> None:
    path = build()
    seconds = sum(s.get("seconds", 0) for s in SLIDES)
    print(f"{path}\n  {1 + len(SLIDES) + len(BACKUP)} slides "
          f"({1 + len(SLIDES)} core + {len(BACKUP)} backup)\n")
    print("  timing plan")
    for i, s in enumerate(SLIDES, start=2):
        sec = s.get("seconds", 0)
        print(f"    {sec:>4}s  {'#' * max(1, round(sec / 10)):<9} {s['title'][:54]}")
    print(f"    {'-' * 4}\n    {seconds:>4}s  = {seconds // 60}m {seconds % 60:02d}s core talk")
    print(f"    {600 - seconds}s spare against 10 minutes."
          if seconds <= 600 else f"    OVER by {seconds - 600}s.")


if __name__ == "__main__":
    main()
