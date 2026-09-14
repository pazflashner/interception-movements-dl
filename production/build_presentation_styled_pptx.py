"""Export the styled deck as an editable PowerPoint file.

Run:  python production/build_presentation_styled_pptx.py

Same content as the PDF renderer - both read production/styled_slides.py - but
written as native PowerPoint shapes, so every text box, table cell and card can
be edited in PowerPoint. Speaker-facing timings become slide notes.

Each slide is laid out twice: once measured with nothing drawn, then again at a
scale chosen so the content reaches the bottom margin. Sparse slides therefore
come out in large type filling the frame, rather than as a small block stranded
under the title. Slide titles stay at a fixed size so headers line up across the
deck.

Needs python-pptx (pure Python); unlike the course PPTX chain it does not need
Node. Two things cannot be reproduced exactly: the three-stop gradient rule
becomes two two-stop halves, and card corners use the rounded-rectangle geometry
PowerPoint provides.
"""
from __future__ import annotations

from pathlib import Path
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.parts.image import Image as PptxImage
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production.styled_slides import BACKUP, SLIDES, TITLE

OUTPUT = ROOT / "production" / "Interception_Movements_Presentation_Styled.pptx"

W, H = Inches(13.333), Inches(7.5)            # 16:9
M = Inches(0.85)
INNER = W - 2 * M
TOP = Inches(1.30)                            # first content row, under the rule
FLOOR = H - Inches(0.38)                      # content is grown to reach this
MIN_SCALE, MAX_SCALE = 0.85, 2.00

NAVY = RGBColor(0x0E, 0x17, 0x29)
INK = RGBColor(0x16, 0x1E, 0x2E)
BODY = RGBColor(0x3D, 0x47, 0x56)
MUTED = RGBColor(0x73, 0x7E, 0x8B)
RULE = RGBColor(0xE1, 0xE6, 0xEC)
CARD = RGBColor(0xF8, 0xFA, 0xFC)
BLUE = RGBColor(0x1B, 0x6F, 0xEA)
CYAN = RGBColor(0x3E, 0xB1, 0xEC)
TEAL = RGBColor(0x17, 0xA2, 0xA2)
VIOLET = RGBColor(0x7A, 0x5C, 0xF0)
GREEN = RGBColor(0x1D, 0x7A, 0x3E)
GREEN_BG = RGBColor(0xEA, 0xF7, 0xEE)
AMBER = RGBColor(0xE0, 0xA1, 0x1B)
AMBER_BG = RGBColor(0xFE, 0xF8, 0xE3)
AMBER_INK = RGBColor(0x8A, 0x62, 0x16)
INFO_BG = RGBColor(0xEF, 0xF5, 0xFC)
PILL_BG = RGBColor(0xE3, 0xF0, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# Calibri averages a little over half an em per character; 0.53 leaves margin so
# the measuring pass over-counts lines rather than under-counting them.
CHAR_EM, LINE_EM = 0.53, 1.30


def _char_w(pt: float) -> float:
    return Inches(CHAR_EM * pt / 72)


def _line_h(pt: float) -> float:
    return Inches(LINE_EM * pt / 72)


def _lines(text, w: float, pt: float) -> int:
    """Greedy word wrap, to estimate how tall a run of text will be."""
    per_line = max(6, int(w / _char_w(pt)))
    count, filled = 1, 0
    for word in str(text).split():
        step = len(word) + (1 if filled else 0)
        if filled and filled + step > per_line:
            count, filled = count + 1, len(word)
        else:
            filled += step
    return count


def _rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None):
    if slide is None:                          # measuring pass draws nothing
        return None
    box = slide.shapes.add_shape(shape, int(x), int(y), int(w), int(h))
    if fill is None:
        box.fill.background()
    else:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(0.75)
    box.shadow.inherit = False
    return box


def _text(slide, x, y, w, h, runs, size=11, color=BODY, bold=False,
          align=PP_ALIGN.LEFT, font="Calibri"):
    """runs: a string, or a list of (text, bold) tuples for mixed weight."""
    if slide is None:
        return None
    box = slide.shapes.add_textbox(int(x), int(y), int(w), int(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    para = frame.paragraphs[0]
    para.alignment = align
    for text, is_bold in ([(runs, bold)] if isinstance(runs, str) else runs):
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = is_bold
        run.font.color.rgb = color
        run.font.name = font
    return box


def _gradient_rule(slide):
    """Blue -> teal -> violet. PowerPoint gives two stops per shape, so use two
    halves rather than a strip of slivers - it keeps the deck editable."""
    half = int(W / 2) + Emu(9000)               # overlap so no seam shows
    for i, (start, end) in enumerate(((BLUE, TEAL), (TEAL, VIOLET))):
        box = _rect(slide, int(W / 2) * i, 0, half, Inches(0.085), start)
        box.fill.gradient()
        box.fill.gradient_angle = 0.0           # left to right
        stops = box.fill.gradient_stops
        stops[0].color.rgb, stops[0].position = start, 0.0
        stops[1].color.rgb, stops[1].position = end, 1.0


def _pill(slide, text, right_x, y, fg, bg, size=10, centre=False):
    w = _char_w(size) * len(text) + Inches(0.40)
    x = (W - w) / 2 if centre else right_x - w
    box = _rect(slide, x, y, w, _line_h(size) + Inches(0.16),
                bg, MSO_SHAPE.ROUNDED_RECTANGLE)
    box.adjustments[0] = 0.5
    frame = box.text_frame
    frame.margin_left = frame.margin_right = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = frame.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = fg
    run.font.name = "Calibri"


def _image_aspect(path: Path) -> float:
    """Height over width, read without adding the picture to a slide."""
    image = PptxImage.from_file(str(path))
    px_w, px_h = image.size
    return px_h / px_w


class Builder:
    """Renders slides. `s` scales every content block; headers stay fixed."""

    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.s = 1.0
        self.g = 1.0

    # -- scaled units --------------------------------------------------------
    def U(self, inches: float) -> int:
        return int(Inches(inches) * self.s)

    def F(self, points: float) -> float:
        return points * self.s

    def G(self, inches: float) -> int:
        """A gap between blocks: scaled, and widened by the slide's spacing."""
        return int(Inches(inches) * self.s * self.g)

    # -- chrome --------------------------------------------------------------
    def _slide(self, dark=False):
        slide = self.prs.slides.add_slide(self.blank)
        _rect(slide, 0, 0, W, H, NAVY if dark else WHITE)
        _gradient_rule(slide)
        return slide

    def title(self, spec):
        slide = self._slide(dark=True)
        _text(slide, M, Inches(1.95), INNER, Inches(1.7), spec["title"],
              size=38, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        _text(slide, M, Inches(3.45), INNER, Inches(0.6), spec["subtitle"],
              size=16, color=RGBColor(0x9A, 0xA6, 0xB5), align=PP_ALIGN.CENTER)
        _text(slide, M, Inches(4.35), INNER, Inches(0.45), spec["authors"],
              size=18, color=CYAN, bold=True, align=PP_ALIGN.CENTER)
        for i, line in enumerate(spec["affiliation"]):
            _text(slide, M, Inches(4.98 + 0.30 * i), INNER, Inches(0.34), line,
                  size=12, color=RGBColor(0x8A, 0x95, 0xA4), align=PP_ALIGN.CENTER)
        _pill(slide, spec["badge"], 0, Inches(6.05),
              RGBColor(0xC8, 0xD2, 0xDE), RGBColor(0x20, 0x2A, 0x3A),
              size=12, centre=True)
        return slide

    def header(self, slide, title, seconds=None, index=None, total=None, backup=False):
        _text(slide, M, Inches(0.32), INNER - Inches(2.1), Inches(0.62), title,
              size=24, color=INK, bold=True)
        if backup:
            _pill(slide, "BACKUP", W - M, Inches(0.38), MUTED,
                  RGBColor(0xEF, 0xF1, 0xF4), 11)
        else:
            _pill(slide, f"{seconds} s  |  {index}/{total}", W - M, Inches(0.38),
                  BLUE, PILL_BG, 11)
        _rect(slide, M, Inches(1.04), INNER, Emu(9525), RULE)

    # -- blocks: each returns the y to continue from -------------------------
    def table(self, slide, spec, x, y, w, min_row=0.36, font=10.5):
        pt = self.F(font)
        rows = spec["rows"]
        mark = spec.get("mark")
        n_rows, n_cols = len(rows), len(rows[0])
        weights = [max(len(str(r[i])) for r in rows) for i in range(n_cols)]
        total = sum(weights) or 1
        widths = [int(w * weight / total) for weight in weights]
        pad = self.U(0.09)
        heights = []
        for row in rows:
            wrapped = max(_lines(v, widths[c] - 2 * pad, pt) for c, v in enumerate(row))
            heights.append(max(self.U(min_row), int(wrapped * _line_h(pt)) + 2 * pad))
        height = sum(heights)
        if slide is None:
            return y + height + self.G(0.20)

        shape = slide.shapes.add_table(n_rows, n_cols, int(x), int(y), int(w), height)
        table = shape.table
        for i, width in enumerate(widths):
            table.columns[i].width = width
        for r, row in enumerate(rows):
            table.rows[r].height = heights[r]
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.text = str(value)
                cell.margin_left = cell.margin_right = pad
                cell.margin_top = cell.margin_bottom = self.U(0.03)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                if r == 0:
                    cell.fill.fore_color.rgb = NAVY
                elif r == mark:
                    cell.fill.fore_color.rgb = GREEN_BG
                else:
                    cell.fill.fore_color.rgb = CARD if r % 2 == 0 else WHITE
                for run in cell.text_frame.paragraphs[0].runs:
                    run.font.size = Pt(pt)
                    run.font.name = "Calibri"
                    if r == 0:
                        run.font.bold, run.font.color.rgb = True, WHITE
                    elif r == mark:
                        run.font.bold, run.font.color.rgb = True, GREEN
                    else:
                        run.font.bold = c == 0
                        run.font.color.rgb = INK if c == 0 else BODY
        return y + height + self.G(0.20)

    def callout(self, slide, kind, title, text, x, y, w):
        """A tinted band. `text` is a string, or a list of (paragraph, bold)
        pairs when a lead sentence should stand bold on its own line. A falsy
        `title` leaves the band carrying its body alone."""
        warn = kind == "warn"
        tp, bp = self.F(12), self.F(11)
        pad = self.U(0.24)
        gap = self.U(0.10)
        inner = w - 2 * pad - self.U(0.10)
        paragraphs = [(text, False)] if isinstance(text, str) else list(text)
        head = (("! " if warn else "") + title) if title else ""
        head_h = int(_lines(head, inner, tp) * _line_h(tp)) if head else 0
        para_h = [int(_lines(t, inner, bp) * _line_h(bp)) for t, _ in paragraphs]
        height = (self.U(0.18) + head_h + (gap if head else 0) + sum(para_h)
                  + gap * (len(paragraphs) - 1) + self.U(0.18))
        if slide is not None:
            _rect(slide, x, y, w, height, AMBER_BG if warn else INFO_BG)
            _rect(slide, x, y, self.U(0.06), height, AMBER if warn else BLUE)
            ty = y + self.U(0.16)
            if head:
                _text(slide, x + pad, ty, inner, head_h, head, size=tp,
                      color=AMBER_INK if warn else BLUE, bold=True)
                ty += head_h + gap
            for (paragraph, bold), h in zip(paragraphs, para_h):
                _text(slide, x + pad, ty, inner, h, paragraph, size=bp, bold=bold,
                      color=INK if bold else BODY)
                ty += h + gap
        return y + height + self.G(0.16)

    def card_height(self, spec, w):
        pad = self.U(0.26)
        inner = w - 2 * pad
        tp, bp, ep = self.F(18), self.F(11), self.F(10.5)
        height = self.U(0.26)
        if spec.get("eyebrow"):
            height += int(_line_h(ep)) + self.U(0.08)
        height += int(_lines(spec["title"], inner, tp) * _line_h(tp)) + self.U(0.12)
        height += int(_lines(spec["text"], inner, bp) * _line_h(bp)) + self.U(0.26)
        return height

    def card(self, slide, spec, x, y, w, h):
        if slide is None:
            return y + h
        pad = self.U(0.26)
        inner = w - 2 * pad
        tp, bp, ep = self.F(18), self.F(11), self.F(10.5)
        _rect(slide, x, y, w, h, CARD, MSO_SHAPE.ROUNDED_RECTANGLE,
              line=RGBColor(0xE5, 0xE9, 0xF0))
        ty = y + self.U(0.26)
        if spec.get("eyebrow"):
            _text(slide, x + pad, ty, inner, int(_line_h(ep)), spec["eyebrow"],
                  size=ep, bold=True,
                  color=GREEN if spec.get("accent") == "green" else BLUE)
            ty += int(_line_h(ep)) + self.U(0.08)
        n_title = _lines(spec["title"], inner, tp)
        _text(slide, x + pad, ty, inner, int(n_title * _line_h(tp)), spec["title"],
              size=tp, color=INK, bold=True)
        ty += int(n_title * _line_h(tp)) + self.U(0.12)
        _text(slide, x + pad, ty, inner,
              int(_lines(spec["text"], inner, bp) * _line_h(bp)), spec["text"], size=bp)
        return y + h

    def stat(self, slide, value, title, text, x, y, w):
        vp, tp, bp = self.F(52), self.F(14), self.F(11)
        inner = w - self.U(0.50)
        value_h, title_h = int(_line_h(vp)), int(_line_h(tp))
        body_h = int(_lines(text, inner, bp) * _line_h(bp))
        height = (self.U(0.44) + value_h + self.U(0.18) + title_h + self.U(0.16)
                  + body_h + self.U(0.44))
        if slide is not None:
            _rect(slide, x, y, w, height, GREEN_BG, MSO_SHAPE.ROUNDED_RECTANGLE)
            ty = y + self.U(0.44)
            _text(slide, x, ty, w, value_h, value, size=vp, color=GREEN, bold=True,
                  align=PP_ALIGN.CENTER)
            ty += value_h + self.U(0.18)
            _text(slide, x, ty, w, title_h, title, size=tp, color=INK, bold=True,
                  align=PP_ALIGN.CENTER)
            ty += title_h + self.U(0.16)
            _text(slide, x + self.U(0.25), ty, inner, body_h, text, size=bp,
                  align=PP_ALIGN.CENTER)
        return y + height + self.G(0.16)

    def equation(self, slide, text, x, y, w):
        pt = self.F(13)
        inner = w - self.U(0.40)
        n = _lines(text, inner, pt)
        height = self.U(0.20) + int(n * _line_h(pt)) + self.U(0.20)
        if slide is not None:
            _rect(slide, x, y, w, height, RGBColor(0xFA, 0xFB, 0xFD),
                  MSO_SHAPE.ROUNDED_RECTANGLE, line=RGBColor(0xE5, 0xE9, 0xF0))
            _text(slide, x + self.U(0.20), y + self.U(0.20), inner,
                  int(n * _line_h(pt)), text, size=pt, color=INK, align=PP_ALIGN.CENTER)
        return y + height + self.G(0.18)

    def terminal(self, slide, spec, x, y, w):
        lp, bp, cp = self.F(10), self.F(11), self.F(12)
        badge_h = int(_line_h(bp)) + self.U(0.16)
        height = self.U(0.18) + int(_line_h(lp)) + self.U(0.12) + badge_h + self.U(0.18)
        if slide is not None:
            _rect(slide, x, y, w, height, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE)
            _text(slide, x + self.U(0.26), y + self.U(0.18), w - self.U(0.52),
                  int(_line_h(lp)), spec["label"], size=lp, color=CYAN)
            by = y + self.U(0.18) + int(_line_h(lp)) + self.U(0.12)
            badge_w = _char_w(bp) * len(spec["badge"]) + self.U(0.40)
            badge = _rect(slide, x + self.U(0.26), by, badge_w, badge_h, BLUE,
                          MSO_SHAPE.ROUNDED_RECTANGLE)
            frame = badge.text_frame
            frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER
            run = para.add_run()
            run.text, run.font.size, run.font.bold = spec["badge"], Pt(bp), True
            run.font.color.rgb, run.font.name = WHITE, "Calibri"
            _text(slide, x + self.U(0.26) + badge_w + self.U(0.30),
                  by + (badge_h - int(_line_h(cp))) // 2,
                  w - self.U(0.92) - badge_w, int(_line_h(cp)), spec["command"],
                  size=cp, color=RGBColor(0xD6, 0xDE, 0xE8), font="Consolas")
        return y + height + self.G(0.16)

    def figure(self, slide, name, x, y, w, max_h):
        path = ROOT / name
        if not path.exists():
            _text(slide, x, y, w, Inches(0.3), f"[missing: {name}]", size=9, color=MUTED)
            return y + Inches(0.3)
        width, height = int(w), int(w * _image_aspect(path))
        if max_h > 0 and height > max_h:
            width, height = int(width * max_h / height), int(max_h)
        if slide is not None:
            slide.shapes.add_picture(str(path), int(x + (w - width) / 2), int(y),
                                     width=width, height=height)
        return y + height + self.G(0.16)

    def body(self, slide, text, x, y, w):
        """Full-width lead paragraphs: one string, or a list of them."""
        pt = self.F(12)
        for paragraph in ([text] if isinstance(text, str) else text):
            n = _lines(paragraph, w, pt)
            _text(slide, x, y, w, int(n * _line_h(pt)), paragraph, size=pt)
            y += int(n * _line_h(pt)) + self.G(0.16)
        return y

    def column(self, slide, blocks, x, y, w, floor):
        for block in blocks:
            kind = block[0]
            if kind in ("heading", "heading-alt"):
                pt = self.F(14)
                n = _lines(block[1], w, pt)
                _text(slide, x, y, w, int(n * _line_h(pt)), block[1], size=pt,
                      bold=True, color=VIOLET if kind == "heading-alt" else INK)
                y += int(n * _line_h(pt)) + self.G(0.14)
            elif kind == "text":
                pt = self.F(11.5)
                n = _lines(block[1], w, pt)
                _text(slide, x, y, w, int(n * _line_h(pt)), block[1], size=pt)
                y += int(n * _line_h(pt)) + self.G(0.16)
            elif kind == "bullet":
                pt = self.F(11.5)
                n = _lines(f"|  {block[1]} {block[2]}", w, pt)
                _text(slide, x, y, w, int(n * _line_h(pt)),
                      [("•  ", True), (block[1] + " ", True), (block[2], False)],
                      size=pt, color=BODY)
                y += int(n * _line_h(pt)) + self.G(0.18)
            elif kind == "table":
                y = self.table(slide, block[1], x, y, w)
            elif kind == "equation":
                y = self.equation(slide, block[1], x, y, w)
            elif kind == "card":
                spec = {"title": block[1], "text": block[2]}
                if len(block) > 3:
                    spec["eyebrow"] = block[3]
                if len(block) > 4:
                    spec["accent"] = block[4]
                h = self.card_height(spec, w)
                self.card(slide, spec, x, y, w, h)
                y += h + self.G(0.20)
            elif kind == "stat":
                y = self.stat(slide, block[1], block[2], block[3], x, y, w)
            elif kind == "callout-info":
                y = self.callout(slide, "info", block[1], block[2], x, y, w)
            elif kind == "figure":
                cap = floor - y
                if len(block) > 2:
                    cap = min(cap, int(Inches(block[2]) * self.s))
                y = self.figure(slide, block[1], x, y, w, cap)
        return y

    # -- slide assembly ------------------------------------------------------
    def layout(self, slide, spec, index=None, total=None, backup=False):
        """Draw (slide given) or measure (slide None); returns the content bottom."""
        if slide is not None:
            self.header(slide, spec["title"], spec.get("seconds"), index, total, backup)
        y = TOP
        if spec.get("body"):
            y = self.body(slide, spec["body"], M, y, INNER)
        if spec.get("columns"):
            gap = self.U(0.40)
            cw = (INNER - gap) / 2
            sides = (("left", M), ("right", M + cw + gap))
            # Measure both columns first so the shorter one can be centred
            # against the taller, instead of leaving a block of white beneath it.
            depths = {key: self.column(None, spec["columns"][key], x, y, cw, FLOOR) - y
                      for key, x in sides}
            tallest = max(depths.values())
            for key, x in sides:
                self.column(slide, spec["columns"][key],
                            x, y + (tallest - depths[key]) // 2, cw, FLOOR)
            y += tallest
        if spec.get("table"):
            y = self.table(slide, spec["table"], M, y, INNER, min_row=0.48, font=11.5)
        if spec.get("cards"):
            cards = spec["cards"]
            gap = self.U(0.34)
            cw = (INNER - gap * (len(cards) - 1)) / len(cards)
            h = max(self.card_height(card, cw) for card in cards)
            for i, card in enumerate(cards):
                self.card(slide, card, M + i * (cw + gap), y, cw, h)
            y += h + self.G(0.22)
        if spec.get("equation"):
            y = self.equation(slide, spec["equation"], M, y, INNER)
        if spec.get("terminal"):
            y = self.terminal(slide, spec["terminal"], M, y, INNER)
        if spec.get("figure"):
            name, cap = spec["figure"], FLOOR - y
            if not isinstance(name, str):
                name, cap = name[0], min(cap, int(Inches(name[1]) * self.s))
            y = self.figure(slide, name, M, y, INNER, cap)
        if spec.get("callout"):
            c = spec["callout"]
            y = self.callout(slide, c["kind"], c["title"], c["text"], M, y, INNER)
        return y

    def fit(self, spec) -> float:
        """Largest scale whose content still clears the floor.

        Content height grows monotonically with the scale, so bisect: measuring
        passes draw nothing, which makes a dozen of them free.
        """
        self.g = spec.get("spacing", 1.0)
        self.s = MIN_SCALE
        if self.layout(None, spec) > FLOOR:
            return MIN_SCALE                   # already too tall at the minimum
        self.s = MAX_SCALE
        if self.layout(None, spec) <= FLOOR:
            return MAX_SCALE
        lo, hi = MIN_SCALE, MAX_SCALE
        for _ in range(14):
            mid = (lo + hi) / 2
            self.s = mid
            if self.layout(None, spec) <= FLOOR:
                lo = mid
            else:
                hi = mid
        return lo

    def render(self, spec, index=None, total=None, backup=False):
        self.s = self.fit(spec)
        slide = self._slide()
        self.layout(slide, spec, index, total, backup)
        note = []
        if spec.get("seconds"):
            note.append(f"Planned: {spec['seconds']} seconds. Slide {index} of {total}.")
        if spec.get("notes"):
            note.append(spec["notes"])
        if note:
            slide.notes_slide.notes_text_frame.text = "\n\n".join(note)
        return slide

    def save(self):
        self.prs.save(str(OUTPUT))


def build() -> Path:
    b = Builder()
    b.title(TITLE)
    total = len(SLIDES) + 1
    for i, spec in enumerate(SLIDES, start=2):
        b.render(spec, i, total)
    for spec in BACKUP:
        b.render(spec, backup=True)
    b.save()
    return OUTPUT


def main() -> None:
    try:
        path = build()
    except PermissionError:
        raise SystemExit(
            f"Cannot write {OUTPUT.name} - it is open in PowerPoint. "
            "Close the deck and run this again.")
    seconds = sum(s.get("seconds", 0) for s in SLIDES)
    print(f"{path}\n  {1 + len(SLIDES) + len(BACKUP)} slides, editable in PowerPoint")
    print(f"  core talk: {seconds // 60}m {seconds % 60:02d}s   "
          f"({600 - seconds}s spare against 10 minutes)")
    print("  Content: production/styled_slides.py - shared with the PDF renderer.")


if __name__ == "__main__":
    main()
