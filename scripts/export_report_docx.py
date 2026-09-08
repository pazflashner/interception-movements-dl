"""Convert a delivered report PDF into an editable .docx.

The PDFs are built by reports/build_final_reports.py, but rebuilding them needs
frozen-study CSVs that the advisor package does not ship, so this converts the
delivered PDF instead. The result is an editing surface, not a second source of
truth: edit here to draft wording, then fold agreed changes back into
reports/concise_article.py so the generated PDF stays authoritative.

Structure is recovered from the report's own typography, which separates roles
cleanly by point size:

    17.0 bold  document title          10.5 bold  subsection heading
    10.0       author block            10.5       body text
    12.0 bold  section heading          9.0       captions and references
                                        8.2       table cells (bold = header)

Table cells are emitted as individual spans sharing a row baseline, so cells are
grouped by rounded y into rows and clustered by x into columns. A row whose
first column is empty is treated as a wrapped continuation of the row above.
Figures are extracted as embedded images and placed in reading order.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message=r".*fitz.*")

import pymupdf
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

ROOT = Path(__file__).resolve().parents[1]

TITLE_SIZE = 17.0
SUBTITLE_SIZE = 10.0
SECTION_SIZE = 12.0
BODY_SIZE = 10.5
CAPTION_SIZE = 9.0
TABLE_SIZE = 8.2
# Page numbers and superscript digits share the smallest sizes; superscripts sit
# inside table or body blocks and are kept, standalone small blocks are dropped.
NOISE_SIZES = (8.5, 6.6)

ROW_TOLERANCE = 3.0      # points; cells on one visual row share a baseline
COLUMN_TOLERANCE = 6.0   # points; x jitter within a column


def _dominant_size(block: dict) -> float:
    counts: dict[float, int] = {}
    for line in block.get("lines", []):
        for span in line["spans"]:
            counts[round(span["size"], 1)] = counts.get(round(span["size"], 1), 0) + len(span["text"])
    return max(counts, key=counts.get) if counts else 0.0


def _spans(block: dict) -> list[dict]:
    return [s for line in block.get("lines", []) for s in line["spans"] if s["text"].strip()]


def _paragraph_text(block: dict) -> str:
    """Join spans into one string, inserting spaces only where the PDF implies them."""
    out: list[str] = []
    for line in block.get("lines", []):
        pieces = [s["text"] for s in line["spans"]]
        out.append("".join(pieces).strip())
    # reportlab wraps paragraphs across lines; rejoin with single spaces.
    return " ".join(part for part in out if part)


def _cluster(values: list[float], tolerance: float) -> list[float]:
    centres: list[float] = []
    for value in sorted(values):
        if centres and value - centres[-1] <= tolerance:
            continue
        centres.append(value)
    return centres


def _table_grid(spans: list[dict]) -> tuple[list[list[str]], list[bool]]:
    """Rebuild a cell grid plus a per-row flag marking bold header rows.

    Reportlab emits one span per cell and one block per visual row, so the grid
    is rebuilt from the absolute span geometry of a whole run of table blocks.
    Clustering columns across the entire table, rather than per row, keeps
    columns aligned when a row omits a trailing cell.
    """
    if not spans:
        return [], []
    # Superscripts (MMD squared) are typeset smaller and slightly right of their
    # base cell; fold them into that cell instead of opening a spurious column.
    spans = sorted(spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    body = [s for s in spans if s["size"] >= 7.0]
    marks = [s for s in spans if s["size"] < 7.0]
    row_keys = _cluster([s["bbox"][1] for s in body], ROW_TOLERANCE)
    col_keys = _cluster([s["bbox"][0] for s in body], COLUMN_TOLERANCE)

    def nearest(value: float, keys: list[float]) -> int:
        return min(range(len(keys)), key=lambda i: abs(keys[i] - value))

    grid = [["" for _ in col_keys] for _ in row_keys]
    bold = [False] * len(row_keys)
    for span in body:
        r = nearest(span["bbox"][1], row_keys)
        c = nearest(span["bbox"][0], col_keys)
        text = span["text"].strip()
        grid[r][c] = (grid[r][c] + " " + text).strip() if grid[r][c] else text
        if "Bold" in span["font"]:
            bold[r] = True
    for span in marks:
        r = nearest(span["bbox"][1], row_keys)
        c = nearest(span["bbox"][0], col_keys)
        grid[r][c] = grid[r][c] + span["text"].strip()

    # Merge wrapped continuation rows into the row above.
    merged_grid: list[list[str]] = []
    merged_bold: list[bool] = []
    for row, is_bold in zip(grid, bold):
        if merged_grid and not row[0] and any(row):
            for i, cell in enumerate(row):
                if cell:
                    merged_grid[-1][i] = (merged_grid[-1][i] + " " + cell).strip()
            continue
        merged_grid.append(list(row))
        merged_bold.append(is_bold)
    return merged_grid, merged_bold


def _add_table(document: Document, grid: list[list[str]], bold_rows: list[bool]) -> None:
    if not grid or not grid[0]:
        return
    table = document.add_table(rows=len(grid), cols=len(grid[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(grid):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(8.5)
            run.bold = bold_rows[r] if r < len(bold_rows) else False
    document.add_paragraph()


def _add_image(document: Document, doc: pymupdf.Document, xref: int, out_dir: Path, index: int) -> None:
    try:
        info = doc.extract_image(xref)
    except Exception:
        return
    path = out_dir / f"figure_{index:02d}.{info['ext']}"
    path.write_bytes(info["image"])
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(6.0))


def convert(pdf_path: Path, docx_path: Path) -> dict[str, int]:
    doc = pymupdf.open(pdf_path)
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    assets = docx_path.parent / (docx_path.stem + "_figures")
    assets.mkdir(parents=True, exist_ok=True)

    counts = {"headings": 0, "paragraphs": 0, "tables": 0, "figures": 0}
    seen_title = False
    title_paragraph = None
    figure_index = 0

    for page in doc:
        blocks = sorted(page.get_text("dict")["blocks"], key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
        pending: list[dict] = []

        def flush_table() -> None:
            """Emit buffered consecutive row-blocks as a single Word table."""
            nonlocal pending
            if not pending:
                return
            grid, bold_rows = _table_grid(pending)
            _add_table(document, grid, bold_rows)
            counts["tables"] += 1
            pending = []

        for block in blocks:
            if block["type"] == 1:
                continue  # images are placed from the xref list below
            size = _dominant_size(block)
            text = _paragraph_text(block)
            if not text:
                continue
            if size in NOISE_SIZES and len(text) <= 4:
                continue  # standalone page number
            if abs(size - TABLE_SIZE) >= 0.3:
                flush_table()
            if size >= TITLE_SIZE:
                # A long title wraps into consecutive blocks; keep it one heading.
                if seen_title:
                    title_paragraph.add_run(" " + text)
                else:
                    title_paragraph = document.add_heading(text, level=0)
                    seen_title = True
                    counts["headings"] += 1
            elif abs(size - SUBTITLE_SIZE) < 0.2 and counts["paragraphs"] == 0:
                paragraph = document.add_paragraph(text)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.italic = True
            elif abs(size - SECTION_SIZE) < 0.2:
                document.add_heading(text, level=1)
                counts["headings"] += 1
            elif abs(size - BODY_SIZE) < 0.2 and any("Bold" in s["font"] for s in _spans(block)):
                document.add_heading(text, level=2)
                counts["headings"] += 1
            elif abs(size - TABLE_SIZE) < 0.3:
                pending.extend(_spans(block))
            elif abs(size - CAPTION_SIZE) < 0.3:
                paragraph = document.add_paragraph(text)
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
                counts["paragraphs"] += 1
            else:
                document.add_paragraph(text)
                counts["paragraphs"] += 1

        flush_table()
        for image in page.get_images(full=True):
            figure_index += 1
            _add_image(document, doc, image[0], assets, figure_index)
            counts["figures"] += 1

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(docx_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path,
                        default=ROOT / "output/pdf/Interception_Movements_Final_Scientific_Report.pdf")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.pdf.exists():
        sys.exit(f"not found: {args.pdf}")
    out = args.out or (ROOT / "output" / "docx" / (args.pdf.stem + ".docx"))
    counts = convert(args.pdf, out)
    print(f"{out}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    print("  Editable draft converted from the delivered PDF. Check tables and figure")
    print("  placement before sharing; reports/concise_article.py remains authoritative.")


if __name__ == "__main__":
    main()
