# -*- coding: utf-8 -*-
"""
text_extract.py — MinerU-style LLM-ready markdown from a born-digital PDF.

Uses the already-bundled PP-DocLayoutV2 model (layout classes + reading-order
head, vendored from MinerU — see MINERU_LICENSE.md) to segment each page, then
pulls the actual text per block from the PDF text layer via PyMuPDF. No OCR,
no extra models: the same weights that power figure detection also fix the
two-column reading order, strip headers/footers/page numbers, and tag titles,
captions and tables — which is what makes the output "LLM ready".

    from mineru_layout.text_extract import pdf_to_markdown
    res = pdf_to_markdown("paper.pdf", detector=chart_detector)
    res["markdown"]   # reading-ordered markdown, or res is None for scanned PDFs

Scanned PDFs (no text layer) return None — callers fall back to sending the
raw PDF to the model (vision path).
"""

import re
from typing import Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

# Block classes that are page furniture — never part of the body text.
_SKIP_LABELS = {
    "header", "header_image", "footer", "footer_image", "number", "seal",
    "aside_text",  # margin notes (journal DOI strips etc.)
}
# Figure-like classes: represented by a placeholder (captions carry the info).
_FIGURE_LABELS = {"image", "chart"}
# Reference-list classes (skipped by default — long and rarely needed).
_REFERENCE_LABELS = {"reference", "reference_content"}


def _clean_text(txt: str) -> str:
    """De-hyphenate line breaks and collapse a block's lines into one flow."""
    txt = re.sub(r"-\s*\n\s*", "", txt)          # hyphenation at line end
    txt = re.sub(r"\s*\n\s*", " ", txt)          # line breaks -> spaces
    return re.sub(r"[ \t]{2,}", " ", txt).strip()


def _block_text(page, bbox_px, zoom: float, pad_pt: float = 2.0) -> str:
    """Text-layer content of one layout block (bbox in rendered-image pixels)."""
    x0, y0, x1, y1 = [v / zoom for v in bbox_px]
    rect = fitz.Rect(x0 - pad_pt, y0 - pad_pt, x1 + pad_pt, y1 + pad_pt) & page.rect
    if rect.is_empty:
        return ""
    return _clean_text(page.get_text("text", clip=rect))


def _table_markdown(page, bbox_px, zoom: float) -> Optional[str]:
    """Try PyMuPDF's table finder on the block region; None if it fails."""
    x0, y0, x1, y1 = [v / zoom for v in bbox_px]
    rect = fitz.Rect(x0, y0, x1, y1) & page.rect
    try:
        tabs = page.find_tables(clip=rect)
        if tabs.tables:
            return tabs.tables[0].to_markdown().strip()
    except Exception:
        pass
    return None


def pdf_to_markdown(pdf_path: str, detector, dpi: int = 200,
                    max_pages: Optional[int] = None,
                    include_references: bool = False) -> Optional[Dict]:
    """
    Convert a born-digital PDF into reading-ordered markdown.

    detector: a mineru_layout.ChartDetector (its .detect_layout() returns
        PP-DocLayoutV2 blocks already sorted by the reading-order head).
    Returns {"markdown", "n_pages", "n_blocks", "n_figures", "n_tables"} or
    None when the PDF has no usable text layer (scanned — use the vision path).
    """
    if fitz is None:
        return None
    import numpy as np

    doc = fitz.open(pdf_path)
    n_pages = min(len(doc), max_pages) if max_pages else len(doc)

    # Scanned-PDF guard: require a real text layer on most pages.
    text_chars = sum(len(doc[i].get_text()) for i in range(n_pages))
    if text_chars < 200 * max(1, n_pages // 2):
        doc.close()
        return None

    zoom = dpi / 72.0
    out: List[str] = []
    n_blocks = n_figs = n_tabs = 0
    refs_seen = False

    for pno in range(n_pages):
        page = doc[pno]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        blocks = detector.detect_layout(img)  # RGB in; reading-ordered out

        page_parts: List[str] = [f"*[page {pno + 1}]*"]
        # 'reference' is the list frame; skip it when its items are also detected
        page_labels = {b.get("label") for b in blocks}
        skip_ref_frame = "reference_content" in page_labels

        for b in blocks:
            label = b.get("label", "")
            bbox = b.get("bbox")
            if not bbox or label in _SKIP_LABELS:
                continue
            if label in _REFERENCE_LABELS:
                refs_seen = True
                if not include_references or (label == "reference" and skip_ref_frame):
                    continue
            if label in _FIGURE_LABELS:
                n_figs += 1
                page_parts.append(f"*[{label.upper()} on page {pno + 1} — see caption below/above]*")
                continue
            if label == "inline_formula":
                continue  # lives inside its text block; avoid duplication

            if label == "table":
                n_tabs += 1
                md = _table_markdown(page, bbox, zoom)
                if md is None:
                    raw = _block_text(page, bbox, zoom)
                    md = f"```table\n{raw}\n```" if raw else ""
                if md:
                    page_parts.append(md)
                n_blocks += 1
                continue

            txt = _block_text(page, bbox, zoom)
            if not txt:
                continue
            n_blocks += 1
            if label == "doc_title":
                page_parts.append(f"# {txt}")
            elif label == "paragraph_title":
                page_parts.append(f"## {txt}")
            elif label == "abstract":
                page_parts.append(f"**Abstract.** {txt}")
            elif label == "figure_title":
                page_parts.append(f"> **{txt}**")
            elif label in ("vision_footnote", "footnote"):
                page_parts.append(f"> {txt}")
            elif label in ("display_formula", "formula_number"):
                page_parts.append(f"$$ {txt} $$")
            else:  # text, algorithm, content, vertical_text, reference_content
                page_parts.append(txt)

        # Fallback: layout found nothing textual on a page that has text.
        if len(page_parts) == 1:
            whole = _clean_text(page.get_text())
            if whole:
                page_parts.append(whole)

        out.append("\n\n".join(page_parts))

    doc.close()
    md = "\n\n".join(out)
    if refs_seen and not include_references:
        md += "\n\n*(reference list omitted)*"
    return {
        "markdown": md,
        "n_pages": n_pages,
        "n_blocks": n_blocks,
        "n_figures": n_figs,
        "n_tables": n_tabs,
    }
