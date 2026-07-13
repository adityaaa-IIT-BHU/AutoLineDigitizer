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

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
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


def _crop_png(img: "np.ndarray", bbox, max_dim: int = 1000,
              pad_px: int = 8) -> Optional[bytes]:
    """PNG bytes of a layout-block crop, downscaled to cap vision tokens."""
    try:
        from PIL import Image
        import io
        H, W = img.shape[:2]
        x0 = max(0, int(bbox[0]) - pad_px); y0 = max(0, int(bbox[1]) - pad_px)
        x1 = min(W, int(bbox[2]) + pad_px); y1 = min(H, int(bbox[3]) + pad_px)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        pil = Image.fromarray(img[y0:y1, x0:x1])
        if max(pil.size) > max_dim:
            s = max_dim / max(pil.size)
            pil = pil.resize((int(pil.width * s), int(pil.height * s)))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full-MinerU backend (optional). The real MinerU pipeline adds what the
# lightweight path above cannot do: formulas as LaTeX (dedicated recognition
# model — the PDF text layer garbles math-font glyphs), tables as HTML, and
# OCR for scanned PDFs. It is a ~2GB install, so it lives in its OWN venv and
# is used via its CLI whenever one is found; everything falls back to the
# lightweight bundled path otherwise.
# ---------------------------------------------------------------------------

def find_mineru_cli() -> Optional[str]:
    """Locate a full-MinerU CLI: $ALD_MINERU_CLI, the repo's
    external/mineru-venv, or PATH. None -> use the lightweight path."""
    cand = os.environ.get("ALD_MINERU_CLI")
    if cand:
        return cand if os.path.exists(cand) else None
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.dirname(src_dir)
    local = os.path.join(repo, "external", "mineru-venv", "bin", "mineru")
    if os.path.exists(local):
        return local
    return shutil.which("mineru")


def _file_png(path: str, max_dim: int = 1000) -> Optional[bytes]:
    """PNG bytes of a saved crop image, downscaled to cap vision tokens."""
    try:
        from PIL import Image
        import io
        pil = Image.open(path).convert("RGB")
        if max(pil.size) > max_dim:
            s = max_dim / max(pil.size)
            pil = pil.resize((int(pil.width * s), int(pil.height * s)))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _content_list_markdown(cl: List[Dict], images_dir: str,
                           include_references: bool = True,
                           return_figures: bool = False) -> Dict:
    """Rebuild our marker-annotated markdown from MinerU's content_list.json
    (reading-ordered items with page_idx/type/text/img_path)."""
    out: List[str] = []
    figures: List[Dict] = []
    n_blocks = n_figs = n_tabs = 0
    cur_page = None

    for item in cl:
        t = item.get("type")
        if t in ("header", "footer", "page_number", "aside_text"):
            continue
        pg = int(item.get("page_idx", 0)) + 1
        if pg != cur_page:
            out.append(f"*[page {pg}]*")
            cur_page = pg

        if t in ("image", "chart", "table"):
            label = "table" if t == "table" else ("chart" if t == "chart" else "image")
            cap = " ".join(item.get(f"{t}_caption") or [])
            foot = " ".join(item.get(f"{t}_footnote") or [])
            png = None
            if return_figures and item.get("img_path"):
                png = _file_png(os.path.join(images_dir, item["img_path"]))
            if png is not None:
                figures.append({"index": len(figures) + 1, "page": pg,
                                "label": label, "png": png})
                out.append(f"*[TABLE crop {len(figures)} — page {pg}; crop attached]*"
                           if t == "table" else
                           f"*[FIGURE {len(figures)} ({label}) — page {pg}; "
                           f"crop attached — see caption below/above]*")
            else:
                out.append(f"*[{label.upper()} on page {pg} — see caption below/above]*")
            if cap:
                out.append(f"> **{cap}**")
            if foot:
                out.append(f"> {foot}")
            if t == "table":
                n_tabs += 1
                body = (item.get("table_body") or "").strip()
                if body:
                    out.append(body)   # HTML table — models read it natively
                n_blocks += 1
            else:
                n_figs += 1
            continue

        if t == "equation":
            txt = (item.get("text") or "").strip()
            if txt:
                out.append(txt if txt.startswith("$$") else f"$$ {txt} $$")
                n_blocks += 1
            continue

        if t == "list":
            if item.get("sub_type") == "ref_text" and not include_references:
                continue
            items = [str(x).strip() for x in (item.get("list_items") or []) if str(x).strip()]
            out.extend(items)
            n_blocks += len(items)
            continue

        txt = (item.get("text") or "").strip()
        if not txt:
            continue
        lvl = item.get("text_level")
        if lvl:
            out.append("#" * min(int(lvl), 4) + " " + txt)
        elif t == "page_footnote":
            out.append(f"> {txt}")
        else:
            out.append(txt)
        n_blocks += 1

    return {
        "markdown": "\n\n".join(out),
        "n_pages": (max((int(i.get("page_idx", 0)) for i in cl), default=-1) + 1),
        "n_blocks": n_blocks,
        "n_figures": n_figs,
        "n_tables": n_tabs,
        "figures": figures,
    }


def pdf_to_markdown_full(pdf_path: str, cli_path: str,
                         include_references: bool = True,
                         return_figures: bool = False,
                         timeout_sec: int = 1800) -> Optional[Dict]:
    """Parse with the full MinerU CLI (formulas->LaTeX, tables->HTML, OCR for
    scans). Same return shape as pdf_to_markdown; None on any failure so the
    caller can fall back to the lightweight path. First run per machine
    downloads ~2GB of models."""
    tmp = tempfile.mkdtemp(prefix="mineru_full_")
    try:
        env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}  # xet stalls on weak Wi-Fi
        proc = subprocess.run([cli_path, "-p", pdf_path, "-o", tmp, "-b", "pipeline"],
                              capture_output=True, text=True, timeout=timeout_sec,
                              env=env)
        if proc.returncode != 0:
            return None
        # MinerU sanitizes the output dir name — glob instead of guessing it.
        hits = glob.glob(os.path.join(tmp, "*", "*", "*_content_list.json"))
        if not hits:
            return None
        with open(hits[0], "r", encoding="utf-8") as f:
            cl = json.load(f)
        res = _content_list_markdown(cl, os.path.dirname(hits[0]),
                                     include_references=include_references,
                                     return_figures=return_figures)
        return res if res["markdown"].strip() else None
    except Exception:  # noqa: BLE001 — optional backend, never break the caller
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pdf_to_markdown(pdf_path: str, detector, dpi: int = 200,
                    max_pages: Optional[int] = None,
                    include_references: bool = True,
                    return_figures: bool = False) -> Optional[Dict]:
    """
    Convert a born-digital PDF into reading-ordered markdown — everything the
    paper contains: body text, titles, captions, tables, footnotes, and (by
    default) the reference list.

    detector: a mineru_layout.ChartDetector (its .detect_layout() returns
        PP-DocLayoutV2 blocks already sorted by the reading-order head).
    return_figures: also return PNG crops of every figure/chart/table block —
        each crop k corresponds to a `*[FIGURE k …]*` / `*[TABLE crop k …]*`
        marker in the markdown, so a vision model can match image to context.
    Returns {"markdown", "n_pages", "n_blocks", "n_figures", "n_tables",
    "figures": [{"index","page","label","png"}...]} or None when the PDF has
    no usable text layer (scanned — use the vision path).
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
    figures: List[Dict] = []

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
                if return_figures:
                    png = _crop_png(img, bbox)
                    if png is not None:
                        figures.append({"index": len(figures) + 1, "page": pno + 1,
                                        "label": label, "png": png})
                        page_parts.append(
                            f"*[FIGURE {len(figures)} ({label}) — page {pno + 1}; "
                            f"crop attached — see caption below/above]*")
                        continue
                page_parts.append(f"*[{label.upper()} on page {pno + 1} — see caption below/above]*")
                continue
            if label == "inline_formula":
                continue  # lives inside its text block; avoid duplication

            if label == "table":
                n_tabs += 1
                if return_figures:
                    png = _crop_png(img, bbox)
                    if png is not None:
                        figures.append({"index": len(figures) + 1, "page": pno + 1,
                                        "label": "table", "png": png})
                        page_parts.append(
                            f"*[TABLE crop {len(figures)} — page {pno + 1}; crop attached]*")
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
        "figures": figures,
    }
