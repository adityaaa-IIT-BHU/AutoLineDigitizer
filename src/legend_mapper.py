# -*- coding: utf-8 -*-
"""
legend_mapper.py — deterministic curve → legend/sample mapping.

ChartDete detects the legend (`legend_patch` swatches, `legend_label` text,
and a `legend_area` box). This turns that into "which curve is which sample"
WITHOUT a VLM:

  1. read each legend swatch's color + OCR its label (patch/label pairs, with
     a fallback that segments the whole `legend_area` when the per-patch boxes
     are incomplete),
  2. sample each extracted curve's own color from the chart (mode, not mean,
     so a colored line over a white gap doesn't wash out),
  3. assign every curve the legend label whose swatch color is nearest
     (optimal 1:1 when the counts match, else nearest-per-curve).

    from legend_mapper import map_curves_to_legend
    names = map_curves_to_legend(img_bgr, detections, series_px, ocr_reader)
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _dominant_color(pixels: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Dominant FOREGROUND color (BGR): drop near-white/black/gray, then take
    the largest color cluster (mode) rather than the median, so a swatch or
    curve sampled together with background pixels keeps its true hue."""
    if pixels.size == 0:
        return None
    p = pixels.reshape(-1, 3).astype(np.int16)
    mx, mn = p.max(axis=1), p.min(axis=1)
    sat = mx - mn
    fg = p[(mx < 245) & (mn > 8) & (sat > 22)]      # coloured foreground
    if len(fg) < 3:
        fg = p[mx < 235]                             # dark/gray lines (no hue)
        if len(fg) < 3:
            return None
    # mode over a coarse 24-level quantization -> mean of the biggest cluster
    q = (fg // 24)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    sel = fg[keys == vals[counts.argmax()]]
    m = sel.mean(axis=0)
    return (int(m[0]), int(m[1]), int(m[2]))


def _crop(img, bbox, pad=0):
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    H, W = img.shape[:2]
    return img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]


def _curve_color(img: np.ndarray, points_px: List[List[float]], r: int = 1
                 ) -> Optional[Tuple[int, int, int]]:
    """Dominant color of a curve, sampled from small windows at its points."""
    H, W = img.shape[:2]
    win = []
    step = max(1, len(points_px) // 80)
    for p in points_px[::step]:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < W and 0 <= y < H:
            win.append(img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3))
    if not win:
        return None
    return _dominant_color(np.concatenate(win))


def _color_dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _pairs_from_boxes(img, patches, labels, ocr_reader) -> List[Dict[str, Any]]:
    """Pair each legend_patch swatch with its nearest legend_label and OCR it."""
    def _ocr(bbox):
        if ocr_reader is None:
            return ""
        crop = _crop(img, bbox, pad=2)
        if crop.size == 0:
            return ""
        try:
            res = ocr_reader.readtext(crop, detail=0)
            return " ".join(t.strip() for t in res if t and t.strip())
        except Exception:  # noqa: BLE001
            return ""

    entries, used = [], set()
    for pb in patches:
        color = _dominant_color(_crop(img, pb))
        if color is None:
            continue
        pcx, pcy = _center(pb)
        best, best_d = None, 1e9
        for j, lb in enumerate(labels):
            if j in used:
                continue
            lcx, lcy = _center(lb)
            if abs(lcy - pcy) > (pb[3] - pb[1]) * 1.6 + 10:
                continue
            d = abs(lcx - pcx) + abs(lcy - pcy) * 3 + (400 if lcx < pcx else 0)
            if d < best_d:
                best, best_d = j, d
        label = ""
        if best is not None:
            used.add(best)
            label = _ocr(labels[best])
        entries.append({"color": color, "label": label,
                        "patch_bbox": list(pb[:4])})
    return entries


def _from_legend_area(img, area_bbox, ocr_reader) -> List[Dict[str, Any]]:
    """Fallback: OCR the whole legend_area, and for each text line read the
    swatch color from the strip just LEFT of the text (handles charts where
    ChartDete found the area but not clean per-patch boxes)."""
    if ocr_reader is None:
        return []
    x1, y1, x2, y2 = [int(v) for v in area_bbox[:4]]
    crop = _crop(img, area_bbox)
    if crop.size == 0:
        return []
    try:
        res = ocr_reader.readtext(crop)     # [(box, text, conf), ...]
    except Exception:  # noqa: BLE001
        return []
    entries = []
    for box, text, conf in res:
        text = (text or "").strip()
        if not text or conf < 0.25:
            continue
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        tx1, ty1, ty2 = int(min(xs)), int(min(ys)), int(max(ys))
        h = max(6, ty2 - ty1)
        # swatch strip immediately left of the text, same row
        sw = crop[max(0, ty1 - 2):ty2 + 2, max(0, tx1 - int(2.2 * h)):tx1]
        color = _dominant_color(sw) if sw.size else None
        if color is None:                    # marker may sit under/after text
            row = crop[max(0, ty1 - 2):ty2 + 2, :]
            color = _dominant_color(row)
        if color is None:
            continue
        entries.append({"color": color, "label": text, "patch_bbox": None,
                        "label_bbox": [x1 + tx1, y1 + ty1, x1 + int(max(xs)), y1 + ty2]})
    return entries


def extract_legend_entries(img: np.ndarray, detections: Dict[str, Any],
                           ocr_reader=None) -> List[Dict[str, Any]]:
    """[{color:(b,g,r), label:str, ...}] for the chart legend. Uses per-patch
    boxes first; if those yield few labeled entries, falls back to segmenting
    the whole legend_area."""
    patches = detections.get("legend_patch") or []
    labels = detections.get("legend_label") or []
    area = detections.get("legend_area") or []

    box_entries = _pairs_from_boxes(img, patches, labels, ocr_reader)
    labeled = [e for e in box_entries if e["label"]]

    # If per-patch pairing missed labels (or found none), try the area fallback
    # and keep whichever set has more usable (labeled) entries.
    if area and len(labeled) < max(1, len(patches)):
        area_entries = [e for e in _from_legend_area(img, area[0], ocr_reader) if e["label"]]
        if len(area_entries) > len(labeled):
            return area_entries
    return labeled if labeled else box_entries


def _assign(curve_colors, entries, max_dist):
    """Optimal 1:1 when counts match (Hungarian), else nearest-per-curve."""
    n, m = len(curve_colors), len(entries)
    names: List[Optional[str]] = [None] * n
    valid = [i for i, c in enumerate(curve_colors) if c is not None]
    if not valid or not entries:
        return names
    if n == m and n == len(valid):
        try:
            from scipy.optimize import linear_sum_assignment
            cost = np.array([[_color_dist(cc, e["color"]) for e in entries]
                             for cc in curve_colors])
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] <= max_dist * 1.4:     # a touch looser for 1:1
                    names[r] = entries[c]["label"]
            return names
        except Exception:  # noqa: BLE001
            pass
    # greedy nearest, one entry per curve (entries may repeat if fewer)
    pairs = sorted((_color_dist(cc, e["color"]), i, j)
                   for i, cc in enumerate(curve_colors) if cc is not None
                   for j, e in enumerate(entries))
    taken_c = set()
    for dist, i, j in pairs:
        if dist > max_dist or i in taken_c:
            continue
        names[i] = entries[j]["label"]
        taken_c.add(i)
    return names


def map_curves_to_legend(img: np.ndarray, detections: Dict[str, Any],
                         series_px: List[List[List[float]]],
                         ocr_reader=None, max_dist: float = 95.0
                         ) -> List[Optional[str]]:
    """Assign each extracted curve its nearest-color legend label. Returns a
    list aligned to series_px (None where no confident match)."""
    entries = extract_legend_entries(img, detections, ocr_reader)
    entries = [e for e in entries if e.get("label")]
    if not entries or not series_px:
        return [None] * len(series_px)
    curve_colors = [_curve_color(img, s) for s in series_px]
    return _assign(curve_colors, entries, max_dist)
