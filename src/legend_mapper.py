# -*- coding: utf-8 -*-
"""
legend_mapper.py — deterministic curve → legend/sample mapping.

ChartDete already detects the legend (`legend_patch` color swatches +
`legend_label` text boxes). This module turns that into "which curve is which
sample" WITHOUT a VLM:

  1. read each legend patch's swatch color and OCR its adjacent label,
  2. sample each extracted curve's own color from the chart,
  3. assign every curve the legend label whose swatch color is nearest.

    from legend_mapper import map_curves_to_legend
    names = map_curves_to_legend(img_bgr, detections, series_px, ocr_reader)
    # names[i] is the sample label for curve i, or None if unmatched.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _dominant_color(pixels: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Dominant foreground color (BGR) of a pixel array, ignoring near-white /
    near-black / near-gray background. None if nothing colored remains."""
    if pixels.size == 0:
        return None
    p = pixels.reshape(-1, 3).astype(np.int16)
    mx = p.max(axis=1)
    mn = p.min(axis=1)
    sat = mx - mn                                   # crude saturation
    fg = p[(mx < 245) & (mn > 8) & (sat > 25)]      # drop white/black/gray
    if len(fg) < 3:
        # low-saturation (black/gray) lines: keep non-white, non-pure-white
        fg = p[(mx < 235)]
        if len(fg) < 3:
            return None
    med = np.median(fg, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _patch_color(img: np.ndarray, bbox) -> Optional[Tuple[int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return _dominant_color(img[y1:y2, x1:x2])


def _curve_color(img: np.ndarray, points_px: List[List[float]],
                 r: int = 1) -> Optional[Tuple[int, int, int]]:
    """Dominant color of a curve, sampled from small windows at its points."""
    H, W = img.shape[:2]
    samples = []
    step = max(1, len(points_px) // 60)             # ~60 samples along the curve
    for p in points_px[::step]:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < W and 0 <= y < H:
            samples.append(img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1])
    if not samples:
        return None
    return _dominant_color(np.concatenate([s.reshape(-1, 3) for s in samples]))


def _color_dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


def extract_legend_entries(img: np.ndarray, detections: Dict[str, Any],
                           ocr_reader=None) -> List[Dict[str, Any]]:
    """[{color:(b,g,r), label:str, patch_bbox, label_bbox}] from ChartDete
    legend detections. Each patch is paired with the nearest label box (to its
    right, then any), whose text is OCR'd."""
    patches = detections.get("legend_patch") or []
    labels = detections.get("legend_label") or []

    def _center(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _ocr(bbox):
        if ocr_reader is None:
            return ""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        H, W = img.shape[:2]
        pad = 2
        crop = img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
        if crop.size == 0:
            return ""
        try:
            res = ocr_reader.readtext(crop, detail=0)
            return " ".join(t.strip() for t in res if t and t.strip())
        except Exception:  # noqa: BLE001
            return ""

    entries = []
    used = set()
    for pb in patches:
        color = _patch_color(img, pb)
        if color is None:
            continue
        pcx, pcy = _center(pb)
        # nearest label whose vertical center is close and that sits to the right
        best, best_d = None, 1e9
        for j, lb in enumerate(labels):
            if j in used:
                continue
            lcx, lcy = _center(lb)
            if abs(lcy - pcy) > (pb[3] - pb[1]) * 1.5 + 8:
                continue
            d = abs(lcx - pcx) + abs(lcy - pcy) * 3      # prefer same row, right side
            if lcx < pcx:
                d += 500                                  # label left of patch is unusual
            if d < best_d:
                best, best_d = j, d
        label_text = ""
        label_bbox = None
        if best is not None:
            used.add(best)
            label_bbox = labels[best]
            label_text = _ocr(label_bbox)
        entries.append({"color": color, "label": label_text,
                        "patch_bbox": pb[:4], "label_bbox": label_bbox})
    return entries


def map_curves_to_legend(img: np.ndarray, detections: Dict[str, Any],
                         series_px: List[List[List[float]]],
                         ocr_reader=None, max_dist: float = 90.0
                         ) -> List[Optional[str]]:
    """Assign each extracted curve the nearest-color legend label. Returns a
    list aligned to series_px (None where no confident match). Greedy 1:1 when
    curve and legend counts match, else nearest-color per curve."""
    entries = [e for e in extract_legend_entries(img, detections, ocr_reader)
               if e["label"]]
    if not entries or not series_px:
        return [None] * len(series_px)
    curve_colors = [_curve_color(img, s) for s in series_px]

    names: List[Optional[str]] = [None] * len(series_px)
    # build all (curve, entry) distances
    pairs = []
    for ci, cc in enumerate(curve_colors):
        if cc is None:
            continue
        for ei, e in enumerate(entries):
            pairs.append((_color_dist(cc, e["color"]), ci, ei))
    pairs.sort()
    taken_c, taken_e = set(), set()
    one_to_one = len(series_px) == len(entries)
    for dist, ci, ei in pairs:
        if dist > max_dist:
            break
        if ci in taken_c:
            continue
        if one_to_one and ei in taken_e:
            continue
        names[ci] = entries[ei]["label"]
        taken_c.add(ci)
        taken_e.add(ei)
    return names
