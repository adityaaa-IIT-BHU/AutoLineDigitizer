#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression suite for marker_extractor (scatter point detection).

Five synthetic chart classes modeled on real failure modes:
  T1 tiny markers + error bars (low-res PDF panel crops)
  T2 markers on connecting lines + overlapping pair + gridlines
  T3 same-colour series split by marker shape
  T4 anti-aliased black-outlined markers + JPEG artifacts (series count!)
  T5 axis frame + inward tick marks (must NOT become points)

Run:  python tools/test_marker_extractor.py
"""
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from marker_extractor import MarkerExtractor  # noqa: E402


def run(name, img, plot, gt=None, want_series=None, want_total=None, tol=6):
    series, meta = MarkerExtractor(img, plot_area=plot).extract()
    alldet = [p for s in series for p in s]
    ok = True
    msg = f"{name}: {[(len(s), m['shape'], 'o' if m['hollow'] else 'f') for s, m in zip(series, meta)]}"
    if gt is not None:
        hit = sum(1 for (gx, gy) in gt
                  if any(abs(gx - dx) <= tol and abs(gy - dy) <= tol
                         for (dx, dy) in alldet))
        msg += f" recall={hit}/{len(gt)} n={len(alldet)}"
        ok = hit >= len(gt) - 1 and len(alldet) <= len(gt) + 2
    if want_total is not None:
        ok = ok and sum(len(s) for s in series) == want_total
    if want_series is not None:
        ok = ok and len(series) == want_series
    print(("PASS " if ok else "FAIL ") + msg)
    return ok


def main():
    fails = 0

    # T1: tiny markers with error bars, two series
    img = np.full((360, 380, 3), 255, np.uint8)
    rng = np.random.default_rng(0)
    orange = (30, 120, 200); blue = (180, 80, 60); gt = []
    for i in range(13):
        x = 40 + i * 25
        ya = 300 - i * 12 + int(rng.integers(-6, 6))
        yb = 180 - i * 4 + int(rng.integers(-6, 6))
        cv2.line(img, (x, ya - 8), (x, ya + 8), orange, 1)
        cv2.line(img, (x - 2, ya - 8), (x + 2, ya - 8), orange, 1)
        cv2.line(img, (x - 2, ya + 8), (x + 2, ya + 8), orange, 1)
        cv2.circle(img, (x, ya), 3, orange, -1)
        cv2.line(img, (x, yb - 7), (x, yb + 7), blue, 1)
        cv2.rectangle(img, (x - 3, yb - 3), (x + 3, yb + 3), blue, -1)
        gt += [(x, ya), (x, yb)]
    fails += not run("T1 tiny+errbars", img, (25, 15, 360, 330), gt=gt,
                     want_series=2, want_total=26)

    # T2: markers ON a line, overlapping pair, gridlines
    img2 = np.full((420, 640, 3), 255, np.uint8)
    for gx in range(80, 640, 80):
        cv2.line(img2, (gx, 20), (gx, 380), (215, 215, 215), 1)
    ptsA = [(60 + i * 55, 340 - i * 18) for i in range(10)]
    for i in range(9):
        cv2.line(img2, ptsA[i], ptsA[i + 1], (0, 0, 220), 2)
    for p in ptsA:
        cv2.circle(img2, p, 6, (0, 0, 220), -1)
    ptsB = ([(60 + i * 55, 180 - i * 6) for i in range(8)]
            + [(430, 132), (438, 128)])
    for (x, y) in ptsB:
        cv2.rectangle(img2, (x - 6, y - 6), (x + 6, y + 6), (200, 80, 0), 2)
    fails += not run("T2 line+overlap", img2, (40, 20, 600, 380),
                     gt=ptsA + ptsB, want_total=20)

    # T3: same-colour series split by shape
    img3 = np.full((400, 600, 3), 255, np.uint8)
    for i in range(8):
        cv2.circle(img3, (60 + i * 60, 300 - i * 12), 6, (0, 0, 220), -1)
        cv2.circle(img3, (60 + i * 60, 200 - i * 8), 6, (200, 80, 0), 2)
        p = np.array([[60 + i * 60, 120 - i * 5], [52 + i * 60, 136 - i * 5],
                      [68 + i * 60, 136 - i * 5]])
        cv2.fillPoly(img3, [p], (0, 0, 220))
    fails += not run("T3 shape-split", img3, (30, 20, 580, 360),
                     want_series=3, want_total=24)

    # T4: AA + black outlines + JPEG — exactly 2 series
    img4 = np.full((400, 560, 3), 255, np.uint8)
    gt4 = []
    rng4 = np.random.default_rng(7)
    for i in range(12):
        x = 50 + i * 40
        y1 = 320 - i * 14 + int(rng4.integers(-8, 8))
        y2 = 170 - i * 3 + int(rng4.integers(-8, 8))
        cv2.circle(img4, (x, y1), 6, (60, 160, 240), -1, lineType=cv2.LINE_AA)
        cv2.circle(img4, (x, y1), 6, (30, 30, 30), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(img4, (x - 5, y2 - 5), (x + 5, y2 + 5), (190, 100, 40),
                      -1, cv2.LINE_AA)
        cv2.rectangle(img4, (x - 5, y2 - 5), (x + 5, y2 + 5), (30, 30, 30),
                      1, cv2.LINE_AA)
        gt4 += [(x, y1), (x, y2)]
    _, enc = cv2.imencode('.jpg', img4, [cv2.IMWRITE_JPEG_QUALITY, 82])
    img4 = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    fails += not run("T4 AA+outline+jpeg", img4, (30, 20, 540, 360), gt=gt4,
                     want_series=2)

    # T5: axis frame + inward ticks must NOT become points. (No want_total:
    # one circle/square pair overlaps almost fully — arbitration collapsing
    # it to one point is intended behaviour.)
    img5 = np.full((400, 560, 3), 255, np.uint8)
    cv2.rectangle(img5, (50, 30), (520, 350), (20, 20, 20), 2)
    for tx in range(90, 520, 55):
        cv2.line(img5, (tx, 350), (tx, 340), (20, 20, 20), 2)
        cv2.line(img5, (tx, 30), (tx, 40), (20, 20, 20), 2)
    for ty in range(70, 350, 45):
        cv2.line(img5, (50, ty), (60, ty), (20, 20, 20), 2)
        cv2.line(img5, (520, ty), (510, ty), (20, 20, 20), 2)
    gt5 = []
    rng5 = np.random.default_rng(3)
    for i in range(10):
        x = 80 + i * 44
        y1 = 310 - i * 20 + int(rng5.integers(-6, 6))
        y2 = 150 + i * 8 + int(rng5.integers(-6, 6))
        cv2.circle(img5, (x, y1), 6, (0, 0, 220), -1)
        cv2.rectangle(img5, (x - 5, y2 - 5), (x + 5, y2 + 5), (200, 80, 0), 2)
        gt5 += [(x, y1), (x, y2)]
    fails += not run("T5 axis-ticks", img5, (50, 30, 520, 350), gt=gt5,
                     want_series=2)

    print("=" * 44)
    print("ALL PASS" if fails == 0 else f"{fails} TEST(S) FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(main())
