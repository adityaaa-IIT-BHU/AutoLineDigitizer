# -*- coding: utf-8 -*-
"""Unit tests for line_merger.merge_curves.

Pure-logic tests (numpy only) — no torch/mmdet/flet required.
Curves are pixel-space [[x, y], ...] lists on a nominal 500x500 image.
"""
import numpy as np

from line_merger import merge_curves

SHAPE = (500, 500)


def _line(x0, x1, y_fn):
    return [[x, int(round(y_fn(x)))] for x in range(x0, x1 + 1)]


def test_empty_and_single_curve_pass_through():
    assert merge_curves([], img_shape=SHAPE) == []
    one = [_line(0, 100, lambda x: 200)]
    assert len(merge_curves(one, img_shape=SHAPE)) == 1


def test_exact_duplicates_merge():
    a = _line(50, 400, lambda x: 100 + 0.2 * x)
    b = _line(50, 400, lambda x: 101 + 0.2 * x)   # same line, 1px off
    merged = merge_curves([a, b], img_shape=SHAPE)
    assert len(merged) == 1


def test_distinct_parallel_curves_are_kept():
    a = _line(50, 400, lambda x: 100)
    b = _line(50, 400, lambda x: 130)              # 30px apart: distinct
    merged = merge_curves([a, b], img_shape=SHAPE)
    assert len(merged) == 2


def test_nested_extent_bundle_is_protected():
    # Battery-cycling signature: near-coincident curves that end at
    # different x must NOT be collapsed (extent-Jaccard gate).
    a = _line(50, 200, lambda x: 100)
    b = _line(50, 400, lambda x: 101)
    merged = merge_curves([a, b], img_shape=SHAPE)
    assert len(merged) == 2


def test_sequential_fragments_join():
    # One straight line split into two pieces with a small gap.
    a = _line(50, 200, lambda x: 100 + 0.5 * x)
    b = _line(215, 400, lambda x: 100 + 0.5 * x)   # gap of 15px, collinear
    merged = merge_curves([a, b], img_shape=SHAPE)
    assert len(merged) == 1
    xs = [p[0] for p in merged[0]]
    assert min(xs) == 50 and max(xs) == 400


def test_fragments_at_inconsistent_angles_do_not_join():
    a = _line(50, 200, lambda x: 100 + 0.5 * x)    # rising
    b = _line(215, 400, lambda x: 300 - 0.5 * x)   # falling, far endpoint
    merged = merge_curves([a, b], img_shape=SHAPE)
    assert len(merged) == 2


def test_drop_lower_keeps_higher_score_points():
    a = _line(50, 400, lambda x: 100)
    b = _line(50, 400, lambda x: 102)
    merged = merge_curves([a, b], img_shape=SHAPE, scores=[0.9, 0.4],
                          dup_action="drop_lower")
    assert len(merged) == 1
    ys = np.array([p[1] for p in merged[0]])
    assert np.all(ys == 100)                       # kept a, dropped b


def test_min_span_filter_drops_specks():
    long = _line(50, 400, lambda x: 100)
    speck = _line(200, 210, lambda x: 300)
    merged = merge_curves([long, speck], img_shape=SHAPE, min_span_frac=0.05)
    assert len(merged) == 1
