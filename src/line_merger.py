"""Post-hoc merging of over-segmented LineFormer curves.

LineFormer frequently emits several instances for one plotted line: duplicate
masks of the same segment, and sequential fragments split at markers, dashes,
legend boxes, or line crossings. On the WPD benchmark this over-segmentation
(60/100 charts) is the dominant Task-6b loss — matched-curve quality is high
(6a ~ 0.88) while the count penalty drags 6b to ~0.67.

This module merges curves using geometry (+ optional color veto):

  * duplicate/parallel merge — two curves share a substantial x-overlap and
    agree in y over that overlap (they trace the same line twice);
  * sequential join — two curves barely overlap in x, and one's endpoint
    extrapolates (slope-continuously) onto the other's start across the gap.

All tolerances are fractions of the image height (y) / width (x, gaps) so
they are resolution-independent.

Input curves are lists of [x, y] pixel points (LineFormer centerlines, one y
per x). Output has the same format, with each merged curve rebuilt as the
median y per x over the union of its members' points.
"""

import numpy as np

try:
    import cv2
except ImportError:          # color veto simply disabled without OpenCV
    cv2 = None

# Fallback image dimension when neither img nor img_shape is given.
_DEFAULT_DIM = 1000.0


def _canonical(points):
    """-> (xs sorted unique int array, ys median-per-x float array)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return np.array([], dtype=int), np.array([])
    xs = pts[:, 0].round().astype(int)
    ys = pts[:, 1]
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    ux, starts = np.unique(xs, return_index=True)
    uy = np.array([np.median(ys[s:e]) for s, e in zip(starts, list(starts[1:]) + [len(ys)])])
    return ux, uy


def _median_lab(img_lab, xs, ys):
    """Median LAB color sampled at curve points (robust to crossings/dashes)."""
    h, w = img_lab.shape[:2]
    xi = np.clip(xs, 0, w - 1)
    yi = np.clip(ys.round().astype(int), 0, h - 1)
    samples = img_lab[yi, xi].astype(float)
    return np.median(samples, axis=0)


def _tail_slope(xs, ys, k=15, from_start=False):
    """Robust slope of the first/last k points (linear fit)."""
    n = min(k, len(xs))
    if n < 2:
        return 0.0
    seg = slice(0, n) if from_start else slice(len(xs) - n, len(xs))
    sx, sy = xs[seg].astype(float), ys[seg]
    if sx.max() == sx.min():
        return 0.0
    return float(np.polyfit(sx, sy, 1)[0])


class _Curve:
    __slots__ = ("xs", "ys", "raw", "color", "mask", "score")

    def __init__(self, points, img_lab=None, mask=None, score=None):
        self.raw = [list(map(float, p)) for p in points]
        self.xs, self.ys = _canonical(points)
        self.color = _median_lab(img_lab, self.xs, self.ys) if img_lab is not None else None
        self.mask = mask
        self.score = score

    @property
    def span(self):
        return float(self.xs[-1] - self.xs[0]) if len(self.xs) else 0.0

    def y_at(self, x_grid):
        return np.interp(x_grid, self.xs, self.ys)


def nms_prune(masks, scores, novel_frac=0.3):
    """Greedy mask-NMS: keep instances by descending score while they add
    enough novel pixels; duplicates / mostly-contained fragments are dropped
    (NOT merged — the kept instance's centerline stays uncorrupted).

    Returns the kept indices in original order.
    """
    order = np.argsort(-np.asarray(scores, dtype=float))
    covered = None
    keep = []
    for i in order:
        m = np.asarray(masks[i], dtype=bool)
        area = int(m.sum())
        if area == 0:
            continue
        novel = area if covered is None else int(np.logical_and(m, ~covered).sum())
        if novel / area >= novel_frac:
            keep.append(int(i))
            covered = m.copy() if covered is None else np.logical_or(covered, m)
    return sorted(keep)


def _median_line_spacing(items, w):
    """Median vertical distance between adjacent curves, sampled across x.

    Gives a chart-level density estimate: dense multi-series charts have
    small spacing, so merge tolerances must shrink to avoid collapsing
    genuinely distinct neighbors.
    """
    diffs = []
    for x in np.linspace(0, w - 1, 25):
        ys = sorted(float(c.y_at(x)) for c in items
                    if len(c.xs) >= 2 and c.xs[0] <= x <= c.xs[-1])
        d = np.diff(ys)
        diffs.extend(d[d > 3.0])   # near-zero gaps are duplicates, not spacing
    return float(np.median(diffs)) if diffs else None


def _pair_badness(a, b, p):
    """Return merge badness in [0, 1) if (a, b) should merge, else None.

    Lower badness = stronger evidence they are the same plotted line.
    """
    if len(a.xs) < 2 or len(b.xs) < 2:
        return None
    lo = max(a.xs[0], b.xs[0])
    hi = min(a.xs[-1], b.xs[-1])
    ov = hi - lo
    min_span = max(1.0, min(a.span, b.span))

    # color veto (applies to both merge modes)
    if a.color is not None and b.color is not None:
        if np.linalg.norm(a.color - b.color) > p["color_veto_delta"]:
            return None

    if ov >= p["dup_overlap_frac"] * min_span and ov >= 3:
        # Substantial x-overlap -> duplicate-instance candidate.
        # True duplicate instances have near-identical x-extents; nested or
        # staggered extents are the signature of bundled-but-distinct curves
        # (e.g. battery cycling charts, where different cycles coincide for
        # most of their span and differ only in where they end).
        union = max(a.xs[-1], b.xs[-1]) - min(a.xs[0], b.xs[0])
        if union > 0 and ov / union < p["dup_extent_jaccard"]:
            return None
        if a.mask is not None and b.mask is not None:
            # True duplicates literally share pixels; distinct parallel
            # curves — even a few px apart — share almost none. Intersection
            # over the SMALLER mask also catches containment.
            inter = np.logical_and(a.mask, b.mask).sum()
            smaller = min(a.mask.sum(), b.mask.sum())
            ios = inter / smaller if smaller else 0.0
            if ios >= p["dup_mask_ios"]:
                return 1.0 - float(ios)
            return None
        # Fallback without masks: y agreement over (nearly) the WHOLE
        # overlap. A high percentile — not the median — separates true
        # duplicates from distinct parallel curves that converge in part
        # of the overlap.
        grid = np.linspace(lo, hi, max(8, int(ov) // 2))
        dy = float(np.percentile(np.abs(a.y_at(grid) - b.y_at(grid)), p["dup_pct"]))
        if dy <= p["dup_y_tol"]:
            return dy / p["dup_y_tol"]
        return None

    # little/no overlap -> sequential-fragment join
    left, right = (a, b) if a.xs[0] <= b.xs[0] else (b, a)
    gap = float(right.xs[0] - left.xs[-1])
    if gap > p["join_gap"]:
        return None
    slope_l = _tail_slope(left.xs, left.ys)
    slope_r = _tail_slope(right.xs, right.ys, from_start=True)
    # the two fragments must leave/enter the junction at a consistent angle,
    # otherwise we bridge distinct curves that merely end near each other
    ang = abs(np.degrees(np.arctan(slope_l) - np.arctan(slope_r)))
    if ang > p["join_max_angle_deg"]:
        return None
    if gap <= 0:
        # tiny shared region: agreement over it decides
        grid = np.linspace(lo, hi, max(4, int(ov) + 1)) if hi > lo else np.array([lo])
        err = float(np.mean(np.abs(left.y_at(grid) - right.y_at(grid))))
    else:
        # slope-continuous extrapolation across the gap, from both sides
        fwd = abs((left.ys[-1] + slope_l * gap) - right.ys[0])
        bwd = abs((right.ys[0] - slope_r * gap) - left.ys[-1])
        err = min(fwd, bwd)
        # guard against extrapolation flattery when endpoints are far apart
        if abs(left.ys[-1] - right.ys[0]) > 4 * p["join_y_tol"]:
            return None
    if err <= p["join_y_tol"]:
        return err / p["join_y_tol"]
    return None


def merge_curves(curves, img=None, img_shape=None, masks=None, scores=None,
                 dup_overlap_frac=0.25, dup_y_tol_frac=0.01, dup_pct=90,
                 dup_extent_jaccard=0.85, dup_mask_ios=0.5,
                 dup_action="merge",
                 join_gap_frac=0.06, join_y_tol_frac=0.01,
                 join_max_angle_deg=20.0,
                 color_veto_delta=30.0, min_span_frac=0.0,
                 spacing_alpha=0.0, max_iters=200):
    """Merge over-segmented pixel-space curves.

    Args:
        curves: list of curves, each a list/array of [x, y] pixel points.
        img: optional BGR image; enables the color veto.
        img_shape: (h, w) fallback when img is None (for tolerance scaling).
        masks: optional list of HxW bool instance masks aligned with curves;
            enables the (much sharper) pixel-IoU duplicate test.
        scores: optional detection confidences aligned with curves; with
            dup_action="drop_lower", duplicate pairs keep only the
            higher-confidence member instead of merging points.
        dup_mask_ios: min intersection-over-smaller-mask to merge two
            x-overlapping instances as duplicates (only when masks given).
        dup_action: "merge" combines a duplicate pair's points; "drop_lower"
            discards the lower-confidence member (needs scores) — avoids
            corrupting the kept centerline when the pair straddles two
            adjacent lines.
        dup_overlap_frac: min x-overlap (fraction of the shorter curve's span)
            for the duplicate/parallel test.
        dup_y_tol_frac: max |dy| (at the dup_pct percentile) over the
            overlap, fraction of image height, to call two overlapping
            curves the same line.
        dup_pct: percentile of |dy| used for the duplicate test; high values
            demand agreement over (nearly) the whole overlap.
        dup_extent_jaccard: min x-interval Jaccard (overlap/union) for the
            duplicate test — protects bundles of near-coincident curves that
            end at different x from being collapsed.
        join_gap_frac: max x gap (fraction of image width) to consider a
            sequential join.
        join_y_tol_frac: max endpoint/extrapolation error (fraction of image
            height) for a sequential join.
        join_max_angle_deg: max angle difference between the two fragments'
            junction slopes for a sequential join.
        color_veto_delta: LAB distance above which a merge is vetoed
            (only when img is given).
        min_span_frac: after merging, drop curves whose x-span is below this
            fraction of the widest curve's span (0 disables).
        spacing_alpha: if > 0, cap the y tolerances at spacing_alpha x the
            chart's median inter-curve spacing, so dense multi-series charts
            don't get their close-but-distinct neighbors collapsed.

    Returns:
        list of curves in the same [[x, y], ...] format.
    """
    keep = [i for i, c in enumerate(curves) if len(c) >= 2]
    if masks is not None:
        masks = [np.asarray(masks[i], dtype=bool) for i in keep]
    if scores is not None:
        scores = [float(scores[i]) for i in keep]
    curves = [curves[i] for i in keep]
    if len(curves) <= 1:
        return [list(map(list, c)) for c in curves]

    if img is not None:
        h, w = img.shape[:2]
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if cv2 is not None else None
    else:
        h, w = img_shape if img_shape is not None else (_DEFAULT_DIM, _DEFAULT_DIM)
        img_lab = None

    params = {
        "dup_overlap_frac": dup_overlap_frac,
        "dup_y_tol": dup_y_tol_frac * h,
        "dup_pct": dup_pct,
        "dup_extent_jaccard": dup_extent_jaccard,
        "dup_mask_ios": dup_mask_ios,
        "join_gap": join_gap_frac * w,
        "join_y_tol": join_y_tol_frac * h,
        "join_max_angle_deg": join_max_angle_deg,
        "color_veto_delta": color_veto_delta,
    }

    items = [_Curve(c, img_lab,
                    masks[i] if masks is not None else None,
                    scores[i] if scores is not None else None)
             for i, c in enumerate(curves)]

    if spacing_alpha > 0:
        spacing = _median_line_spacing(items, w)
        if spacing is not None:
            params["dup_y_tol"] = min(params["dup_y_tol"], spacing_alpha * spacing)
            params["join_y_tol"] = min(params["join_y_tol"], spacing_alpha * spacing)

    drop_lower = dup_action == "drop_lower" and scores is not None

    for _ in range(max_iters):
        best = None      # (badness, i, j)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                bad = _pair_badness(items[i], items[j], params)
                if bad is not None and (best is None or bad < best[0]):
                    best = (bad, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = items[i], items[j]
        lo, hi = max(a.xs[0], b.xs[0]), min(a.xs[-1], b.xs[-1])
        is_dup = (hi - lo) >= params["dup_overlap_frac"] * max(1.0, min(a.span, b.span))
        if drop_lower and is_dup:
            merged = a if (a.score or 0) >= (b.score or 0) else b
        else:
            union = (np.logical_or(a.mask, b.mask)
                     if a.mask is not None and b.mask is not None else None)
            sc = max(a.score or 0, b.score or 0) if scores is not None else None
            merged = _Curve(a.raw + b.raw, img_lab, union, sc)
        items = [c for k, c in enumerate(items) if k not in (i, j)] + [merged]

    if min_span_frac > 0 and items:
        max_span = max(c.span for c in items)
        items = [c for c in items if c.span >= min_span_frac * max_span] or items

    return [[[int(x), int(round(y))] for x, y in zip(c.xs, c.ys)] for c in items]
