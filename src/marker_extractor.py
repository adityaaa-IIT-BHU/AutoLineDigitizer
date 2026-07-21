# -*- coding: utf-8 -*-
"""
Marker-based data extraction for charts that plot data points as markers
(circles, squares, triangles, diamonds), with or without connecting lines.

Why this exists
---------------
Line-segmentation models (LineFormer) struggle to keep line identities apart
where curves cross, and colour-based refinement cannot separate two series that
share a colour. But in many battery charts the *data points* are drawn as
discrete markers, and markers carry signal those approaches throw away:

  * They are locally THICK (unlike the thin lines that may connect them), so a
    marker can be isolated from its connecting line.
  * They are the authors' actual measured points - arguably the most faithful
    thing to extract.
  * Open markers enclose a small, compact patch of background, which is a clean
    topological cue independent of any connecting line.

This module finds markers two complementary ways, groups them into series by
colour, and returns one (x, y) point-list per series - the SAME format the rest
of the pipeline already consumes, so it slots in beside ColorLineExtractor.

It degrades gracefully: on a line-only chart (no markers) it returns [], so the
caller can fall back to LineFormer.

Detection strategy
------------------
1. Foreground = non-background pixels inside the plot area. Thin gridlines,
   axes and ticks survive here but are removed in step 2 because they are thin.
2a. FILLED markers - compute a distance transform of the foreground. Marker
    centres sit far from any edge (large distance); thin lines do not. Threshold
    at a multiple of the *median* foreground distance (which approximates the
    line half-width) and the surviving compact blobs are filled-marker cores.
    An optional morphological close first fills small open markers so they are
    caught here too.
2b. HOLLOW markers - an open marker encloses a small background hole. Find
    connected components of the *background* inside the plot area and keep the
    small, compact ones that do not touch the plot border. Their centroids are
    marker centres, regardless of any connecting line.
3. Union the two sets and de-duplicate centres that coincide.
4. Sample each marker's colour from the foreground pixels around its centre.
5. Group markers into series by LAB colour distance (single-linkage).

Usage
-----
    from marker_extractor import MarkerExtractor
    ext = MarkerExtractor(img_bgr, plot_area=(x1, y1, x2, y2))
    series_points, series_colors = ext.extract()
    # series_points: list of [[x, y], ...]  - drop straight into raw_lines
"""

import numpy as np
import cv2


# Pixel mean above which a pixel is treated as near-white background
BACKGROUND_LUMINANCE = 235

# Marker cores are kept where the distance transform exceeds this multiple of
# the median foreground distance (~line half-width). Higher = stricter.
CORE_THRESH_MULT = 1.7

# LAB distance below which two markers are considered the same series
COLOR_MERGE_LAB = 16.0

# Series with fewer markers than this are dropped as noise
MIN_MARKERS_PER_SERIES = 3


class MarkerExtractor:
    def __init__(
        self,
        img_bgr,
        plot_area=None,
        background_luminance=BACKGROUND_LUMINANCE,
        core_thresh_mult=CORE_THRESH_MULT,
        color_merge_lab=COLOR_MERGE_LAB,
        min_markers_per_series=MIN_MARKERS_PER_SERIES,
        fill_hollow=True,
        sat_thresh=35,
        val_thresh=160,
        exclude_boxes=None,
    ):
        """
        img_bgr      : H x W x 3 BGR image (e.g. from cv2.imread).
        plot_area    : (x1, y1, x2, y2) bbox of the plot region, or None for the
                       whole image. Pass it to exclude the legend / axis labels.
        fill_hollow  : also try to fill small open markers so the distance-based
                       detector catches them (the hole detector catches the rest).
        exclude_boxes: [(x1, y1, x2, y2), ...] regions to blank out — in-plot
                       legend boxes and tick/label detections are the main
                       false-positive source (legend swatches ARE markers).
        """
        self.img_bgr = img_bgr
        self.img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.h, self.w = img_bgr.shape[:2]
        self.bg_lum = background_luminance
        self.core_thresh_mult = core_thresh_mult
        self.color_merge_lab = color_merge_lab
        self.min_markers = min_markers_per_series
        self.fill_hollow = fill_hollow
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh

        if plot_area is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in plot_area]
            self.plot_box = (
                max(0, x1), max(0, y1),
                min(self.w, x2), min(self.h, y2),
            )
        else:
            self.plot_box = (0, 0, self.w, self.h)
        self.exclude_boxes = [tuple(int(round(v)) for v in b[:4])
                              for b in (exclude_boxes or [])]

    # ------------------------------------------------------------------ #
    # Foreground
    # ------------------------------------------------------------------ #
    def _foreground(self):
        """Binary mask of 'ink' pixels inside the plot area.

        A pixel is ink if it is clearly coloured (high saturation) OR clearly
        dark (low value). This keeps coloured and black markers/lines while
        dropping the white background AND faint grey gridlines - the latter
        would otherwise carve the background into cells that mimic hollow
        markers.
        """
        x1, y1, x2, y2 = self.plot_box
        hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        ink = ((s > self.sat_thresh) | (v < self.val_thresh)).astype(np.uint8)
        area = np.zeros_like(ink)
        area[y1:y2, x1:x2] = 1
        for (ex1, ey1, ex2, ey2) in self.exclude_boxes:
            pad = 3
            area[max(0, ey1 - pad):min(self.h, ey2 + pad),
                 max(0, ex1 - pad):min(self.w, ex2 + pad)] = 0
        return ink * area

    @staticmethod
    def _is_compact(area, w, h, min_extent=0.30, max_aspect=2.8):
        """True for roughly equidimensional, well-filled blobs.

        Rejects thin line fragments and the slim background wedges that form
        where two lines cross (which would otherwise look like hollow markers).
        """
        if w == 0 or h == 0:
            return False
        extent = area / float(w * h)
        aspect = w / float(h)
        return extent > min_extent and (1.0 / max_aspect) < aspect < max_aspect

    # ------------------------------------------------------------------ #
    # 2a. Filled markers: locally-thick blobs (survives connecting lines)
    # ------------------------------------------------------------------ #
    def _filled_centers(self, fg):
        """Candidates from BOTH the raw foreground and a morphologically
        closed copy. The close catches open markers, but it can also fuse a
        marker with error-bar caps or a dense line into one elongated blob
        that the compactness filter then rejects — the raw pass still sees
        those markers as clean compact cores. Union both; the caller's
        centre-merge de-duplicates."""
        variants = [fg]
        if self.fill_hollow and np.any(fg > 0):
            dt0 = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
            stroke = float(np.median(dt0[fg > 0]))
            k = int(max(3, round(stroke * 3))) | 1  # odd kernel
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            variants.append(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker))

        out = []
        stroke_half_out = 1.0
        for vi, work in enumerate(variants):
            dt = cv2.distanceTransform(work, cv2.DIST_L2, 5)
            fgp = dt[work > 0]
            if fgp.size == 0:
                continue
            # Lower quartile, not median: on marker-dominated charts the
            # median distance IS the marker half-width, which made the core
            # threshold eat every marker.
            stroke_half = max(0.75, float(np.percentile(fgp, 25)))
            if vi == 0:
                stroke_half_out = stroke_half
            cores = (dt > stroke_half * self.core_thresh_mult).astype(np.uint8)

            n, _, stats, cent = cv2.connectedComponentsWithStats(cores, 8)
            for i in range(1, n):
                a = stats[i, cv2.CC_STAT_AREA]
                w_ = stats[i, cv2.CC_STAT_WIDTH]
                h_ = stats[i, cv2.CC_STAT_HEIGHT]
                if a < 2 or not self._is_compact(a, w_, h_):
                    continue
                out.append((float(cent[i][0]), float(cent[i][1]), (w_, h_)))
        return out, stroke_half_out

    # ------------------------------------------------------------------ #
    # 2b. Hollow markers: small compact enclosed background holes
    # ------------------------------------------------------------------ #
    def _hollow_centers(self, fg, stroke_half):
        x1, y1, x2, y2 = self.plot_box
        bg = np.zeros((self.h, self.w), np.uint8)
        bg[y1:y2, x1:x2] = 1
        bg = ((bg > 0) & (fg == 0)).astype(np.uint8)

        n, _, stats, cent = cv2.connectedComponentsWithStats(bg, 8)
        out = []
        max_area = (0.06 * min(x2 - x1, y2 - y1)) ** 2 * np.pi
        min_area = max(2.0, stroke_half * stroke_half * 0.5)
        for i in range(1, n):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_ = stats[i, cv2.CC_STAT_WIDTH]
            h_ = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            # The big surrounding background touches the plot border - skip it.
            if x <= x1 + 1 or y <= y1 + 1 or x + w_ >= x2 - 1 or y + h_ >= y2 - 1:
                continue
            if a < min_area or a > max_area:
                continue
            # Stricter compactness for holes (wedges at line crossings are slim).
            if not self._is_compact(a, w_, h_, min_extent=0.40, max_aspect=2.2):
                continue
            out.append((float(cent[i][0]), float(cent[i][1]), (w_, h_)))
        return out

    # ------------------------------------------------------------------ #
    # Combine + de-duplicate centres
    # ------------------------------------------------------------------ #
    def _detect_centers(self, fg):
        filled, stroke_half = self._filled_centers(fg)
        hollow = self._hollow_centers(fg, stroke_half)
        cand = ([(cx, cy, wh, False) for (cx, cy, wh) in filled]
                + [(cx, cy, wh, True) for (cx, cy, wh) in hollow])
        if not cand:
            return [], stroke_half

        # De-duplicate SIZE-AWARE: most compact candidates first, and each
        # kept candidate suppresses others only within ITS OWN radius. A
        # global merge radius fails here — elongated marker+errorbar fusions
        # from the closed pass would inflate it and eat the clean detections.
        cand.sort(key=lambda c: np.hypot(*c[2]))
        kept = []
        for (cx, cy, wh, hol) in cand:
            if all(np.hypot(cx - kx, cy - ky) > max(3.0, 0.6 * np.hypot(*kwh))
                   for (kx, ky, kwh, _) in kept):
                kept.append((cx, cy, wh, hol))
        return kept, stroke_half

    # ------------------------------------------------------------------ #
    # Marker shape: circle / triangle / rect / diamond — same-colour series
    # in different shapes are extremely common (e.g. heating vs cooling runs)
    # ------------------------------------------------------------------ #
    def _shape_of(self, cx, cy, wh, fg):
        w_, h_ = wh
        r = int(max(4, 1.3 * max(w_, h_)))
        x0, x1 = max(0, int(cx) - r), min(self.w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(self.h, int(cy) + r + 1)
        patch = fg[y0:y1, x0:x1]
        if patch.size == 0 or not np.any(patch):
            return "other"
        # close the patch so hollow outlines become solid shapes
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        solid = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return "other"
        # the component containing the centre (or the largest as fallback)
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 4 or peri <= 0:
            return "other"
        circularity = 4 * np.pi * area / (peri * peri)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        nv = len(approx)
        if circularity > 0.75 or nv >= 6:
            return "circle"
        if nv == 3:
            return "triangle"
        if nv == 4:
            (_, _), (rw, rh), ang = cv2.minAreaRect(cnt)
            a = abs(ang) % 90
            return "diamond" if 25 < a < 65 else "rect"
        return "other"

    # ------------------------------------------------------------------ #
    # Colour at a marker
    # ------------------------------------------------------------------ #
    def _color_at(self, cx, cy, fg, wh):
        w_, h_ = wh
        r = int(max(3, 0.8 * max(w_, h_)))
        x0, x1 = max(0, int(cx) - r), min(self.w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(self.h, int(cy) + r + 1)
        sub_fg = fg[y0:y1, x0:x1]
        sub = self.img_bgr[y0:y1, x0:x1]
        pix = sub[sub_fg > 0]
        if len(pix) < 3:
            return None
        return np.median(pix.astype(np.int32), axis=0).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Group markers into series by colour (single-linkage union-find)
    # ------------------------------------------------------------------ #
    def _cluster_by_color(self, markers):
        n = len(markers)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        idx = [i for i, m in enumerate(markers) if m["color"] is not None]
        labs = {}
        for i in idx:
            c = markers[i]["color"]
            labs[i] = cv2.cvtColor(
                np.array([[c]], np.uint8), cv2.COLOR_BGR2LAB
            )[0, 0].astype(np.float32)

        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j = idx[ii], idx[jj]
                if np.linalg.norm(labs[i] - labs[j]) < self.color_merge_lab:
                    union(i, j)

        groups = {}
        for i in idx:
            groups.setdefault(find(i), []).append(i)
        return groups

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def extract(self):
        """
        Returns
        -------
        series_points : list of [[x, y], ...]   one entry per detected series
        series_meta   : list of {"color": BGR uint8 array or None,
                                 "shape": str, "hollow": bool}, one per series

        Low-resolution figures (PDF panel crops) are auto-upscaled first:
        3-px markers are indistinguishable from noise at native size, and were
        the main real-world failure mode.
        """
        x1, y1, x2, y2 = self.plot_box
        m = min(x2 - x1, y2 - y1)
        if 0 < m < 640:
            scale = int(min(4, max(2, round(900.0 / m))))
            img2 = cv2.resize(self.img_bgr, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
            sub = MarkerExtractor(
                img2,
                plot_area=[v * scale for v in self.plot_box],
                background_luminance=self.bg_lum,
                core_thresh_mult=self.core_thresh_mult,
                color_merge_lab=self.color_merge_lab,
                min_markers_per_series=self.min_markers,
                fill_hollow=self.fill_hollow,
                sat_thresh=self.sat_thresh,
                val_thresh=self.val_thresh,
                exclude_boxes=[[v * scale for v in b] for b in self.exclude_boxes],
            )
            series, meta = sub._extract_native()
            series = [[[int(round(px / scale)), int(round(py / scale))]
                       for (px, py) in pts] for pts in series]
            return series, meta
        return self._extract_native()

    # ------------------------------------------------------------------ #
    # Colour layers: split the ink into per-colour masks FIRST. Inside one
    # layer marker sizes are uniform and lines/error bars of OTHER series
    # can't interfere, which makes every later threshold self-evident.
    # ------------------------------------------------------------------ #
    def _color_layers(self, fg):
        ys, xs = np.nonzero(fg)
        if len(xs) < 20:
            return []
        cols = self.img_lab[ys, xs]
        # quantized-histogram seeding: deterministic, no k to choose
        q = np.floor(cols / 22.0).astype(np.int32)
        keys, inv, counts = np.unique(q, axis=0, return_inverse=True,
                                      return_counts=True)
        min_pix = max(30, int(0.0015 * len(xs)))
        seeds = []
        for bi in np.argsort(-counts):
            if counts[bi] < min_pix:
                break
            seeds.append(np.median(cols[inv == bi], axis=0))
        # merge close seeds (biggest bins first, so they anchor the merge);
        # generous radius — anti-aliasing smears one ink colour across bins
        merge_r = max(self.color_merge_lab, 24.0)
        centers = []
        for s in seeds:
            for c in centers:
                if np.linalg.norm(s - c) < merge_r:
                    break
            else:
                centers.append(s)
        centers = centers[:8]
        if not centers:
            return []
        # NEAREST-centre assignment (Voronoi in LAB): every ink pixel belongs
        # to exactly one layer — overlapping masks would re-detect the same
        # markers once per near-duplicate colour
        C = np.stack(centers)                          # K x 3
        D = np.linalg.norm(cols[:, None, :] - C[None, :, :], axis=2)   # N x K
        best = np.argmin(D, axis=1)
        bestd = D[np.arange(len(cols)), best]
        ok = bestd < max(merge_r * 1.4, 30.0)
        layers = []
        for k, c in enumerate(centers):
            sel = ok & (best == k)
            if int(sel.sum()) < min_pix:
                continue
            mask = np.zeros((self.h, self.w), np.uint8)
            mask[ys[sel], xs[sel]] = 1
            bgr = cv2.cvtColor(np.array([[np.clip(c, 0, 255)]], np.uint8),
                               cv2.COLOR_LAB2BGR)[0, 0]
            layers.append([mask, bgr])

        # HALO ABSORPTION: anti-aliasing / JPEG artifacts put a rim of
        # intermediate colour around every marker and line. Those rim pixels
        # form their own small layer that then re-detects the same markers as
        # a phantom series. A layer whose pixels mostly sit right next to a
        # bigger layer's pixels is a halo — merge it into that layer.
        layers.sort(key=lambda le: -int(le[0].sum()))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = []
        for mask, bgr in layers:
            absorbed = False
            for big in result:
                dil = cv2.dilate(big[0], ker)
                inter = int((mask & dil).sum())
                if inter >= 0.6 * int(mask.sum()):
                    big[0] = big[0] | mask
                    absorbed = True
                    break
            if not absorbed:
                result.append([mask, bgr])
        return [(m, b) for m, b in result]

    # ------------------------------------------------------------------ #
    # Markers within ONE colour layer: peaks of the distance transform.
    # Marker interiors are the thickest structure of their own layer, so the
    # threshold is relative to the layer's own maximum — no global stroke
    # estimate to get wrong. Line-only layers yield elongated cores that the
    # compactness filter rejects wholesale.
    # ------------------------------------------------------------------ #
    def _layer_markers(self, mask):
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        mx = float(dt.max())
        if mx < 2.0:                     # nothing thicker than a hairline
            return []
        cores = (dt >= 0.62 * mx).astype(np.uint8)
        n, _, stats, cent = cv2.connectedComponentsWithStats(cores, 8)
        out = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w_ = stats[i, cv2.CC_STAT_WIDTH]
            h_ = stats[i, cv2.CC_STAT_HEIGHT]
            if a < 2 or not self._is_compact(a, w_, h_):
                continue
            out.append((float(cent[i][0]), float(cent[i][1]), (w_, h_)))
        return out

    def _extract_native(self):
        fg = self._foreground()
        series_points, series_meta, series_mask = [], [], []

        for (mask, layer_bgr) in self._color_layers(fg):
            dtv = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            vals = dtv[mask > 0]
            stroke_half = max(0.75, float(np.percentile(vals, 25))) if vals.size else 1.0

            hollow_cand = sorted(((cx, cy, wh, True) for (cx, cy, wh)
                                  in self._hollow_centers(mask, stroke_half)),
                                 key=lambda c: np.hypot(*c[2]))
            filled_cand = sorted(((cx, cy, wh, False) for (cx, cy, wh)
                                  in self._layer_markers(mask)),
                                 key=lambda c: np.hypot(*c[2]))
            # HOLLOW detections first: an open marker's ring also spawns
            # arc-fragment "filled" candidates all around it — the enclosed
            # hole is the marker, and its exclusion zone (a full ring radius +
            # stroke) swallows the fragments. Then filled candidates, each
            # suppressing only within its own compact radius.
            kept = []
            for (cx, cy, wh, hol) in hollow_cand + filled_cand:
                ok = True
                for (kx, ky, kwh, khol) in kept:
                    # hollow keeper: cover its own ring (arc fragments sit at
                    # ~half the hole diagonal + one stroke) but NOT a
                    # neighbouring marker
                    excl = (0.55 if khol else 0.6) * np.hypot(*kwh) + \
                           (2.0 * stroke_half if khol else 0.0)
                    if np.hypot(cx - kx, cy - ky) <= max(3.0, excl):
                        ok = False
                        break
                if ok:
                    kept.append((cx, cy, wh, hol))
            if len(kept) < self.min_markers:
                continue
            # second pass at the layer's MEDIAN marker scale: a noisy hole and
            # its own filled core can both survive the per-candidate radii
            # (tiny hole ⇒ tiny exclusion) and double-count the marker
            med_sz = float(np.median([np.hypot(*wh) for (_, _, wh, _) in kept]))
            r2 = max(4.0, 0.55 * med_sz)
            final = []
            for (cx, cy, wh, hol) in kept:
                if all(np.hypot(cx - fx, cy - fy) > r2 for (fx, fy, _, _) in final):
                    final.append((cx, cy, wh, hol))
            kept = final
            if len(kept) < self.min_markers:
                continue

            markers = [{"xy": (cx, cy), "shape": self._shape_of(cx, cy, wh, mask),
                        "hollow": hol} for (cx, cy, wh, hol) in kept]

            # split by (shape, hollow) only when ≥2 sub-buckets stand on their
            # own — shape reads are noisy, over-splitting is worse than none.
            # A "solid" bucket must also hold a real share of the layer, not
            # just scrape past the absolute minimum.
            buckets = {}
            for i, m in enumerate(markers):
                buckets.setdefault((m["shape"], m["hollow"]), []).append(i)
            solid_n = max(self.min_markers, int(0.25 * len(markers)))
            solid = [b for b in buckets.values() if len(b) >= solid_n]
            if len(solid) >= 2:
                leftovers = [i for b in buckets.values()
                             if len(b) < self.min_markers for i in b]
                groups = [list(b) for b in solid]
                groups[int(np.argmax([len(g) for g in groups]))] += leftovers
            else:
                groups = [list(range(len(markers)))]

            for members in groups:
                if len(members) < self.min_markers:
                    continue
                pts = sorted((markers[i]["xy"] for i in members), key=lambda p: p[0])
                shapes = [markers[i]["shape"] for i in members]
                shape = max(set(shapes), key=shapes.count)
                hollow = (sum(1 for i in members if markers[i]["hollow"])
                          > len(members) / 2)
                msize = float(np.median([np.hypot(*kept[i][2]) for i in members]))
                series_points.append([[int(x), int(y)] for (x, y) in pts])
                series_meta.append({"color": np.array(layer_bgr, np.uint8),
                                    "shape": shape, "hollow": hollow,
                                    "size": msize})
                series_mask.append(mask)

        # cross-series safety dedupe: a series most of whose points coincide
        # with a bigger series is a colour-layer echo, not real data. The
        # tolerance is MARKER-SIZED (we're in upscaled pixels here) — echo
        # detections land anywhere on the marker, not on its exact centre.
        keep_idx = []
        by_size = sorted(range(len(series_points)),
                         key=lambda k: -len(series_points[k]))
        for k in by_size:
            pts = series_points[k]
            tol_k = max(5.0, 0.7 * series_meta[k].get("size", 8.0))
            dup = False
            for j in keep_idx:
                ref = series_points[j]
                tol = max(tol_k, 0.7 * series_meta[j].get("size", 8.0))
                near = sum(1 for (px, py) in pts
                           if any(abs(px - rx) <= tol and abs(py - ry) <= tol
                                  for (rx, ry) in ref))
                if near > 0.5 * len(pts):
                    dup = True
                    break
            if not dup:
                keep_idx.append(k)
        series_points = [series_points[k] for k in keep_idx]
        series_meta = [series_meta[k] for k in keep_idx]
        series_mask = [series_mask[k] for k in keep_idx]

        # TEMPLATE AMPLIFICATION — the trick strong scatter extractors use
        # (Scatteract-style): learn each series' actual marker appearance from
        # the confident detections, then re-scan the plot with normalized
        # cross-correlation to recover what stage 1 missed — overlapping
        # markers, markers touching lines/gridlines, faint or clipped ones.
        # snap FILLED-series centres to the local distance-transform peak of
        # their own layer: JPEG/AA can pull a core centroid off the marker
        # centre, and an off-centre "known" point both corrupts the template
        # and survives the amplifier's NMS as a duplicate of the true centre.
        # The dt peak is each marker's own core summit — a neighbouring marker
        # cannot drag it the way an ink centroid gets dragged. Hollow series
        # skip this: their hole centroids are already exact.
        for i, pts in enumerate(series_points):
            if series_meta[i].get("hollow"):
                continue
            ldt = cv2.distanceTransform(series_mask[i], cv2.DIST_L2, 5)
            hw = int(max(3, 0.7 * series_meta[i].get("size", 10.0)))
            refined = []
            for (px, py) in pts:
                x0, y0 = max(0, px - hw), max(0, py - hw)
                win = ldt[y0:py + hw + 1, x0:px + hw + 1]
                if win.size and float(win.max()) > 1.0:
                    my, mx = np.unravel_index(int(np.argmax(win)), win.shape)
                    refined.append([int(x0 + mx), int(y0 + my)])
                else:
                    refined.append([px, py])
            series_points[i] = refined

        # fill support measured in the series' OWN colour layer — global ink
        # would let another series' markers or JPEG noise pass the fill test
        series_points = [self._amplify(pts, series_mask[i])
                         for i, pts in enumerate(series_points)]

        # competitive arbitration: when two series claim the same spot (JPEG
        # colour bleed lets one series' amplifier fire on another's markers),
        # the series with clearly more of its OWN ink there keeps the point
        def _support(i, px, py):
            hw = int(max(3, 0.5 * series_meta[i].get("size", 10.0)))
            win = series_mask[i][max(0, py - hw):py + hw + 1,
                                 max(0, px - hw):px + hw + 1]
            return float(win.mean()) if win.size else 0.0

        for i in range(len(series_points)):
            for j in range(len(series_points)):
                if i == j:
                    continue
                r = 0.5 * max(series_meta[i].get("size", 10.0),
                              series_meta[j].get("size", 10.0))
                pruned = []
                for (px, py) in series_points[i]:
                    rival = any(np.hypot(px - qx, py - qy) <= r
                                for (qx, qy) in series_points[j])
                    if rival:
                        si, sj = _support(i, px, py), _support(j, px, py)
                        if si < 0.6 * sj:
                            continue          # clearly the other series' marker
                    pruned.append([px, py])
                series_points[i] = pruned

        # final within-series merge at marker scale: a JPEG-shifted stage-1
        # centroid and the amplifier's true-centre find can straddle the NMS
        # boundary — cluster them and keep the average (better centres too)
        for i in range(len(series_points)):
            r = max(4.0, 0.8 * series_meta[i].get("size", 10.0))
            clusters = []
            for (px, py) in series_points[i]:
                for cl in clusters:
                    cx = np.mean([p[0] for p in cl])
                    cy = np.mean([p[1] for p in cl])
                    if np.hypot(px - cx, py - cy) <= r:
                        cl.append((px, py))
                        break
                else:
                    clusters.append([(px, py)])
            series_points[i] = sorted(
                [[int(round(np.mean([p[0] for p in cl]))),
                  int(round(np.mean([p[1] for p in cl])))] for cl in clusters],
                key=lambda p: p[0])

        # AXIS-FURNITURE filter: inward tick marks (and frame corners) form a
        # row of small "markers" hugging a plot edge. A series with ≥80% of
        # its points inside a thin border band along ONE edge is ticks, not
        # data — real flat-lying series sit at least a marker radius inside.
        x1, y1, x2, y2 = self.plot_box
        bw = max(5.0, 0.018 * (x2 - x1))
        bh = max(5.0, 0.018 * (y2 - y1))

        def _axis_series(pts):
            n = float(len(pts))
            for frac in (
                sum(1 for (px, _) in pts if px - x1 <= bw) / n,       # left
                sum(1 for (px, _) in pts if x2 - px <= bw) / n,       # right
                sum(1 for (_, py) in pts if py - y1 <= bh) / n,       # top
                sum(1 for (_, py) in pts if y2 - py <= bh) / n,       # bottom
            ):
                if frac >= 0.8:
                    return True
            return False

        keep2 = [k for k in range(len(series_points))
                 if not _axis_series(series_points[k])]
        series_points = [series_points[k] for k in keep2]
        series_meta = [series_meta[k] for k in keep2]

        # ... and individual border-band strays inside surviving series
        # (a tick that got amplified into a data series): drop points pressed
        # against an edge unless the series as a whole lives there
        for i in range(len(series_points)):
            inner = [[px, py] for (px, py) in series_points[i]
                     if not (px - x1 <= bw * 0.6 or x2 - px <= bw * 0.6
                             or py - y1 <= bh * 0.6 or y2 - py <= bh * 0.6)]
            if len(inner) >= self.min_markers:
                series_points[i] = inner

        # Deterministic order: topmost series (smallest mean y) first.
        order = sorted(
            range(len(series_points)),
            key=lambda k: np.mean([p[1] for p in series_points[k]]),
        )
        series_points = [series_points[k] for k in order]
        series_meta = [series_meta[k] for k in order]
        return series_points, series_meta

    # ------------------------------------------------------------------ #
    # Stage 2: template matching per series
    # ------------------------------------------------------------------ #
    def _amplify(self, pts, fg, corr_thresh=0.70):
        if len(pts) < 3:
            return pts
        x1, y1, x2, y2 = self.plot_box

        # template = median stack of patches around the confident centres
        spans = []
        for (px, py) in pts:
            r = 14
            xa, xb = max(0, px - r), min(self.w, px + r + 1)
            ya, yb = max(0, py - r), min(self.h, py + r + 1)
            sub = fg[ya:yb, xa:xb]
            ys, xs = np.nonzero(sub)
            if len(xs):
                spans.append(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
        size = int(np.median(spans)) if spans else 9
        size = max(5, min(31, size)) | 1
        half = size // 2

        patches = []
        for (px, py) in pts:
            if (px - half < x1 or px + half + 1 > x2
                    or py - half < y1 or py + half + 1 > y2):
                continue
            patches.append(self.img_bgr[py - half:py + half + 1,
                                        px - half:px + half + 1].astype(np.float32))
        if len(patches) < 3:
            return pts
        template = np.median(np.stack(patches), axis=0).astype(np.uint8)

        region = self.img_bgr[y1:y2, x1:x2]
        if region.shape[0] <= size or region.shape[1] <= size:
            return pts
        res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)

        # Self-calibrate the threshold: the template must at least match the
        # markers it was built from. A fixed threshold lets a noisy template
        # (e.g. marker-on-line) fire all along line segments.
        known = []
        for (px, py) in pts:
            ry, rx = py - half - y1, px - half - x1
            if 0 <= ry < res.shape[0] and 0 <= rx < res.shape[1]:
                known.append(float(res[ry, rx]))
        if known:
            corr_thresh = max(corr_thresh, float(np.percentile(known, 25)) - 0.08)

        # reference ink-fill: how much of a marker-sized window this series'
        # real markers cover. A bare line through the window covers far less —
        # the discriminator correlation alone can't provide.
        fills = []
        for (px, py) in pts:
            win = fg[max(0, py - half):py + half + 1,
                     max(0, px - half):px + half + 1]
            if win.size:
                fills.append(float(win.mean()))
        min_fill = 0.6 * float(np.median(fills)) if fills else 0.0

        # peaks above threshold, greedy NMS at ~0.9 marker size. Collect down
        # to a low floor: overlapping markers score poorly on correlation
        # (their neighbour corrupts the window) but their EXTRA ink gives them
        # away — tier-2 acceptance below.
        floor = min(corr_thresh, 0.60)
        med_fill = float(np.median(fills)) if fills else 0.0
        cand = np.argwhere(res >= floor)
        if cand.size == 0:
            return pts
        scores = res[cand[:, 0], cand[:, 1]]
        order = np.argsort(-scores)
        min_sep = max(3.0, 0.9 * size)
        accepted = [(float(px), float(py)) for (px, py) in pts]
        added = []
        for k in order:
            cy, cx = cand[k]
            gx, gy = float(cx + half + x1), float(cy + half + y1)
            # skip excluded regions
            if any(bx1 - 2 <= gx <= bx2 + 2 and by1 - 2 <= gy <= by2 + 2
                   for (bx1, by1, bx2, by2) in self.exclude_boxes):
                continue
            if any(np.hypot(gx - ax, gy - ay) < min_sep for (ax, ay) in accepted):
                continue
            gxi, gyi = int(round(gx)), int(round(gy))
            win = fg[max(0, gyi - half):gyi + half + 1,
                     max(0, gxi - half):gxi + half + 1]
            if win.size == 0:
                continue
            fill = float(win.mean())
            score = float(res[cy, cx])
            # tier 1: confident correlation + plausible ink
            # tier 2: modest correlation but MORE ink than a typical marker —
            #         the signature of overlapping markers
            if not ((score >= corr_thresh and fill >= min_fill)
                    or (score >= floor and med_fill > 0 and fill >= 1.15 * med_fill)):
                continue
            accepted.append((gx, gy))
            added.append((gx, gy))
        if not added:
            return pts
        # a good template recovers SOME missed markers; one that more than
        # doubles the series is matching chart furniture — don't trust it
        if len(added) > max(4, len(pts)):
            return pts
        out = pts + [[int(round(gx)), int(round(gy))] for (gx, gy) in added]
        out.sort(key=lambda p: p[0])
        return out