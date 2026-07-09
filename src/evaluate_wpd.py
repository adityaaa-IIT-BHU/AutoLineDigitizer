"""
Batch evaluation over the WPD hand-annotated benchmark.
=======================================================

This drives the SAME scoring math as evaluate.py (CHART-Info Task 6a/6b,
Hungarian curve matching, normalized MAE, axis-range error) over the
WebPlotDigitizer (WPD) ground-truth benchmark in demo/WPD_file.zip.

Two things differ from evaluate.py:
  - GROUND TRUTH comes from WPD project tars instead of a Starrydata
    manifest. Each WPD curve point already carries a `value: [x, y]` field
    in *data units* (WPD computed it from the human's axis calibration),
    so GT needs no recalibration.
  - PREDICTIONS come from running the live pipeline (LineFormer + ChartDete
    + pixel_to_data) on each chart's image.png, instead of reading a
    StarryDigitizer pred.zip.

The scoring functions are imported from evaluate.py so the metrics remain
identical and directly comparable to the LineFormer paper.

USAGE
-----
  # 1. Verify the GT loader against real data WITHOUT loading any models:
  python evaluate_wpd.py --gt-only --benchmark /path/to/WPD_file

  # 2. Full run (needs LineFormer + ChartDete models installed):
  python evaluate_wpd.py --benchmark /path/to/WPD_file --lf-model general_v2

A WPD "benchmark dir" is a tree of *.tar files (one per figure), each tar
containing wpd.json + info.json + image.png. Already-extracted folders
(with a bare wpd.json + image.png) are also accepted.

OUTPUTS
-------
  eval_summary_wpd.csv   one row per figure (same columns as eval_summary.csv,
                         plus calibration_ok and a source path)
  prints aggregate stats (mean/median Task 6b, MAE_norm, % charts under
  threshold) and a breakdown by GT curve count.
"""

import argparse
import csv
import io
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# Reuse the exact scoring primitives from the existing evaluator so the
# numbers stay comparable to evaluate.py / the LineFormer paper.
from evaluate import pairwise_similarity, curve_mae, worst_gap

MATCH_THRESHOLD_FRAC = 0.20   # MAE < 20% of GT Y range => an identity "match"


# ===========================================================================
# WPD ground-truth loader  (no models needed — independently testable)
# ===========================================================================

def _read_wpd_json(members):
    """Given {name: bytes}, return the parsed wpd.json dict and image bytes."""
    wpd, image = None, None
    for name, data in members.items():
        base = name.split("/")[-1]
        if base == "wpd.json":
            wpd = json.loads(data.decode("utf-8"))
        elif base.lower().endswith((".png", ".jpg", ".jpeg")) and image is None:
            image = data
    return wpd, image


def _curves_from_wpd(wpd, space="data"):
    """Extract GT curves from a parsed wpd.json, skipping empty datasets
    (e.g. WPD's 'Default Dataset' with 0 points).

    space="data"  -> use the calibrated `value: [x, y]` field (data units)
    space="pixel" -> use the raw `x`, `y` pixel coordinates (image space),
                     for a calibration-free, pure-tracing evaluation.
    Returns list of {name, x: np.array, y: np.array}."""
    coll = wpd.get("datasetColl") or wpd.get("wpd", {}).get("datasetColl", [])
    curves = []
    for ds in coll:
        pts = ds.get("data", [])
        if not pts:
            continue
        xy = []
        for p in pts:
            if space == "pixel":
                if p.get("x") is not None and p.get("y") is not None:
                    xy.append((float(p["x"]), float(p["y"])))
            else:
                v = p.get("value")
                if v and len(v) >= 2:
                    xy.append((float(v[0]), float(v[1])))
        if len(xy) < 2:
            continue
        xy.sort(key=lambda t: t[0])
        arr = np.array(xy, dtype=float)
        curves.append({"name": ds.get("name", "?"),
                       "x": arr[:, 0], "y": arr[:, 1]})
    return curves


def load_wpd_project(tar_or_dir, extract_image_to=None, space="data"):
    """Load one WPD project from a .tar or an extracted dir.

    Returns (gt_curves, image_path). image_path is written under
    extract_image_to (a temp dir) when reading from a tar.
    """
    p = Path(tar_or_dir)
    if p.is_dir():
        members = {}
        for f in p.iterdir():
            if f.is_file():
                members[f.name] = f.read_bytes()
    else:
        members = {}
        with tarfile.open(p) as tf:
            for m in tf.getmembers():
                if m.isfile() and "__MACOSX" not in m.name:
                    members[m.name] = tf.extractfile(m).read()

    wpd, image = _read_wpd_json(members)
    if wpd is None:
        raise ValueError(f"No wpd.json in {p}")
    gt_curves = _curves_from_wpd(wpd, space=space)

    image_path = None
    if image is not None:
        out_dir = Path(extract_image_to or tempfile.mkdtemp())
        image_path = out_dir / f"{p.stem}_image.png"
        image_path.write_bytes(image)
    return gt_curves, (str(image_path) if image_path else None)


def find_projects(benchmark_dir):
    """Yield every WPD project path under benchmark_dir: each .tar, plus any
    already-extracted folder containing a wpd.json."""
    root = Path(benchmark_dir)
    out, ids = [], set()
    # .tar projects first (preferred), then extracted dirs; dedup by stem so a
    # project present as both a tar and an unpacked folder is counted once.
    candidates = [t for t in sorted(root.rglob("*.tar")) if "__MACOSX" not in str(t)]
    candidates += [w.parent for w in sorted(root.rglob("wpd.json")) if "__MACOSX" not in str(w)]
    for c in candidates:
        key = c.stem.replace("_", "").lower()
        if key in ids:
            continue
        ids.add(key)
        out.append(c)
    return out


# ===========================================================================
# Live-pipeline prediction  (needs LineFormer + ChartDete models)
# ===========================================================================

def make_predictor(lf_model="general_v2", pixel_space=False):
    """Build a predict(image_path) -> pred dict closure. Loads models once.

    pixel_space=True compares raw LineFormer curve pixels against GT pixels
    (no axis calibration) — isolates tracing quality, comparable to the
    pixel-level numbers in the training section / LineFormer paper.
    """
    import cv2
    from desktop_app import LineFormerApp

    app = LineFormerApp()
    app.load_chartdete_model()
    app.load_lineformer_model(lf_model)

    def predict(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"unreadable image: {image_path}")

        if pixel_space:
            # color refinement still wants a plot area; detect it cheaply but
            # do NOT calibrate to data units.
            calibration_ok = True
        else:
            cfg, ocr = app.detect_axis_calibration(img)
            app.axis_config = cfg        # pixel_to_data reads self.axis_config
            app.ocr_results = ocr
            calibration_ok = cfg is not None

        app.extract_lines(img)           # fills app.raw_lines (pixel curves)
        pred_curves = []
        for i, curve in enumerate(app.raw_lines or []):
            if len(curve) < 2:
                continue
            if pixel_space:
                xy = [(float(pt[0]), float(pt[1])) for pt in curve]
            else:
                xy = [app.pixel_to_data(pt[0], pt[1]) for pt in curve]
            xy.sort(key=lambda t: t[0])
            arr = np.array(xy, dtype=float)
            pred_curves.append({"name": f"Line {i+1}",
                                "x": arr[:, 0], "y": arr[:, 1]})

        # Predicted data-span (for the axis-range diagnostic). We use the
        # extent of predicted curve values; comment in evaluate.py uses the
        # digitizer's axis calibration values — equivalent in intent.
        if pred_curves:
            allx = np.concatenate([c["x"] for c in pred_curves])
            ally = np.concatenate([c["y"] for c in pred_curves])
            x_range = (float(allx.min()), float(allx.max()))
            y_range = (float(ally.min()), float(ally.max()))
        else:
            x_range = y_range = (0.0, 0.0)
        return {"curves": pred_curves, "x_range": x_range,
                "y_range": y_range, "calibration_ok": calibration_ok}

    return predict


# ===========================================================================
# Scoring  (same math as evaluate.py:evaluate_one, refactored to be reusable)
# ===========================================================================

def score(test_id, gt_curves, pred):
    n_gt = len(gt_curves)
    n_pred = len(pred["curves"])

    all_x = np.concatenate([c["x"] for c in gt_curves]) if n_gt else np.array([0.0])
    all_y = np.concatenate([c["y"] for c in gt_curves]) if n_gt else np.array([0.0])
    gt_xr = (float(all_x.min()), float(all_x.max()))
    gt_yr = (float(all_y.min()), float(all_y.max()))
    gt_xs = gt_xr[1] - gt_xr[0]
    gt_ys = gt_yr[1] - gt_yr[0]

    pred_xs = pred["x_range"][1] - pred["x_range"][0]
    pred_ys = pred["y_range"][1] - pred["y_range"][0]
    x_range_err = abs(pred_xs - gt_xs) / gt_xs * 100 if gt_xs else float("nan")
    y_range_err = abs(pred_ys - gt_ys) / gt_ys * 100 if gt_ys else float("nan")

    # ---- Official CHART-Info Task 6a/6b via Hungarian on similarity ----
    sim = np.zeros((n_gt, n_pred))
    cost = np.full((n_gt, n_pred), 1e9)
    for i, g in enumerate(gt_curves):
        for j, pr in enumerate(pred["curves"]):
            sim[i, j] = pairwise_similarity(pr["x"], pr["y"], g["x"], g["y"], gt_ys)
            m = curve_mae(pr["x"], pr["y"], g["x"], g["y"])
            cost[i, j] = m if np.isfinite(m) else 1e9

    if n_gt == 0 or n_pred == 0:
        matched_sim = 0.0
    else:
        r, c = linear_sum_assignment(-sim)
        matched_sim = float(sim[r, c].sum())
    task_6a = matched_sim / n_gt if n_gt else float("nan")
    task_6b = matched_sim / (max(n_gt, n_pred) or 1)

    # ---- Identity accuracy + matched MAE (diagnostic) ----
    if n_gt and n_pred:
        mr, mc = linear_sum_assignment(cost)
    else:
        mr, mc = np.array([], int), np.array([], int)
    thresh = MATCH_THRESHOLD_FRAC * gt_ys
    matched_maes, worsts, n_matched = [], [], 0
    for i, j in zip(mr, mc):
        if cost[i, j] < thresh:
            n_matched += 1
            matched_maes.append(cost[i, j])
            worsts.append(worst_gap(pred["curves"][j]["x"], pred["curves"][j]["y"],
                                    gt_curves[i]["x"], gt_curves[i]["y"]))
    identity_acc = n_matched / n_gt * 100 if n_gt else float("nan")
    avg_mae = float(np.mean(matched_maes)) if matched_maes else float("nan")
    avg_mae_norm = (avg_mae / gt_ys * 100) if (matched_maes and gt_ys > 0) else float("nan")
    worst_norm = (float(np.nanmax(worsts)) / gt_ys * 100) if (worsts and gt_ys > 0) else float("nan")

    def rnd(v, n=2):
        return round(v, n) if isinstance(v, float) and not np.isnan(v) else None

    return {
        "test_id": test_id,
        "num_curves_gt": n_gt,
        "num_curves_pred": n_pred,
        "task_6a": rnd(task_6a, 4),
        "task_6b": rnd(task_6b, 4),
        "identity_acc_pct": rnd(identity_acc, 1),
        "mae_matched_norm_pct": rnd(avg_mae_norm),
        "worst_gap_norm_pct": rnd(worst_norm),
        "x_range_err_pct": rnd(x_range_err),
        "y_range_err_pct": rnd(y_range_err),
        "calibration_ok": pred.get("calibration_ok"),
    }


# ===========================================================================
# Driver
# ===========================================================================

def aggregate(rows):
    def col(name):
        return [r[name] for r in rows if isinstance(r.get(name), (int, float))]
    lines = ["", "AGGREGATE (n=%d figures)" % len(rows), "-" * 50]
    for metric in ("task_6a", "task_6b", "mae_matched_norm_pct", "identity_acc_pct"):
        vals = col(metric)
        if vals:
            lines.append(f"  {metric:<22} mean={np.mean(vals):7.3f}  median={np.median(vals):7.3f}")
    t6b = col("task_6b")
    if t6b:
        good = sum(1 for v in t6b if v >= 0.5)
        lines.append(f"  charts with Task6b>=0.5 : {good}/{len(t6b)}")
    # breakdown by GT curve count
    lines.append("  by GT curve count:")
    by_n = {}
    for r in rows:
        by_n.setdefault(r["num_curves_gt"], []).append(r)
    for n in sorted(by_n):
        sub = [r["task_6b"] for r in by_n[n] if isinstance(r.get("task_6b"), (int, float))]
        if sub:
            lines.append(f"    {n} curves (n={len(by_n[n]):3d}): mean Task6b={np.mean(sub):.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="Dir of WPD .tar projects (e.g. extracted WPD_file/)")
    ap.add_argument("--lf-model", default="general_v2")
    ap.add_argument("--gt-only", action="store_true",
                    help="Only parse + report ground truth; load no models")
    ap.add_argument("--out", default="eval_summary_wpd.csv")
    ap.add_argument("--pixel-space", action="store_true",
                    help="Score raw LineFormer pixels vs GT pixels (no calibration) "
                         "— isolates tracing quality, comparable to the LineFormer paper.")
    args = ap.parse_args()

    space = "pixel" if args.pixel_space else "data"
    projects = find_projects(args.benchmark)
    print(f"Found {len(projects)} WPD projects under {args.benchmark} "
          f"[{'PIXEL-space (tracing only)' if args.pixel_space else 'DATA-units (end-to-end)'}]")

    tmp = Path(tempfile.mkdtemp())
    predict = None if args.gt_only else make_predictor(
        args.lf_model, pixel_space=args.pixel_space)

    rows = []
    for proj in projects:
        test_id = Path(proj).stem
        try:
            gt_curves, image_path = load_wpd_project(proj, extract_image_to=tmp, space=space)
        except Exception as e:
            print(f"  [{test_id}] GT load failed: {e}")
            continue

        if args.gt_only:
            npts = sum(len(c["x"]) for c in gt_curves)
            print(f"  [{test_id}] {len(gt_curves)} GT curves, {npts} pts, "
                  f"image={'yes' if image_path else 'MISSING'}")
            continue

        if not image_path:
            print(f"  [{test_id}] SKIP: no image in project")
            continue
        try:
            pred = predict(image_path)
        except Exception as e:
            print(f"  [{test_id}] prediction failed: {e}")
            continue
        row = score(test_id, gt_curves, pred)
        rows.append(row)
        print(f"  [{test_id}] Task6b={row['task_6b']} "
              f"MAE_norm={row['mae_matched_norm_pct']}% "
              f"({row['num_curves_pred']}/{row['num_curves_gt']} curves, "
              f"calib={'ok' if row['calibration_ok'] else 'FAIL'})")

    if rows:
        cols = ["test_id", "num_curves_gt", "num_curves_pred", "task_6a", "task_6b",
                "identity_acc_pct", "mae_matched_norm_pct", "worst_gap_norm_pct",
                "x_range_err_pct", "y_range_err_pct", "calibration_ok"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")
        print(aggregate(rows))


if __name__ == "__main__":
    main()
