"""Sweep LineFormer score thresholds x merge post-processing on the WPD benchmark.

Runs inference ONCE per image (cached to disk as PNG-compressed masks +
scores), then scores every (threshold, merge) config offline in pixel space —
the same protocol as eval_summary_wpd_*_pixel.csv, so results are directly
comparable to general_v2_pixel (6a=0.881, 6b=0.666).

Usage:
  python src/sweep_lineformer.py --benchmark <WPD_file dir> \
      [--lf-model general_v2] [--limit N] [--cache-dir DIR] [--out-dir DIR]
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINEFORMER_DIR = os.path.join(SCRIPT_DIR, "submodules", "lineformer")
MMDET_DIR = os.path.join(LINEFORMER_DIR, "mmdetection")
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
for p in (SCRIPT_DIR, SRC_DIR, MMDET_DIR, LINEFORMER_DIR):
    sys.path.insert(0, p)

import cv2  # noqa: E402

from evaluate_wpd import find_projects, load_wpd_project, score  # noqa: E402
from line_merger import merge_curves  # noqa: E402

# Model checkpoints (mirrors desktop_app.LINEFORMER_MODELS, no download logic)
CHECKPOINTS = {
    "general_v2": "lineformer_general.pth",
    "baseline": "iter_3000.pth",
    "battery_finetuned": "lineformer_battery_finetuned.pth",
}

LOW_THR = 0.05  # inference-time floor; sweep thresholds re-filter offline


# ---------------------------------------------------------------------------
# Inference + cache
# ---------------------------------------------------------------------------

def load_model(lf_model):
    from mmdet.apis import init_detector
    config = os.path.join(LINEFORMER_DIR, "lineformer_swin_t_config.py")
    ckpt = os.path.join(SCRIPT_DIR, "models", CHECKPOINTS[lf_model])
    return init_detector(config, ckpt, device="cpu")


def run_inference(model, img):
    """-> (scores float array [n], masks bool array [n, h, w]) above LOW_THR."""
    from mmdet.apis import inference_detector
    result = inference_detector(model, img)
    bbox, masks = result[0][0], result[1][0]
    scores = bbox[:, 4]
    keep = scores > LOW_THR
    kept_masks = [m for m, k in zip(masks, keep) if k]
    return scores[keep], kept_masks


def cache_path(cache_dir, test_id):
    return os.path.join(cache_dir, f"{test_id}.npz")


def save_cache(path, scores, masks):
    blobs = [cv2.imencode(".png", m.astype(np.uint8) * 255)[1] for m in masks]
    np.savez_compressed(path, scores=scores,
                        n=len(blobs),
                        **{f"mask_{i}": b for i, b in enumerate(blobs)})


def load_cache(path):
    z = np.load(path)
    scores = z["scores"]
    masks = [cv2.imdecode(z[f"mask_{i}"], cv2.IMREAD_GRAYSCALE) > 127
             for i in range(int(z["n"]))]
    return scores, masks


# ---------------------------------------------------------------------------
# Curves from masks (same centerline logic as infer.get_dataseries)
# ---------------------------------------------------------------------------

def mask_to_curve(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    x_to_ys = {}
    for x, y in zip(xs, ys):
        x_to_ys.setdefault(x, []).append(y)
    return [[int(x), int(np.median(x_to_ys[x]))] for x in sorted(x_to_ys)]


def curves_to_pred(curves):
    pred_curves = []
    for i, curve in enumerate(curves):
        if len(curve) < 2:
            continue
        arr = np.array(sorted((float(p[0]), float(p[1])) for p in curve))
        pred_curves.append({"name": f"Line {i+1}", "x": arr[:, 0], "y": arr[:, 1]})
    if pred_curves:
        allx = np.concatenate([c["x"] for c in pred_curves])
        ally = np.concatenate([c["y"] for c in pred_curves])
        xr, yr = (float(allx.min()), float(allx.max())), (float(ally.min()), float(ally.max()))
    else:
        xr = yr = (0.0, 0.0)
    return {"curves": pred_curves, "x_range": xr, "y_range": yr, "calibration_ok": True}


# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------

def _merge_params(jac, **extra):
    base = dict(dup_pct=90, dup_y_tol_frac=0.010, dup_extent_jaccard=jac,
                join_gap_frac=0.06, join_y_tol_frac=0.010,
                join_max_angle_deg=20)
    base.update(extra)
    return base


# Each variant: optional mask-NMS pre-prune ("nms" = novel-pixel fraction)
# then optional centerline merge (params passed through to merge_curves).
VARIANTS = {
    "tight":     dict(merge=_merge_params(0.0)),
    "u_y04":     dict(merge=_merge_params(0.90, dup_y_tol_frac=0.004)),
    "u_y08":     dict(merge=_merge_params(0.90, dup_y_tol_frac=0.008)),
    "d_tight":   dict(merge=_merge_params(0.0, dup_action="drop_lower")),
    "d_y04":     dict(merge=_merge_params(0.90, dup_y_tol_frac=0.004,
                                          dup_action="drop_lower")),
    "d_y08":     dict(merge=_merge_params(0.90, dup_y_tol_frac=0.008,
                                          dup_action="drop_lower")),
    "d_y08_j0":  dict(merge=_merge_params(0.0, dup_y_tol_frac=0.008,
                                          dup_action="drop_lower")),
}


def build_configs():
    configs = []
    for thr in (0.3, 0.4, 0.5, 0.6):
        configs.append({"name": f"thr{thr}", "thr": thr, "variant": None})
        for vname, v in VARIANTS.items():
            configs.append({"name": f"thr{thr}_{vname}", "thr": thr,
                            "variant": v})
    return configs


def apply_config(cfg, scores, masks, img):
    idx = [i for i, s in enumerate(scores) if s >= cfg["thr"]]
    curves = [mask_to_curve(masks[i]) for i in idx]
    v = cfg["variant"]
    if v is not None and v["merge"] is not None:
        curves = merge_curves(curves, img=img,
                              scores=[scores[i] for i in idx], **v["merge"])
    else:
        curves = [c for c in curves if len(c) >= 2]
    return curves_to_pred(curves)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--lf-model", default="general_v2", choices=sorted(CHECKPOINTS))
    ap.add_argument("--limit", type=int, default=0, help="only first N projects")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    cache_dir = args.cache_dir or os.path.join(
        tempfile.gettempdir(), f"lf_sweep_cache_{args.lf_model}")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    projects = find_projects(args.benchmark)
    if args.limit:
        projects = projects[: args.limit]
    print(f"{len(projects)} projects | model={args.lf_model} | cache={cache_dir}")

    configs = build_configs()
    rows = {c["name"]: [] for c in configs}
    tmp = Path(tempfile.mkdtemp())
    model = None

    for k, proj in enumerate(projects):
        test_id = Path(proj).stem
        try:
            gt_curves, image_path = load_wpd_project(proj, extract_image_to=tmp,
                                                     space="pixel")
        except Exception as e:
            print(f"  [{test_id}] GT load failed: {e}")
            continue
        if not image_path:
            print(f"  [{test_id}] SKIP: no image")
            continue
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [{test_id}] SKIP: unreadable image")
            continue

        cpath = cache_path(cache_dir, test_id)
        if os.path.exists(cpath):
            scores, masks = load_cache(cpath)
        else:
            if model is None:
                print("loading model...")
                model = load_model(args.lf_model)
            t0 = time.time()
            scores, masks = run_inference(model, img)
            save_cache(cpath, scores, masks)
            print(f"  [{test_id}] inference {time.time()-t0:.1f}s "
                  f"({len(masks)} inst) [{k+1}/{len(projects)}]")

        for cfg in configs:
            pred = apply_config(cfg, scores, masks, img)
            rows[cfg["name"]].append(score(test_id, gt_curves, pred))

    # ---- write per-config CSVs + aggregate table ----
    cols = ["test_id", "num_curves_gt", "num_curves_pred", "task_6a", "task_6b",
            "identity_acc_pct", "mae_matched_norm_pct", "worst_gap_norm_pct",
            "x_range_err_pct", "y_range_err_pct", "calibration_ok"]
    print(f"\n{'config':22s} {'6a':>6s} {'6b':>6s} {'exact':>6s} {'over':>5s} "
          f"{'under':>5s} {'ident':>6s} {'mae':>5s}")
    summary = []
    for cfg in configs:
        rws = rows[cfg["name"]]
        if not rws:
            continue
        with open(os.path.join(args.out_dir, f"sweep_{cfg['name']}.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rws)
        t6a = np.mean([r["task_6a"] for r in rws if r["task_6a"] is not None])
        t6b = np.mean([r["task_6b"] for r in rws if r["task_6b"] is not None])
        ident = np.mean([r["identity_acc_pct"] for r in rws
                         if r["identity_acc_pct"] is not None])
        maes = [r["mae_matched_norm_pct"] for r in rws
                if r["mae_matched_norm_pct"] is not None]
        exact = sum(1 for r in rws if r["num_curves_pred"] == r["num_curves_gt"])
        over = sum(1 for r in rws if r["num_curves_pred"] > r["num_curves_gt"])
        under = sum(1 for r in rws if r["num_curves_pred"] < r["num_curves_gt"])
        print(f"{cfg['name']:22s} {t6a:6.3f} {t6b:6.3f} {exact:4d}/{len(rws):<3d} "
              f"{over:5d} {under:5d} {ident:6.1f} {np.mean(maes):5.2f}")
        summary.append({"config": cfg["name"], "task_6a": round(float(t6a), 4),
                        "task_6b": round(float(t6b), 4), "count_exact": exact,
                        "over": over, "under": under, "n": len(rws)})

    with open(os.path.join(args.out_dir, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
