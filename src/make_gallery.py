"""
Visual gallery of WPD-benchmark extractions.

For every chart, renders four panels over the original image (no color
refinement, pixel space):
  1. Ground truth curves (WPD)
  2. baseline general (iter_3000)
  3. general_v2 (multi-cat finetune)
  4. battery_finetuned
plus per-chart metrics, so the extractions can be inspected by eye.

Outputs wpd_gallery/index.html (+ one PNG per chart). Cards are sorted
worst-Task6b (general_v2) first so problem charts surface at the top.

Run:
  python make_gallery.py [--benchmark /tmp/wpd_bench/WPD_file]
"""

import argparse
import html
import tempfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_wpd as E

OUT = Path("/Users/Shared/AutoLineDigitizer/wpd_gallery")
PALETTE = (list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors))


def overlay(ax, image_rgb, curves, title):
    ax.imshow(image_rgb)
    for i, c in enumerate(curves):
        ax.plot(c["x"], c["y"], color=PALETTE[i % len(PALETTE)],
                lw=1.6, alpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


MODELS = [
    ("baseline general", "general"),
    ("general_v2 (ft)", "general_v2"),
    ("battery_finetuned", "battery_finetuned"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="/tmp/wpd_bench/WPD_file")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    projects = E.find_projects(args.benchmark)
    print(f"{len(projects)} charts; loading {len(MODELS)} models (no color refinement)...")
    predictors = [(label, E.make_predictor(key, use_color_refinement=False, pixel_space=True))
                  for label, key in MODELS]
    tmp = Path(tempfile.mkdtemp())

    cards = []
    for n, proj in enumerate(projects, 1):
        tid = Path(proj).stem
        try:
            gt, img = E.load_wpd_project(proj, extract_image_to=tmp, space="pixel")
            preds = [(label, fn(img)) for label, fn in predictors]
        except Exception as e:
            print(f"  [{tid}] skipped: {e}")
            continue
        scores = [(label, E.score(tid, gt, p)) for label, p in preds]

        im = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1 + len(MODELS), figsize=(5 * (1 + len(MODELS)), 4.6))
        overlay(ax[0], im, gt, f"GT: {len(gt)} curves")
        for k, (label, p) in enumerate(preds, 1):
            r = dict(scores[k - 1][1])
            overlay(ax[k], im, p["curves"],
                    f"{label}: {len(p['curves'])}\n6a={r['task_6a']} 6b={r['task_6b']}")
        png = OUT / f"{tid}.png"
        fig.savefig(png, dpi=80, bbox_inches="tight")
        plt.close(fig)

        card = {"tid": tid, "n_gt": len(gt), "png": png.name}
        for label, r in scores:
            card[label] = r
        cards.append(card)
        summary = " | ".join(f"{label.split()[0]} {len(p['curves'])}c "
                             f"6b={scores[i][1]['task_6b']}"
                             for i, (label, p) in enumerate(preds))
        print(f"  [{n}/{len(projects)}] {tid}: GT {len(gt)} | {summary}")

    # worst general_v2 Task6b first
    def key6b(c):
        v = c.get("general_v2 (ft)", {}).get("task_6b")
        return v if v is not None else -1
    cards.sort(key=key6b)

    rows = []
    for c in cards:
        cells = f"GT={c['n_gt']}"
        for label, _ in MODELS:
            r = c.get(label, {})
            cells += (f" &nbsp;|&nbsp; <b>{html.escape(label.split()[0])}</b> "
                      f"n={r.get('num_curves_pred','?')} 6a={r.get('task_6a')} 6b={r.get('task_6b')}")
        rows.append(f"""
        <div class="card">
          <div class="hdr"><b>{html.escape(c['tid'])}</b> &nbsp; {cells}</div>
          <img src="{c['png']}" loading="lazy">
        </div>""")

    n = len(cards)
    def avg(label, k):
        vals = [c[label][k] for c in cards if c.get(label, {}).get(k) is not None]
        return round(sum(vals) / max(1, len(vals)), 3)
    means = " · ".join(f"{label.split()[0]}: 6a {avg(label,'task_6a')} / 6b {avg(label,'task_6b')}"
                       for label, _ in MODELS)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>WPD gallery — 3 models</title>
    <style>
      body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa}}
      h1{{font-size:18px}} .sub{{color:#555;font-size:13px;margin-bottom:14px}}
      .card{{background:#fff;border:1px solid #ddd;border-radius:8px;margin:10px 0;padding:8px}}
      .hdr{{font-size:13px;margin-bottom:6px}} img{{width:100%;height:auto}}
    </style></head><body>
    <h1>WPD extraction gallery — 3 models, no color refinement (pixel-space)</h1>
    <div class="sub">{n} charts · panels: <b>GT | baseline general | general_v2 | battery_finetuned</b> ·
      sorted worst general_v2 Task6b first<br>mean — {means}</div>
    {''.join(rows)}
    </body></html>"""
    (OUT / "index.html").write_text(doc)
    print(f"\nWrote {OUT/'index.html'}  ({n} charts)")


if __name__ == "__main__":
    main()
