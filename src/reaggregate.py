"""
Fair re-aggregation of the WPD benchmark across models.
=======================================================

The WPD ground truth digitizes a *representative subset* of curves, not every
visible curve (e.g. 2 of ~22 overlapping battery cycles). That makes
count-sensitive metrics (Task 6b, count-exact-match) unfair: a model that
finds more curves than the human drew is penalized even when those curves are
really there.

This script re-reports the three model runs using metrics that the GT *can*
fairly measure — Task 6a (recall, normalized by N_gt) and MAE on matched
curves — and flags the charts where the metric is unreliable so they can be
reported separately:

  - cal_broken           : axis calibration blew up (OCR misread / false-log).
                           Detected as an absurd data-unit range error. These
                           corrupt even Task 6a, so we exclude them from the
                           "reliable" headline.
  - gt_undercount_suspect: every model predicts >= 2x the GT curve count,
                           i.e. the chart almost certainly has more visible
                           curves than the human digitized. Count metrics are
                           meaningless here; recall/MAE remain valid.

Reads eval_summary_wpd*.csv (no model re-runs) and prints a per-bucket,
three-model comparison.
"""

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    "baseline general": ROOT / "eval_summary_wpd_baseline.csv",
    "general_v2 (ft)":  ROOT / "eval_summary_wpd.csv",
    "battery_finetuned": ROOT / "eval_summary_wpd_battery.csv",
}

CAL_BROKEN_RANGE_ERR = 200.0   # % data-unit range error => calibration blew up


def load(p):
    return {r["test_id"]: r for r in csv.DictReader(open(p))}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    runs = {name: load(p) for name, p in RUNS.items() if Path(p).exists()}
    ids = sorted(set.intersection(*[set(d) for d in runs.values()]))
    print(f"Loaded {len(runs)} runs, {len(ids)} common charts\n")

    # ---- chart-level flags (computed from the data, reproducible) ----
    flags = {}
    for tid in ids:
        gt = int(runs["general_v2 (ft)"][tid]["num_curves_gt"])
        preds = [int(runs[m][tid]["num_curves_pred"]) for m in runs]
        # calibration broken: absurd range error or explicit calib failure in
        # ANY run (calibration is model-independent; any blow-up flags the chart)
        cal_broken = False
        for m in runs:
            r = runs[m][tid]
            ye = fnum(r["y_range_err_pct"]) or 0
            xe = fnum(r["x_range_err_pct"]) or 0
            if r["calibration_ok"] == "False" or abs(ye) > CAL_BROKEN_RANGE_ERR or abs(xe) > CAL_BROKEN_RANGE_ERR:
                cal_broken = True
        undercount = min(preds) >= 2 * gt and gt >= 1
        if cal_broken:
            bucket = "cal_broken"
        elif undercount:
            bucket = "gt_undercount_suspect"
        else:
            bucket = "reliable"
        flags[tid] = bucket

    counts = {b: sum(1 for t in ids if flags[t] == b) for b in
              ("reliable", "gt_undercount_suspect", "cal_broken")}
    print("CHART BUCKETS")
    for b, n in counts.items():
        print(f"  {b:<24} {n:>3}")
    print()

    def agg(subset_ids):
        rows = []
        for name, d in runs.items():
            t6a = [fnum(d[t]["task_6a"]) for t in subset_ids if fnum(d[t]["task_6a"]) is not None]
            t6b = [fnum(d[t]["task_6b"]) for t in subset_ids if fnum(d[t]["task_6b"]) is not None]
            mae = [fnum(d[t]["mae_matched_norm_pct"]) for t in subset_ids if fnum(d[t]["mae_matched_norm_pct"]) is not None]
            rows.append((name,
                         st.median(t6a) if t6a else float("nan"),
                         st.mean(t6a) if t6a else float("nan"),
                         st.mean(mae) if mae else float("nan"),
                         st.median(t6b) if t6b else float("nan")))
        return rows

    def show(title, subset_ids):
        print("=" * 64)
        print(f"{title}  (n={len(subset_ids)})")
        print("-" * 64)
        print(f"{'model':<20}{'6a med':>9}{'6a mean':>9}{'MAE%':>8}{'6b med':>9}")
        print(f"{'':<20}{'(FAIR headline)':>26}{'(unfair)':>17}")
        for name, a_med, a_mean, mae, b_med in agg(subset_ids):
            print(f"{name:<20}{a_med:>9.3f}{a_mean:>9.3f}{mae:>8.2f}{b_med:>9.3f}")
        print()

    reliable = [t for t in ids if flags[t] == "reliable"]
    show("RELIABLE SUBSET — fair comparison (paper headline)", reliable)
    show("ALL CHARTS (incl. unfair/broken)", ids)

    # write the per-chart bucket assignment for the paper appendix / curation
    out = ROOT / "wpd_chart_buckets.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_id", "num_curves_gt", "bucket",
                    "n_pred_baseline", "n_pred_general_v2", "n_pred_battery"])
        for tid in ids:
            w.writerow([tid, runs["general_v2 (ft)"][tid]["num_curves_gt"], flags[tid],
                        runs["baseline general"][tid]["num_curves_pred"],
                        runs["general_v2 (ft)"][tid]["num_curves_pred"],
                        runs["battery_finetuned"][tid]["num_curves_pred"]])
    print(f"Wrote per-chart buckets -> {out}")


if __name__ == "__main__":
    main()
