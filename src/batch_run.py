# -*- coding: utf-8 -*-
"""
batch_run.py — unattended batch digitization + local KMDS for a folder of
PDFs, producing review-ready state for the AutoLineDigitizer editor.

    python src/batch_run.py papers/ [--model general_v2] [--no-kmds]
                                    [--limit N] [--detector mineru]

Per PDF it:
  1. extracts figures (MinerU by default — the SAME detector + ordering the
     app uses, so figure indices line up at review time),
  2. digitizes every figure headless: axis calibration (ChartDete + smart
     axis extractor), LineFormer curve tracing, axis titles, legend→curve
     names,
  3. runs the local KMDS extraction (unless --no-kmds / unavailable),
  4. saves {pdf_dir}/{stem}_batch/digitizations.json — pixel-space series +
     axis config per figure, the exact state the app needs to reopen every
     figure fully EDITABLE (points, axes, all AI tools) for curator review.

Failures are per-figure and per-paper, never per-batch; rerunning skips
PDFs whose state already exists (--force to redo).
"""
import argparse
import asyncio
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BATCH_VERSION = 1
_VLM = None


def _vlm():
    """Local-model assistant (axis names, legend labels) — None when no
    backend is configured; every use is best-effort."""
    global _VLM
    if _VLM is None:
        try:
            from vlm_verifier import VLMVerifier, backend_available
            if backend_available():
                _VLM = VLMVerifier()
        except Exception:  # noqa: BLE001
            _VLM = False
    return _VLM or None


def _scatter_series(app, img_bgr):
    """Marker/scatter detection with legend+tick exclusion boxes."""
    from marker_extractor import MarkerExtractor
    dets = getattr(app, "_last_detections", {}) or {}
    exclude = []
    for cls in ("legend_area", "legend_patch", "legend_label", "legend_title",
                "x_tick", "y_tick", "x_title", "y_title", "chart_title",
                "value_label", "mark_label"):
        for b in dets.get(cls) or []:
            try:
                exclude.append([float(v) for v in b[:4]])
            except Exception:  # noqa: BLE001
                continue
    ext = MarkerExtractor(img_bgr, plot_area=app.cached_plot_area,
                          exclude_boxes=exclude)
    series, _meta = ext.extract()
    out = []
    for s in series or []:
        pts = s.get("points", []) if isinstance(s, dict) else s
        pts = [[float(p[0]), float(p[1])] for p in pts or [] if len(p) >= 2]
        if pts:
            out.append({"points": pts})
    return out


def batch_dir(pdf_path: str) -> str:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(pdf_path)),
                        f"{base}_batch")


def digitize_figure(app, idx, img_bgr, meta) -> dict:
    """One figure, headless: calibrate → trace → name. Returns a state
    entry in the digitizations.json shape (see module docstring)."""
    from desktop_app import LineFormerApp
    app.current_image = img_bgr
    app.cached_plot_area = None
    app.axis_config = None
    app.ocr_results = None
    app.raw_lines = []
    app.data_series = []

    axis_config, ocr = app.detect_axis_calibration(img_bgr)
    app.axis_config = axis_config
    app.ocr_results = ocr

    series_px = app.extract_lines(img_bgr) or []
    x_name = y_name = ""
    try:
        x_name, y_name = app.get_axis_titles()
    except Exception:  # noqa: BLE001
        pass

    assists = []
    # sparse line tracing usually means a SCATTER chart — detect markers
    n_traced = sum(len(s.get("points", [])) for s in series_px)
    if axis_config and n_traced < 12:
        try:
            found = _scatter_series(app, img_bgr)
            if found:
                series_px = series_px + found
                assists.append(f"scatter:{len(found)}")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ scatter detection: {type(e).__name__}: {e}")

    names = [None] * len(series_px)
    try:
        import legend_mapper
        named = legend_mapper.map_curves_to_legend(
            img_bgr, series_px, app._last_detections)
        if named:
            names = [n or None for n in named]
    except Exception:  # noqa: BLE001
        pass

    # local-VLM fallbacks: axis names OCR missed, legends colors couldn't match
    v = _vlm()
    if v and axis_config and not (x_name and y_name):
        try:
            ax = v.read_axis_properties(img_bgr)
            def _nm(a):
                n = (a or {}).get("name") or ""
                u = (a or {}).get("unit") or ""
                return f"{n} ({u})" if n and u else n
            x_name = x_name or _nm(ax.get("x_axis"))
            y_name = y_name or _nm(ax.get("y_axis"))
            assists.append("vlm-axes")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ vlm axis names: {type(e).__name__}: {e}")
    if v and axis_config and series_px and not any(names):
        try:
            labeled = v.label_lines_by_legend(img_bgr, series_px)
            if labeled and any(labeled):
                names = [l or n for l, n in zip(labeled, names)]
                assists.append("vlm-labels")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ vlm labels: {type(e).__name__}: {e}")

    series_data = []
    if axis_config:
        for s in series_px:
            series_data.append(
                [list(LineFormerApp.pixel_to_data_cfg(axis_config, p[0], p[1]))
                 for p in s.get("points", [])])
    entry = {
        "label": (meta.get("caption") or "").strip() or f"Figure {idx + 1}",
        "page": meta.get("page"),
        "x_name": x_name or "", "y_name": y_name or "",
        "is_log_x": bool(axis_config and axis_config.get("xIsLogScale")),
        "is_log_y": bool(axis_config and axis_config.get("yIsLogScale")),
        "n_lines": len(series_px),
        "n_points": sum(len(s.get("points", [])) for s in series_px),
        "series_px": [[list(map(float, p)) for p in s.get("points", [])]
                      for s in series_px],
        "series": series_data,
        "series_names": [n or f"Line {i + 1}" for i, n in enumerate(names)],
        "axis_config": dict(axis_config) if axis_config else None,
        "calibrated": axis_config is not None,
        "assists": assists,
        "auto": True,
    }
    return entry


def run_pdf(app, pdf_path: str, args) -> dict:
    out_dir = batch_dir(pdf_path)
    state_path = os.path.join(out_dir, "digitizations.json")
    base0 = os.path.splitext(os.path.basename(pdf_path))[0]
    kmds_json = os.path.join(os.path.dirname(os.path.abspath(pdf_path)),
                             f"{base0}_kmds", f"{base0}.json")
    if os.path.exists(state_path) and not args.force:
        # digitization done — but if KMDS failed last time (e.g. the GPU
        # link dropped overnight), retry JUST the KMDS stage on resume
        if args.no_kmds or os.path.exists(kmds_json):
            print(f"  ↷ skip (complete): {os.path.basename(pdf_path)}")
            return {"pdf": pdf_path, "skipped": True}
        print(f"  ↻ digitization exists — retrying KMDS only")
        try:
            import kmds_parallel as kp
            os.makedirs(os.path.dirname(kmds_json), exist_ok=True)
            model = os.environ.get("KMDS_MODEL") or kp.default_model()
            prompt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "extraction_prompt.md")
            summary = asyncio.run(kp.extract_kmds_parallel(
                pdf_path, os.path.dirname(kmds_json), base_name=base0,
                prompt_path=prompt, model=model, translate=False))
            note = summary.get("_error") or "ok"
        except Exception as e:  # noqa: BLE001
            note = f"failed again: {type(e).__name__}: {e}"
        print(f"  KMDS retry: {note}")
        return {"pdf": pdf_path, "skipped": True, "kmds": note}
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    figs = app.extract_pdf_figures(pdf_path, prefer_vlm=False, refine=False,
                                   detector=args.detector)
    print(f"  figures: {len(figs)}")
    figures = {}
    n_pts = 0
    for idx, (img, meta) in enumerate(figs):
        try:
            entry = digitize_figure(app, idx, img, meta)
            figures[str(idx)] = entry
            n_pts += entry["n_points"]
            mark = "✓" if entry["calibrated"] else "◦"
            print(f"    {mark} fig {idx + 1}: {entry['n_lines']} lines, "
                  f"{entry['n_points']} pts"
                  + ("" if entry["calibrated"] else " (no axis calibration)"))
        except Exception as e:  # noqa: BLE001
            print(f"    ✗ fig {idx + 1}: {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()

    state = {"version": BATCH_VERSION, "pdf": os.path.basename(pdf_path),
             "detector": args.detector, "model_key": args.model,
             "created": time.strftime("%Y-%m-%d %H:%M:%S"),
             "n_figures": len(figs), "figures": figures}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    print(f"  ✓ digitization state -> {state_path}")

    kmds_note = "skipped"
    if not args.no_kmds:
        try:
            import kmds_parallel as kp
            model = os.environ.get("KMDS_MODEL") or kp.default_model()
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            kmds_out = os.path.join(os.path.dirname(os.path.abspath(pdf_path)),
                                    f"{base}_kmds")
            os.makedirs(kmds_out, exist_ok=True)
            prompt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "extraction_prompt.md")
            print(f"  ⤷ KMDS ({model})...")
            summary = asyncio.run(kp.extract_kmds_parallel(
                pdf_path, kmds_out, base_name=base, prompt_path=prompt,
                model=model, translate=False))
            kmds_note = (summary.get("_error")
                         or f"{summary.get('n_sections_ok')}/"
                            f"{summary.get('n_sections')} sections, "
                            f"{summary.get('n_schema_violations')} violations")
            print(f"  ✓ KMDS: {kmds_note}")
        except Exception as e:  # noqa: BLE001
            kmds_note = f"failed: {type(e).__name__}: {e}"
            print(f"  ✗ KMDS: {kmds_note}")

    if args.push:
        try:
            _push_session(pdf_path, state, figs, args)
            print(f"  ✓ pushed to review portal: {args.push}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ portal push failed: {type(e).__name__}: {e}")

    return {"pdf": pdf_path, "figures": len(figs),
            "digitized": len(figures), "points": n_pts,
            "kmds": kmds_note, "elapsed_sec": round(time.time() - t0, 1)}


def _push_session(pdf_path, state, figs, args):
    """Send this paper's record + digitized figures to the starrydata3
    review portal, so curation can start while the batch keeps running."""
    import base64
    import cv2
    import httpx
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    kmds_json = os.path.join(os.path.dirname(os.path.abspath(pdf_path)),
                             f"{base}_kmds", f"{base}.json")
    record = {}
    if os.path.exists(kmds_json):
        with open(kmds_json, encoding="utf-8") as f:
            record = json.load(f)
    figures = []
    for k, entry in (state.get("figures") or {}).items():
        idx = int(k)
        png_b64 = ""
        if idx < len(figs):
            ok, buf = cv2.imencode(".png", figs[idx][0])
            if ok:
                png_b64 = base64.b64encode(buf).decode()
        figures.append({"idx": idx, "label": entry.get("label"),
                        "page": entry.get("page"), "png_b64": png_b64,
                        "state": entry})
    key = args.push_key or os.environ.get("SD3_KEY", "")
    r = httpx.post(args.push.rstrip("/") + "/api/v1/review/sessions",
                   json={"pdf": os.path.basename(pdf_path),
                         "record": record, "figures": figures},
                   headers={"X-API-Key": key}, timeout=120)
    r.raise_for_status()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("input", help="folder of PDFs (or one PDF)")
    p.add_argument("--model", default="general_v2")
    p.add_argument("--detector", default="mineru",
                   choices=["mineru", "doclayout", "raster"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-kmds", action="store_true")
    p.add_argument("--push", default="",
                   help="starrydata3 URL — push each paper to its review portal")
    p.add_argument("--push-key", default="", help="API key for --push")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    pdfs = ([args.input] if args.input.lower().endswith(".pdf")
            else sorted(
                os.path.join(args.input, f) for f in os.listdir(args.input)
                if f.lower().endswith(".pdf")))
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        sys.exit("no PDFs found")

    from desktop_app import LineFormerApp
    from app_settings import load_saved_api_key
    load_saved_api_key()
    app = LineFormerApp()
    print(f"⤷ loading models ({args.model})...")
    app.load_lineformer_model(args.model)
    app.load_chartdete_model()

    results = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {os.path.basename(pdf)}")
        try:
            results.append(run_pdf(app, pdf, args))
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ paper failed: {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
            results.append({"pdf": pdf, "error": str(e)})

    done = [r for r in results if r.get("digitized")]
    print(f"\n✅ batch done: {len(done)}/{len(pdfs)} papers digitized, "
          f"{sum(r.get('points', 0) for r in done)} points; review each "
          f"paper in AutoLineDigitizer (state auto-loads on PDF open).")


if __name__ == "__main__":
    main()
