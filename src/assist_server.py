# -*- coding: utf-8 -*-
"""
assist_server.py — digitization compute for the starrydata3 review portal.

The portal's figure editor needs the heavy tools (ChartDete + smart axis
calibration, scatter-marker extraction, local-VLM axis reading / legend
labelling / curve verification) but the starrydata3 server deliberately
carries no ML dependencies. This sidecar runs in AutoLineDigitizer-land
and serves per-figure operations over plain HTTP; starrydata3 proxies
portal requests to it.

    python src/assist_server.py [--port 8390]

Endpoints (POST, JSON in/out; images as base64 PNG):
    /axes     {png_b64}                    -> axis_config, x_name, y_name
    /scatter  {png_b64}                    -> series_px (marker detection)
    /label    {png_b64, series_px}         -> names (legend -> curve, local VLM)
    /fix_axes {png_b64}                    -> x/y name+unit+log (local VLM)
    /verify   {png_b64, series_px}         -> series_px (gap-fill/stray-remove)
"""
import argparse
import base64
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

_APP = None
_VLM = None


def _get_app():
    global _APP
    if _APP is None:
        from app_settings import load_saved_api_key
        load_saved_api_key()
        from desktop_app import LineFormerApp
        _APP = LineFormerApp()
        _APP.load_chartdete_model()
    return _APP


def _get_vlm():
    global _VLM
    if _VLM is None:
        from vlm_verifier import VLMVerifier
        _VLM = VLMVerifier()
    return _VLM


def _img(body):
    raw = base64.b64decode(body["png_b64"])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode png_b64")
    return img


def _series_in(body):
    return [{"points": [[float(p[0]), float(p[1])] for p in s]}
            for s in (body.get("series_px") or [])]


def _series_out(series):
    out = []
    for s in series or []:
        pts = s.get("points", []) if isinstance(s, dict) else s
        out.append([[float(p[0]), float(p[1])] for p in pts or []
                    if len(p) >= 2])
    return out


def op_axes(body):
    app = _get_app()
    img = _img(body)
    cfg, ocr = app.detect_axis_calibration(img)
    app.axis_config, app.ocr_results = cfg, ocr
    x_name = y_name = ""
    try:
        x_name, y_name = app.get_axis_titles()
    except Exception:  # noqa: BLE001
        pass
    return {"axis_config": cfg, "x_name": x_name or "", "y_name": y_name or "",
            "calibrated": cfg is not None}


def op_scatter(body):
    app = _get_app()
    img = _img(body)
    app.detect_axis_calibration(img)     # plot area + legend/tick exclusions
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
    ext = MarkerExtractor(img, plot_area=app.cached_plot_area,
                          exclude_boxes=exclude)
    series, meta = ext.extract()
    return {"series_px": _series_out(series or []),
            "n_series": len(series or [])}


def op_label(body):
    img = _img(body)
    names = _get_vlm().label_lines_by_legend(img, _series_in(body))
    return {"names": names}


def op_fix_axes(body):
    parsed = _get_vlm().read_axis_properties(_img(body))
    return {"axes": parsed}


def op_verify(body):
    img = _img(body)
    corrected, info = _get_vlm().verify_and_correct(img, _series_in(body))
    return {"series_px": _series_out(corrected),
            "assessment": (info.get("parsed") or {}).get("overall_assessment")}


OPS = {"/axes": op_axes, "/scatter": op_scatter, "/label": op_label,
       "/fix_axes": op_fix_axes, "/verify": op_verify}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        fn = OPS.get(self.path)
        if fn is None:
            return self._send(404, {"error": f"no op {self.path}"})
        try:
            body = json.loads(
                self.rfile.read(int(self.headers["Content-Length"])))
            out = fn(body)
            self._send(200, out)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._send(200, {"ok": True, "ops": sorted(OPS)})

    def _send(self, code, obj):
        payload = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8390)
    args = p.parse_args()
    print(f"assist server on http://127.0.0.1:{args.port} "
          f"(models load on first use)")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
