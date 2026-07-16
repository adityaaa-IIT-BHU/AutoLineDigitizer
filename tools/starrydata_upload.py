#!/usr/bin/env python3
"""
starrydata_upload.py — push AutoLineDigitizer digitizations into Starrydata
through its own web endpoints, reusing YOUR interactive login (no passwords in
this code). Built from the API captured by record_starrydata.py.

Write path (one endpoint does most of the work):
  POST /starrydata2/paperlist/uploadpaper/all   {doi}          -> resolve DOI -> paper pk
  POST /starrydata2/paperlist/postdata/<pk>/<project>          -> create figure+sample+points
      property_x, property_y, unit_x, unit_y, xmulti, ymulti,
      caption, fignum, samplename, composition, comments(JSON), xydata("x, y\\n...")

Usage:
  external/pw-venv/bin/python tools/starrydata_upload.py export.json            # DRY-RUN (default)
  external/pw-venv/bin/python tools/starrydata_upload.py export.json --commit   # actually write
  external/pw-venv/bin/python tools/starrydata_upload.py export.json --base https://starrydata-stg.nims.go.jp

DRY-RUN prints every request it WOULD send and writes nothing. Always dry-run
against staging or a test paper before --commit to production.

Export JSON shape (also produced by build_export_from_kmds()):
{
  "doi": "10.1021/...",
  "project": "GeneralDB",
  "figures": [
    {"caption": "Fig 3b", "fignum": "3b",
     "x": {"property": "Temperature", "unit": "K", "multi": 0},
     "y": {"property": "Seebeck coefficient", "unit": "uV/K", "multi": 0},
     "conditions": {"Temperature": "", "Magnetic Field": "", "Orientation": "",
                    "Pressure": "", "Other": "", "comments": ""},
     "curves": [
       {"samplename": "Ta0.84Ti0.16FeSb", "composition": "Ta0.84Ti0.16FeSb",
        "points": [[300, 105.2], [310, 108.9]]}
     ]}
  ]
}
"""
import json
import sys
import urllib.parse
from typing import Any, Dict, List, Optional

DEFAULT_BASE = "https://starrydata.nims.go.jp"
DEFAULT_PROJECT = "GeneralDB"


def _xydata(points: List[List[float]]) -> str:
    """Points -> the endpoint's 'x, y\\n...' text format."""
    return "\n".join(f"{p[0]}, {p[1]}" for p in points if len(p) >= 2)


_UNIT_TEX = [
    (r"\^\{\\circ\}\s*C", "°C"), (r"\\circ", "°"),
    (r"\\Omega", "Ω"), (r"\\cdot", "·"), (r"\\times", "×"),
    (r"\\mu", "µ"), (r"\\Delta", "Δ"), (r"\\degree", "°"),
]


def sanitize_unit(unit: Optional[str]) -> str:
    """KMDS units carry LaTeX (e.g. '^{\\circ}C', 'W (m K)^{-1}') that
    Starrydata rejects. Convert to plain-text units: degree/Greek symbols,
    '^{-1}' -> '^-1', drop braces and stray backslashes."""
    import re
    u = unit or "-"
    for pat, rep in _UNIT_TEX:
        u = re.sub(pat, rep, u)
    u = re.sub(r"\^\{([^}]*)\}", r"^\1", u)   # ^{-1} -> ^-1
    u = re.sub(r"_\{([^}]*)\}", r"_\1", u)     # _{...} -> _...
    u = u.replace("{", "").replace("}", "").replace("\\", "")
    u = re.sub(r"\s+", " ", u).strip()
    return u or "-"


def build_postdata_form(fig: Dict[str, Any], curve: Dict[str, Any]) -> Dict[str, str]:
    """One (figure, curve) -> the postdata form dict."""
    x, y = fig.get("x", {}), fig.get("y", {})
    cond = fig.get("conditions") or {}
    comments = {k: str(cond.get(k, "")) for k in
                ("Temperature", "Magnetic Field", "Orientation", "Pressure", "Other", "comments")}
    return {
        "property_x": x.get("property", ""), "property_y": y.get("property", ""),
        "unit_x": sanitize_unit(x.get("unit")), "unit_y": sanitize_unit(y.get("unit")),
        "xmulti": str(x.get("multi", 0)), "ymulti": str(y.get("multi", 0)),
        "caption": fig.get("caption", ""), "fignum": fig.get("fignum", ""),
        "samplename": curve.get("samplename", ""),
        "composition": curve.get("composition", ""),
        "comments": json.dumps(comments, ensure_ascii=False),
        "xydata": _xydata(curve.get("points") or []),
    }


class StarrydataClient:
    """Thin client over an authenticated Playwright browser context. The user
    logs in interactively in the opened window; requests reuse that session's
    cookies and CSRF token. dry_run=True (default) sends nothing."""

    def __init__(self, base: str = DEFAULT_BASE, project: str = DEFAULT_PROJECT,
                 dry_run: bool = True):
        self.base = base.rstrip("/")
        self.project = project
        self.dry_run = dry_run
        self._pw = self._browser = self._ctx = self._page = None
        self.sent = []   # log of committed requests (or would-be, in dry-run)

    # -- session -------------------------------------------------------------
    def login_interactive(self, timeout_s: int = 420):
        """Open the browser and WAIT (poll) until the user has logged in —
        detected by a Django sessionid cookie once we're off the login/MFA
        pages. No terminal input needed, so this works when launched headless
        of a TTY."""
        import time
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=False)
        self._ctx = self._browser.new_context()
        self._page = self._ctx.new_page()
        self._page.goto(self.base + "/starrydata2/", wait_until="domcontentloaded")
        print("\n" + "=" * 60)
        print("  Log in to Starrydata in the browser window (incl. MFA),")
        print("  then open your database / paper list. I'll detect it and")
        print("  start uploading automatically — no need to touch the terminal.")
        print("=" * 60, flush=True)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            names = {c["name"] for c in self._ctx.cookies()}
            url = ""
            try:
                url = self._page.url
            except Exception:
                pass
            logged_in = ("sessionid" in names and "csrftoken" in names
                         and "/mfa" not in url and "/login" not in url)
            if logged_in:
                print("  ✓ login detected — waiting 3s then uploading…", flush=True)
                time.sleep(3)
                return True
            time.sleep(2)
        print("  ✗ timed out waiting for login.", flush=True)
        return False

    def _csrf(self) -> str:
        for c in self._ctx.cookies():
            if c["name"] == "csrftoken":
                return c["value"]
        return ""

    def _post(self, path: str, form: Dict[str, str]) -> Optional[Dict[str, Any]]:
        url = self.base + path
        if self.dry_run:
            print(f"\n[DRY-RUN] POST {url}")
            for k, v in form.items():
                sv = v if len(str(v)) < 90 else str(v)[:87] + "…"
                print(f"    {k} = {sv}")
            self.sent.append({"url": url, "form": form, "dry_run": True})
            return {"dry_run": True}
        resp = self._ctx.request.post(
            url, form=form,
            headers={"X-CSRFToken": self._csrf(),
                     "X-Requested-With": "XMLHttpRequest",
                     "Referer": self.base + "/starrydata2/"})
        ok = resp.ok
        try:
            data = resp.json()
        except Exception:
            data = {"status": resp.status, "text": resp.text()[:300]}
        self.sent.append({"url": url, "status": resp.status, "response": data})
        print(f"  {'✓' if ok else '✗'} POST {path} -> {resp.status}")
        if not ok:
            msg = data.get("error") or data.get("text") or data if isinstance(data, dict) else data
            print(f"      ↳ {str(msg)[:200]}")
        return data

    # -- operations ----------------------------------------------------------
    def resolve_doi(self, doi: str) -> Optional[str]:
        """DOI -> paper pk via the search endpoint. Returns the pk ONLY on an
        exact DOI match — never guesses the first row (that mis-targeted an
        upload during testing). None if the paper isn't in Starrydata yet."""
        form = {"doi": doi + "\n", "pagelimit": "25", "projectname": self.project,
                "page": "1", "words": "", "searchselect": "DOI",
                "sort_field": "sid", "sort_order": "asc"}
        if self.dry_run:
            print(f"\n[DRY-RUN] resolve DOI {doi} via /paperlist/uploadpaper/all")
            return "<paper_pk:dry-run>"
        data = self._post("/starrydata2/paperlist/uploadpaper/all", form)
        rows = data if isinstance(data, list) else []
        want = doi.strip().lower()
        for r in rows:
            if str(r.get("fields", {}).get("DOI", "")).strip().lower() == want:
                return r.get("pk")
        print(f"  ✗ no EXACT DOI match among {len(rows)} search result(s) for {doi}")
        return None

    def add_paper(self, doi: str, listname: str = "AutoLineDigitizer") -> Optional[str]:
        """Register a paper (by DOI) into Starrydata via the same endpoints the
        UI's "New Paper" uses: create a working list, then upload-to-list, which
        pulls the paper's metadata and returns it with its new SID + pk. Returns
        the pk, or None if it couldn't be added (e.g. DOI not resolvable)."""
        if self.dry_run:
            print(f"\n[DRY-RUN] add paper {doi} to list '{listname}' via "
                  f"createlist + uploadpaper/{listname}")
            return "<paper_pk:dry-run>"
        self._post("/starrydata2/paperlist/createlist/",
                   {"listname": listname, "projectname": self.project})
        form = {"doi": doi + "\n", "pagelimit": "25", "projectname": self.project,
                "page": "1", "words": "", "searchselect": "DOI",
                "sort_field": "sid", "sort_order": "asc"}
        data = self._post(f"/starrydata2/paperlist/uploadpaper/{listname}", form)
        rows = data if isinstance(data, list) else []
        want = doi.strip().lower()
        for r in rows:
            if str(r.get("fields", {}).get("DOI", "")).strip().lower() == want:
                f = r.get("fields", {})
                print(f"  ✓ paper registered: SID-{f.get('sid')} "
                      f"“{str(f.get('title',''))[:60]}”")
                return r.get("pk")
        print(f"  ✗ could not add paper for DOI {doi} "
              f"({len(rows)} row(s) returned, no exact match)")
        return None

    def upload_figure(self, pk: str, fig: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for curve in fig.get("curves") or []:
            form = build_postdata_form(fig, curve)
            out.append(self._post(f"/starrydata2/paperlist/postdata/{pk}/{self.project}", form))
        return out

    def upload_export(self, export: Dict[str, Any],
                      pk_override: Optional[str] = None,
                      add_if_missing: bool = False,
                      listname: str = "AutoLineDigitizer") -> Dict[str, Any]:
        doi = export["doi"]
        self.project = export.get("project", self.project)
        if pk_override:
            pk = pk_override
            print(f"\n▶ Using explicit paper pk {pk} (skipping DOI resolve)")
        else:
            print(f"\n▶ Resolving paper for DOI {doi} …")
            pk = self.resolve_doi(doi)
            if not pk and add_if_missing:
                print(f"  paper not in Starrydata — registering it (New Paper)…")
                pk = self.add_paper(doi, listname=listname)
        if not pk:
            print(f"  ✗ paper not found for DOI {doi}. "
                  f"{'Registration failed.' if add_if_missing else 'Re-run with --add to register it.'}")
            return {"ok": False, "error": "paper not found"}
        print(f"  paper pk = {pk}")
        n_curves = 0
        for fig in export.get("figures") or []:
            print(f"\n▶ Figure “{fig.get('caption') or fig.get('fignum')}” "
                  f"({len(fig.get('curves') or [])} curve(s))")
            self.upload_figure(pk, fig)
            n_curves += len(fig.get("curves") or [])
        return {"ok": True, "pk": pk, "n_figures": len(export.get("figures") or []),
                "n_curves": n_curves}

    def close(self):
        for obj, meth in ((self._browser, "close"), (self._pw, "stop")):
            try:
                getattr(obj, meth)()
            except Exception:
                pass


# -- adapter: AutoLineDigitizer fig_digitizations -> export JSON -------------

def build_export_from_digitizations(doi: str, fig_digitizations: Dict[Any, Dict],
                                    project: str = DEFAULT_PROJECT) -> Dict[str, Any]:
    """Convert the app's fig_digitizations dict into an upload export.
    Each detected line becomes a curve; sample name defaults to the line name."""
    figures = []
    for dig in fig_digitizations.values():
        names = dig.get("series_names") or []
        curves = []
        for i, series in enumerate(dig.get("series") or []):
            sname = names[i] if i < len(names) else f"Line {i + 1}"
            curves.append({"samplename": sname, "composition": sname,
                           "points": series})
        figures.append({
            "caption": dig.get("label", ""), "fignum": str(dig.get("label", "")),
            "x": {"property": dig.get("x_name", "X"), "unit": "-", "multi": 0},
            "y": {"property": dig.get("y_name", "Y"), "unit": "-", "multi": 0},
            "conditions": {}, "curves": curves,
        })
    return {"doi": doi, "project": project, "figures": figures}


def _placeholder_axis(term: Optional[str]) -> bool:
    return (term or "").strip().upper() in ("", "X", "Y")


def _axis_range(axis: Dict[str, Any]) -> Optional[tuple]:
    t = [x for x in (axis.get("reference ticks") or []) if isinstance(x, (int, float))]
    return (min(t), max(t)) if len(t) >= 2 else None


def _containment(dig: tuple, donor: Optional[tuple]) -> float:
    """Fraction of the digitized data range that falls inside the donor axis
    range. ~1.0 means the points live within that axis; low means they don't."""
    if donor is None or dig is None:
        return 0.0
    (a, b), (c, d) = dig, donor
    if b <= a:
        return 1.0 if c <= a <= d else 0.0
    # pad the donor range 10% each side (ticks rarely span the exact data)
    span = d - c
    c, d = c - 0.1 * span, d + 0.1 * span
    inter = max(0.0, min(b, d) - max(a, c))
    return inter / (b - a)


def _collect_donors(pub: Dict[str, Any]) -> List[Dict[str, Any]]:
    """KMDS graphs that carry real axis identity + a value range — the sources
    a digitized curve can inherit property names, units, and samples from."""
    donors = []
    for fg in pub.get("figures") or []:
        for g in fg.get("graphs") or []:
            axes = {str(a.get("axis") or "").lower(): a for a in (g.get("axes") or [])}
            xa, ya = axes.get("x", {}), axes.get("y", {})
            xt = (xa.get("quantity") or {}).get("term")
            yt = (ya.get("quantity") or {}).get("term")
            if _placeholder_axis(xt) or _placeholder_axis(yt):
                continue
            donors.append({"gid": g.get("graph local id"), "x": xa, "y": ya,
                           "xterm": xt, "yterm": yt,
                           "xr": _axis_range(xa), "yr": _axis_range(ya),
                           "samples": g.get("samples") or [], "used": False})
    return donors


def _match_donor(xr: tuple, yr: tuple, n_series: int,
                 donors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Best donor for a digitized stub, by range containment on BOTH axes.
    Requires a strong fit so raw-pixel (uncalibrated) stubs match nothing."""
    best, best_score = None, 0.0
    for d in donors:
        if d["used"] or d["xr"] is None or d["yr"] is None:
            continue
        cx, cy = _containment(xr, d["xr"]), _containment(yr, d["yr"])
        if cx < 0.6 or cy < 0.6:          # both axes must genuinely fit
            continue
        score = cx + cy + (0.5 if n_series == len(d["samples"]) else 0.0)
        if score > best_score:
            best, best_score = d, score
    return best


def build_export_from_kmds(record: Dict[str, Any],
                           project: str = DEFAULT_PROJECT) -> Dict[str, Any]:
    """Convert a merged KMDS paper record (window.__STUDIO_RECORD from
    paper_record_view.html — KMDS metadata + digitization_data) into an upload
    export. Only graphs with REAL axis property names (not placeholder X/Y) and
    digitized points are included; each series is mapped to its linked KMDS
    sample when the counts line up, else to its line label. Returns the export
    plus a 'skipped' list explaining what was left out and why."""
    pub = (record.get("metadata") or {}).get("publication") or {}
    sample_by_id = {}
    for s in pub.get("samples") or []:
        comps = [c.get("name") for c in (s.get("components") or []) if c.get("name")]
        sample_by_id[s.get("sample local id")] = {
            "name": s.get("name") or (comps[0] if comps else ""),
            "composition": s.get("name") or "; ".join(comps),
        }
    donors = _collect_donors(pub)

    figures, skipped = [], []
    seen = set()
    for fg in pub.get("figures") or []:
        for g in fg.get("graphs") or []:
            dd = g.get("digitization_data")
            if not (dd and dd.get("series")):
                continue
            gid = g.get("graph local id")
            axes = {str(a.get("axis") or "").lower(): a for a in (g.get("axes") or [])}
            xax, yax = axes.get("x", {}), axes.get("y", {})
            xterm = (xax.get("quantity") or {}).get("term") or (dd.get("x_axis") or {}).get("name")
            yterm = (yax.get("quantity") or {}).get("term") or (dd.get("y_axis") or {}).get("name")

            series = dd["series"]
            pts_all = [p for s in series for p in (s.get("points_data") or []) if len(p) >= 2]
            if not pts_all:
                continue
            xr = (min(p[0] for p in pts_all), max(p[0] for p in pts_all))
            yr = (min(p[1] for p in pts_all), max(p[1] for p in pts_all))

            # The graph's own axes are placeholders -> try to INHERIT identity
            # from a KMDS donor graph by matching the digitized value ranges.
            linked = g.get("samples") or []
            matched = None
            if _placeholder_axis(xterm) or _placeholder_axis(yterm):
                matched = _match_donor(xr, yr, len(series), donors)
                if matched is None:
                    skipped.append({"graph": gid, "reason":
                        "axes are X/Y and no KMDS graph matches the data range "
                        "(likely digitized without axis calibration — points are "
                        "in pixel space). Recalibrate axes to upload."})
                    continue
                matched["used"] = True
                xax, yax = matched["x"], matched["y"]
                xterm, yterm = matched["xterm"], matched["yterm"]
                linked = matched["samples"] or linked

            key = (xterm, yterm, round(xr[0], 3), round(yr[0], 3), len(series))
            if key in seen:
                continue          # de-duplicate repeated graphs in the merged record
            seen.add(key)

            curves = []
            for i, s in enumerate(series):
                pts = s.get("points_data") or []
                if len(pts) < 2:
                    continue
                if len(linked) == len(series) and i < len(linked) and linked[i] in sample_by_id:
                    smp = sample_by_id[linked[i]]
                    sname, comp = smp["name"], smp["composition"]
                else:
                    sname = s.get("label") or f"Line {i + 1}"
                    comp = sname
                curves.append({"samplename": sname, "composition": comp, "points": pts})
            if not curves:
                continue
            figures.append({
                "caption": (g.get("caption summary") or fg.get("figure name") or gid),
                "fignum": str(matched["gid"] if matched else fg.get("figure local id") or gid),
                "x": {"property": xterm, "unit": xax.get("unit") or "-", "multi": 0},
                "y": {"property": yterm, "unit": yax.get("unit") or "-", "multi": 0},
                "conditions": {}, "curves": curves,
                "_matched_from": (matched["gid"] if matched else None),
            })
    return {"doi": pub.get("DOI"), "project": project,
            "figures": figures, "skipped": skipped}


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    export_path = args[0]
    commit = "--commit" in args
    base = DEFAULT_BASE
    if "--base" in args:
        base = args[args.index("--base") + 1]
    pk_override = args[args.index("--pk") + 1] if "--pk" in args else None
    add_if_missing = "--add" in args
    listname = args[args.index("--list") + 1] if "--list" in args else "AutoLineDigitizer"
    export = json.load(open(export_path, encoding="utf-8"))

    client = StarrydataClient(base=base, dry_run=not commit)
    print(f"\n{'🚀 COMMIT' if commit else '🧪 DRY-RUN'} mode | base={base}", flush=True)
    if commit:
        if not client.login_interactive():
            client.close()
            return
    try:
        result = client.upload_export(export, pk_override=pk_override,
                                      add_if_missing=add_if_missing, listname=listname)
        print(f"\n{'=' * 50}\nResult: {json.dumps(result, ensure_ascii=False)}")
        if not commit:
            print("Dry-run only — nothing was written. Re-run with --commit to upload.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
