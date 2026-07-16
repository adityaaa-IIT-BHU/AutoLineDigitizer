#!/usr/bin/env python3
"""
starrydata_upload.py — push AutoLineDigitizer digitizations into Starrydata
using Starrydata2's OFFICIAL internal API (Tomoya Mato, starrydata_internal_api.yaml),
reusing YOUR interactive login (no passwords in this code).

Write path (official endpoints, app mounted at /starrydata2):
  POST /paper/uploadpaper/{listname}/   {doi, projectname}   -> upsert paper, get pk
  GET  /paper/getpaperlist/{listname}/  ?projectname=        -> resolve DOI -> pk (fallback)
  POST /paper/postdata/{pk}/{project}/                       -> create/update figure+sample+points
      property_x, property_y, unit_x, unit_y, xmulti, ymulti,
      caption, fignum (upsert key), samplename, composition, comments(JSON),
      xydata("x, y\\n..."). Units are validated against SI dimension server-side.

Defaults to the STAGING server (starrydata-stg.nims.go.jp, NIMS network only) so
tests never touch production.

Usage:
  external/pw-venv/bin/python tools/starrydata_upload.py export.json            # DRY-RUN (default)
  external/pw-venv/bin/python tools/starrydata_upload.py export.json --commit   # actually write (staging)
  external/pw-venv/bin/python tools/starrydata_upload.py export.json --commit --base https://starrydata.nims.go.jp  # production
  external/pw-venv/bin/python tools/starrydata_upload.py export.json --commit --fresh   # ignore cached login, log in again

You log in ONCE: the first --commit opens a browser to log in (incl. MFA), then
caches the authenticated session to ~/.starrydata_session.json (0600). Later
uploads reuse that session silently — no browser, no re-login — until it
expires, then it prompts once more. --session <file> picks a different cache;
--fresh forces a new login. (Fully unattended, zero-login automation would need
a service account / API token without MFA — a request for Mato-san.)

DRY-RUN prints every request it WOULD send and writes nothing.

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
import os
import sys
import urllib.parse
from typing import Any, Dict, List, Optional

# Default to the STAGING server (NIMS network only) so tests never touch
# production. Mato-san's official internal API mounts the app at /starrydata2.
DEFAULT_BASE = "https://starrydata-stg.nims.go.jp"
DEFAULT_PREFIX = "/starrydata2"
DEFAULT_PROJECT = "GeneralDB"
# Where the authenticated browser session is cached so you only log in once.
DEFAULT_SESSION_FILE = os.path.expanduser("~/.starrydata_session.json")


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
                 dry_run: bool = True, prefix: str = DEFAULT_PREFIX,
                 session_file: Optional[str] = DEFAULT_SESSION_FILE):
        self.base = base.rstrip("/")
        self.prefix = "/" + prefix.strip("/")     # app mount, e.g. /starrydata2
        self.project = project
        self.dry_run = dry_run
        self.session_file = session_file          # cached login; None = don't cache
        self._pw = self._browser = self._ctx = self._page = None
        self.sent = []   # log of committed requests (or would-be, in dry-run)

    def _url(self, path: str) -> str:
        return self.base + self.prefix + path

    # -- session -------------------------------------------------------------
    def login(self, timeout_s: int = 420, force: bool = False):
        """Establish an authenticated session, reusing a cached one so a person
        only logs in ONCE (incl. MFA) rather than every upload.

        1. If a saved session exists and still works, reuse it silently — no
           browser, no login.
        2. Otherwise open the browser for an interactive login and save the
           session (cookies) to session_file (0600) for next time.

        force=True skips the cache and always logs in fresh."""
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()

        # 1) try the cached session, headlessly — no window if it still works.
        if not force and self.session_file and os.path.exists(self.session_file):
            self._browser = self._pw.chromium.launch(headless=True)
            self._ctx = self._browser.new_context(storage_state=self.session_file)
            self._page = self._ctx.new_page()
            if self._session_valid():
                print(f"  ✓ reused saved session ({self.session_file}) — "
                      f"no login needed.", flush=True)
                return True
            print("  · saved session expired — opening browser to log in again.",
                  flush=True)
            self._teardown_browser()

        # 2) interactive login in a visible window, then cache it.
        self._browser = self._pw.chromium.launch(headless=False)
        self._ctx = self._browser.new_context()
        self._page = self._ctx.new_page()
        self._page.goto(self.base + self.prefix + "/", wait_until="domcontentloaded")
        ok = self._await_login(timeout_s)
        if ok and self.session_file:
            try:
                self._ctx.storage_state(path=self.session_file)
                os.chmod(self.session_file, 0o600)
                print(f"  ✓ session saved to {self.session_file} — future uploads "
                      f"reuse it, no re-login.", flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"  · could not cache session ({ex}); you'll log in again "
                      f"next time.", flush=True)
        return ok

    # Back-compat alias: older callers expect login_interactive().
    def login_interactive(self, timeout_s: int = 420):
        return self.login(timeout_s=timeout_s)

    def _session_valid(self) -> bool:
        """Does the loaded session still authenticate? Hit the app root and
        confirm the server doesn't bounce us to the login/MFA page (an expired
        sessionid cookie survives in storage but the server rejects it)."""
        names = {c["name"] for c in self._ctx.cookies()}
        if "sessionid" not in names:
            return False
        try:
            resp = self._ctx.request.get(self.base + self.prefix + "/")
            final = (getattr(resp, "url", "") or "").lower()
            return resp.ok and "/login" not in final and "/mfa" not in final
        except Exception:  # noqa: BLE001
            return False

    def _teardown_browser(self):
        """Close the browser/context but keep the Playwright driver running."""
        for obj, meth in ((self._ctx, "close"), (self._browser, "close")):
            try:
                getattr(obj, meth)()
            except Exception:
                pass
        self._ctx = self._browser = self._page = None

    def _await_login(self, timeout_s: int = 420):
        """Poll until the user has logged in — detected by a Django sessionid
        cookie once we're off the login/MFA pages. No terminal input needed."""
        import time
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
        url = self._url(path)
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
                     "Referer": self.base + self.prefix + "/"})
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

    def _get(self, path: str, params: Dict[str, str]) -> Any:
        if self.dry_run:
            print(f"\n[DRY-RUN] GET {self._url(path)}  {params}")
            return []
        resp = self._ctx.request.get(self._url(path), params=params,
                                     headers={"X-CSRFToken": self._csrf()})
        try:
            return resp.json()
        except Exception:
            return {"status": resp.status, "text": resp.text()[:300]}

    # -- operations (Mato-san's official internal API) -----------------------
    def register_paper(self, doi: str, listname: str = "all") -> Optional[str]:
        """POST /paper/uploadpaper/{listname}/ — upsert a paper by DOI (metadata
        fetched from CrossRef; linked to the list without duplication) and
        return its ObjectID (pk). Matches the exact DOI in the returned list;
        never guesses the first row."""
        if self.dry_run:
            print(f"\n[DRY-RUN] register paper {doi} via /paper/uploadpaper/{listname}/")
            return "<paper_pk:dry-run>"
        data = self._post(f"/paper/uploadpaper/{listname}/",
                          {"doi": doi.strip(), "projectname": self.project})
        pk = self._pk_from_rows(data, doi)
        if pk:
            return pk
        # upsert response may be paginated/filtered differently — fall back to
        # the authoritative paper list.
        return self.resolve_pk(doi, listname=listname)

    def resolve_pk(self, doi: str, listname: str = "all") -> Optional[str]:
        """GET /paper/getpaperlist/{listname}/ and match the exact DOI -> pk."""
        want = doi.strip().lower()
        for page in range(1, 40):
            rows = self._get(f"/paper/getpaperlist/{listname}/",
                             {"projectname": self.project, "pagelimit": "50", "page": str(page)})
            if not isinstance(rows, list) or not rows:
                break
            pk = self._pk_from_rows(rows, doi)
            if pk:
                return pk
        print(f"  ✗ no exact DOI match for {doi} in list '{listname}'")
        return None

    @staticmethod
    def _pk_from_rows(data: Any, doi: str) -> Optional[str]:
        want = doi.strip().lower()
        for r in (data if isinstance(data, list) else []):
            f = r.get("fields", {})
            if str(f.get("DOI", "")).strip().lower() == want:
                if f.get("sid"):
                    print(f"  ✓ paper SID-{f.get('sid')} "
                          f"“{str(f.get('title', ''))[:55]}”")
                return r.get("pk")
        return None

    def upload_figure(self, pk: str, fig: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for curve in fig.get("curves") or []:
            form = build_postdata_form(fig, curve)
            out.append(self._post(f"/paper/postdata/{pk}/{self.project}/", form))
        return out

    def upload_export(self, export: Dict[str, Any],
                      pk_override: Optional[str] = None,
                      add_if_missing: bool = True,
                      listname: str = "all") -> Dict[str, Any]:
        doi = export["doi"]
        self.project = export.get("project", self.project)
        if pk_override:
            pk = pk_override
            print(f"\n▶ Using explicit paper pk {pk}")
        elif add_if_missing:
            print(f"\n▶ Registering / resolving paper for DOI {doi} …")
            pk = self.register_paper(doi, listname=listname)
        else:
            print(f"\n▶ Resolving paper for DOI {doi} …")
            pk = self.resolve_pk(doi, listname=listname)
        if not pk:
            print(f"  ✗ could not register/resolve paper for DOI {doi}.")
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
    base = args[args.index("--base") + 1] if "--base" in args else DEFAULT_BASE
    prefix = args[args.index("--prefix") + 1] if "--prefix" in args else DEFAULT_PREFIX
    pk_override = args[args.index("--pk") + 1] if "--pk" in args else None
    no_add = "--no-add" in args     # skip registration; paper must already exist
    listname = args[args.index("--list") + 1] if "--list" in args else "all"
    session_file = (args[args.index("--session") + 1] if "--session" in args
                    else DEFAULT_SESSION_FILE)
    fresh = "--fresh" in args        # ignore the cached session, log in again
    export = json.load(open(export_path, encoding="utf-8"))

    client = StarrydataClient(base=base, dry_run=not commit, prefix=prefix,
                              session_file=session_file)
    print(f"\n{'🚀 COMMIT' if commit else '🧪 DRY-RUN'} mode | {base}{client.prefix}", flush=True)
    if commit:
        if not client.login(force=fresh):
            client.close()
            return
    try:
        result = client.upload_export(export, pk_override=pk_override,
                                      add_if_missing=not no_add, listname=listname)
        print(f"\n{'=' * 50}\nResult: {json.dumps(result, ensure_ascii=False)}")
        if not commit:
            print("Dry-run only — nothing was written. Re-run with --commit to upload.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
