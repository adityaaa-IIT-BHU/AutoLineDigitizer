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


def build_postdata_form(fig: Dict[str, Any], curve: Dict[str, Any]) -> Dict[str, str]:
    """One (figure, curve) -> the postdata form dict."""
    x, y = fig.get("x", {}), fig.get("y", {})
    cond = fig.get("conditions") or {}
    comments = {k: str(cond.get(k, "")) for k in
                ("Temperature", "Magnetic Field", "Orientation", "Pressure", "Other", "comments")}
    return {
        "property_x": x.get("property", ""), "property_y": y.get("property", ""),
        "unit_x": x.get("unit", "-") or "-", "unit_y": y.get("unit", "-") or "-",
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
    def login_interactive(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=False)
        self._ctx = self._browser.new_context()
        self._page = self._ctx.new_page()
        self._page.goto(self.base + "/starrydata2/", wait_until="domcontentloaded")
        print("\n" + "=" * 60)
        print("  Log in to Starrydata in the browser window (incl. MFA).")
        print("  When you SEE the paper list / database, press Enter here.")
        print("=" * 60)
        input("  ↵ Enter once logged in… ")

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
        return data

    # -- operations ----------------------------------------------------------
    def resolve_doi(self, doi: str) -> Optional[str]:
        """DOI -> paper pk (via the search endpoint). None if not found."""
        form = {"doi": doi + "\n", "pagelimit": "25", "projectname": self.project,
                "page": "1", "words": "", "searchselect": "DOI",
                "sort_field": "sid", "sort_order": "asc"}
        if self.dry_run:
            print(f"\n[DRY-RUN] resolve DOI {doi} via /paperlist/uploadpaper/all")
            return "<paper_pk:dry-run>"
        data = self._post("/starrydata2/paperlist/uploadpaper/all", form)
        rows = data if isinstance(data, list) else []
        for r in rows:
            f = r.get("fields", {})
            if str(f.get("DOI", "")).lower() == doi.lower():
                return r.get("pk")
        return rows[0].get("pk") if rows else None

    def upload_figure(self, pk: str, fig: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for curve in fig.get("curves") or []:
            form = build_postdata_form(fig, curve)
            out.append(self._post(f"/starrydata2/paperlist/postdata/{pk}/{self.project}", form))
        return out

    def upload_export(self, export: Dict[str, Any]) -> Dict[str, Any]:
        doi = export["doi"]
        self.project = export.get("project", self.project)
        print(f"\n▶ Resolving paper for DOI {doi} …")
        pk = self.resolve_doi(doi)
        if not pk:
            print(f"  ✗ paper not found for DOI {doi}. Add it to Starrydata first.")
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
    export = json.load(open(export_path, encoding="utf-8"))

    client = StarrydataClient(base=base, dry_run=not commit)
    print(f"\n{'🚀 COMMIT' if commit else '🧪 DRY-RUN'} mode | base={base}")
    if commit:
        client.login_interactive()
    try:
        result = client.upload_export(export)
        print(f"\n{'=' * 50}\nResult: {json.dumps(result, ensure_ascii=False)}")
        if not commit:
            print("Dry-run only — nothing was written. Re-run with --commit to upload.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
