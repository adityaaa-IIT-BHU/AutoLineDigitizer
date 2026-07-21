#!/usr/bin/env python3
"""
starrydata_fetch.py — pull an ENTIRE project's data out of Starrydata2 through
the internal API (no direct MongoDB access needed).

  python tools/starrydata_fetch.py OrganicThermoelectricMaterials
  python tools/starrydata_fetch.py OrganicThermoelectricMaterials --out ./otm_dump
  python tools/starrydata_fetch.py GeneralDB --base https://starrydata-stg.nims.go.jp

Auth: your api-token from ~/.sd2_token / $SD2_TOKEN / --token (Mato's token
functionality; NIMS network only — the NIMS proxy is used automatically).

Output, under --out (default ./starrydata_fetch_<project>/):
  papers.json               list of {pk, sid, doi, title}
  paper_<sid>_<pk>.json     full data_api JSON per paper (figures, samples, curves)
  <project>_curves.csv      every curve point, long form
  <project>_full.json       everything in one file

NOTE: this staging deployment matches routes WITHOUT trailing slashes.
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from starrydata_upload import (DEFAULT_BASE, DEFAULT_PREFIX, NIMS_PROXIES,  # noqa: E402
                               load_token)

import requests  # noqa: E402


class Fetcher:
    def __init__(self, token, base=DEFAULT_BASE, prefix=DEFAULT_PREFIX):
        self.base, self.prefix = base.rstrip("/"), prefix
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Token {token}",
                               "X-Requested-With": "XMLHttpRequest"})
        self._proxy = None

    def get(self, path, params=None):
        url = f"{self.base}{self.prefix}{path}"
        kw = {"params": params or {}, "timeout": 120}
        if self._proxy:
            kw["proxies"] = {"http": self._proxy, "https": self._proxy}
        try:
            r = self.s.get(url, **kw)
            if r.status_code == 403 and not self._proxy:
                raise requests.exceptions.ConnectionError("403 -> proxy")
            return r
        except requests.exceptions.RequestException:
            if self._proxy:
                raise
            last = None
            for cand in NIMS_PROXIES:
                kw["proxies"] = {"http": cand, "https": cand}
                try:
                    r = self.s.get(url, **kw)
                    self._proxy = cand
                    return r
                except requests.exceptions.RequestException as ex:  # noqa: PERF203
                    last = ex
            raise last

    def papers(self, project):
        """All papers of the project via paginated getpaperlist (token auth)."""
        out, page = [], 1
        while True:
            r = self.get("/paper/getpaperlist/all",
                         {"projectname": project, "pagelimit": "100", "page": str(page)})
            if r.status_code != 200:
                raise SystemExit(f"getpaperlist page {page}: HTTP {r.status_code} — "
                                 f"{r.text[:200]}")
            rows = r.json()
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                f = row.get("fields", {})
                out.append({"pk": row.get("pk"), "sid": f.get("sid"),
                            "doi": f.get("DOI"), "title": f.get("title")})
            print(f"  page {page}: +{len(rows)} (total {len(out)})")
            if len(rows) < 100:
                break
            page += 1
        return out

    def paper_data(self, pk, project):
        r = self.get(f"/paper/data_api/{pk}/{project}")
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code}", "_body": r.text[:300]}
        try:
            return r.json()
        except Exception:  # noqa: BLE001
            return {"_error": "not json", "_body": r.text[:300]}


def _walk_curves(obj, path=()):
    """Best-effort: find curve point arrays in a data_api payload without
    assuming its exact shape — yields (path, x_label, y_label, points)."""
    if isinstance(obj, dict):
        data = obj.get("data") or obj.get("points") or obj.get("xy_data")
        if isinstance(data, list) and data and isinstance(data[0], (list, dict)):
            pts = []
            for p in data:
                if isinstance(p, dict) and "x" in p and "y" in p:
                    pts.append((p["x"], p["y"]))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((p[0], p[1]))
            if pts:
                yield (path, obj.get("prop_x") or obj.get("x_label") or "",
                       obj.get("prop_y") or obj.get("y_label") or "", obj, pts)
        for k, v in obj.items():
            yield from _walk_curves(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_curves(v, path + (str(i),))


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__)
        return
    project = args[0]

    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    token = load_token(opt("--token"))
    if not token:
        raise SystemExit("No api-token: save it to ~/.sd2_token (chmod 600), "
                         "set $SD2_TOKEN, or pass --token <key>.")
    base = opt("--base", DEFAULT_BASE)
    out_dir = opt("--out", f"./starrydata_fetch_{project}")
    os.makedirs(out_dir, exist_ok=True)

    f = Fetcher(token, base=base)
    print(f"▶ listing papers of {project!r} on {base} …")
    papers = f.papers(project)
    print(f"  {len(papers)} paper(s)")
    json.dump(papers, open(os.path.join(out_dir, "papers.json"), "w"),
              ensure_ascii=False, indent=1)
    if not papers:
        raise SystemExit(f"project {project!r} has no papers here (wrong name, "
                         "or this DB doesn't carry it — try the other base URL).")

    full, n_err = {}, 0
    for i, p in enumerate(papers, 1):
        sid = p.get("sid") or "x"
        print(f"▶ [{i}/{len(papers)}] SID-{sid} {str(p.get('title'))[:50]}")
        data = f.paper_data(p["pk"], project)
        if "_error" in data:
            n_err += 1
            print(f"    ✗ {data['_error']}")
        safe = re.sub(r"[^A-Za-z0-9_-]", "", str(p["pk"]))
        json.dump(data, open(os.path.join(out_dir, f"paper_{sid}_{safe}.json"), "w"),
                  ensure_ascii=False)
        full[p["pk"]] = {"meta": p, "data": data}

    json.dump(full, open(os.path.join(out_dir, f"{project}_full.json"), "w"),
              ensure_ascii=False)

    # long-form CSV of every curve point found
    csv_path = os.path.join(out_dir, f"{project}_curves.csv")
    n_pts = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pk", "sid", "doi", "path", "prop_x", "prop_y", "x", "y"])
        for pk, entry in full.items():
            meta = entry["meta"]
            for path, px, py, _node, pts in _walk_curves(entry["data"]):
                for x, y in pts:
                    w.writerow([pk, meta.get("sid"), meta.get("doi"),
                                "/".join(path), px, py, x, y])
                    n_pts += 1

    print(f"\n✓ done: {len(papers)} papers ({n_err} errors) → {out_dir}")
    print(f"  curves CSV: {csv_path} ({n_pts} points)")
    print(f"  full JSON:  {project}_full.json")


if __name__ == "__main__":
    main()
