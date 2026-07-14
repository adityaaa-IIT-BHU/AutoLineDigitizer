#!/usr/bin/env python3
"""
record_starrydata.py — capture Starrydata's internal write API by watching a
real manual entry. No Tampermonkey, no DevTools: a Chrome window opens, YOU log
in and do ONE normal data entry, then close the window. Every network call the
site made is written to a JSON log that documents its endpoints and payloads.

    external/pw-venv/bin/python tools/record_starrydata.py
    # optional first arg: start URL (defaults to the public site)

Nothing is automated — the script only observes. Non-GET calls (the writes) are
highlighted in the console live so you can see it working.
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

START_URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.starrydata2.org/"
OUT = Path(__file__).resolve().parent.parent / "starrydata_api_capture.json"

SKIP_EXT = (".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".woff", ".woff2",
            ".gif", ".ico", ".map", ".webp")

calls = []


def _same_site(url: str) -> bool:
    return ("starrydata" in url) and not any(url.split("?")[0].endswith(e) for e in SKIP_EXT)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()

        def _scrub(body):
            """Never persist credentials — redact auth fields in any body."""
            if not isinstance(body, str):
                return None
            import re as _re
            for field in ("password", "code", "username", "csrfmiddlewaretoken"):
                body = _re.sub(field + r"=[^&]*", field + "=***REDACTED***", body)
            return body

        def on_request(req):
            if req.method == "GET" or not _same_site(req.url):
                return
            # Skip login / MFA endpoints entirely — we never need their traffic.
            if any(s in req.url for s in ("/login", "/mfa", "/auth", "/accounts/")):
                return
            body = None
            try:
                body = _scrub(req.post_data)
            except Exception:
                pass
            entry = {
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "method": req.method,
                "url": req.url,
                "headers": {k: v for k, v in req.headers.items()
                            if k.lower() in ("content-type", "x-csrftoken",
                                             "x-requested-with", "authorization",
                                             "referer", "accept")},
                "post_data": body[:20000] if isinstance(body, str) else None,
                "status": None,
            }
            calls.append(entry)
            print(f"  ● {req.method} {req.url.split('starrydata')[-1][:90]}")

        def on_response(resp):
            if resp.request.method == "GET" or not _same_site(resp.url):
                return
            for c in reversed(calls):
                if c["url"] == resp.url and c["status"] is None:
                    c["status"] = resp.status
                    try:
                        txt = resp.text()
                        c["response_preview"] = txt[:2000]
                    except Exception:
                        pass
                    break

        page.on("request", on_request)
        page.on("response", on_response)

        page.goto(START_URL, wait_until="domcontentloaded")
        print("\n" + "=" * 68)
        print("  Chrome is open. Now, in that window:")
        print("   1. Log in to Starrydata.")
        print("   2. Do ONE complete manual data entry:")
        print("      paper -> figure -> set axes -> click a few points -> sample -> SAVE.")
        print("   3. Close the browser window when done.")
        print("  Write calls (POST/PUT/PATCH) appear below as you go:")
        print("=" * 68 + "\n")

        # Block until the user closes the window.
        try:
            page.wait_for_event("close", timeout=0)
        except Exception:
            pass
        try:
            browser.close()
        except Exception:
            pass

    OUT.write_text(json.dumps({"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "start_url": START_URL, "n_calls": len(calls),
                               "calls": calls}, indent=2, ensure_ascii=False))
    writes = [c for c in calls if c["method"] != "GET"]
    print(f"\n✅ Saved {len(calls)} same-site call(s) ({len(writes)} write call(s)) -> {OUT}")
    if not writes:
        print("   ⚠ No write calls captured — did the manual entry get saved?")


if __name__ == "__main__":
    main()
