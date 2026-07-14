// ==UserScript==
// @name         Starrydata × AutoLineDigitizer Helper
// @namespace    https://github.com/adityaaa-IIT-BHU/AutoLineDigitizer
// @version      0.1.0
// @description  Read-side helper for Starrydata: (1) coverage check for a paper (SID) via the public Bulk Data API; (2) passive recorder that maps the site's internal write endpoints while you do one manual data entry. No automated writes.
// @author       AutoLineDigitizer
// @match        https://www.starrydata2.org/*
// @match        https://starrydata2.org/*
// @match        https://starrydata.nims.go.jp/*
// @match        https://starrydata-stg.nims.go.jp/*
// @run-at       document-start
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  const BULK = "https://starrydata.github.io/bulk-data-api/v1";
  const LOG_KEY = "ald_api_recorder_log";

  // ---------------------------------------------------------------------
  // 1) WRITE-ENDPOINT RECORDER — patch fetch/XHR before the app loads.
  //    Passively logs every non-GET request the site itself makes, so one
  //    manual data entry produces a complete map of the internal write API.
  // ---------------------------------------------------------------------
  const records = JSON.parse(sessionStorage.getItem(LOG_KEY) || "[]");

  function logCall(method, url, body, status) {
    if (!method || method.toUpperCase() === "GET") return;
    try {
      const u = new URL(url, location.href);
      if (u.origin !== location.origin) return;          // same-origin only
      if (/\.(js|css|png|jpg|svg|woff2?)(\?|$)/.test(u.pathname)) return;
      records.push({
        t: new Date().toISOString(),
        method: method.toUpperCase(),
        path: u.pathname + u.search,
        body: typeof body === "string" ? body.slice(0, 20000) : (body ? "[non-string body]" : null),
        status: status ?? null,
        page: location.pathname,
      });
      sessionStorage.setItem(LOG_KEY, JSON.stringify(records.slice(-500)));
      updateBadge();
    } catch (e) { /* never break the site */ }
  }

  const origFetch = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    const method = (init && init.method) || (input && input.method) || "GET";
    const body = init && init.body;
    return origFetch.apply(this, arguments).then((resp) => {
      logCall(method, url, typeof body === "string" ? body : null, resp.status);
      return resp;
    });
  };

  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url) {
    this.__ald = { method, url };
    return origOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function (body) {
    const meta = this.__ald;
    if (meta) {
      this.addEventListener("loadend", () => {
        logCall(meta.method, meta.url, typeof body === "string" ? body : null, this.status);
      });
    }
    return origSend.apply(this, arguments);
  };

  // ---------------------------------------------------------------------
  // 2) UI PANEL (coverage check + recorder controls)
  // ---------------------------------------------------------------------
  let badge, panelBody;

  function updateBadge() {
    if (badge) badge.textContent = String(records.length);
  }

  function el(tag, style, text) {
    const n = document.createElement(tag);
    if (style) n.style.cssText = style;
    if (text != null) n.textContent = text;
    return n;
  }

  function detectSID() {
    const m = (location.href + " " + document.title).match(/\b(S[A-Z]?\d{4,})\b/i);
    return m ? m[1].toUpperCase() : "";
  }

  async function coverageCheck(sid, pairs, out) {
    out.textContent = "Fetching graph data (" + pairs.length + " property pair(s))…";
    const rows = [];
    for (const p of pairs) {
      const name = `${p.prop_x}-${p.prop_y}`;
      out.textContent = `Fetching ${name}.json …`;
      try {
        const resp = await origFetch(`${BULK}/${encodeURIComponent(p.prop_x)}-${encodeURIComponent(p.prop_y)}.json`);
        if (!resp.ok) { rows.push([name, "n/a", "(file missing)"]); continue; }
        const data = await resp.json();
        const curves = (Array.isArray(data) ? data : data.graphs || data.data || [])
          .filter((c) => String(c.SID || "").toUpperCase() === sid);
        const comps = [...new Set(curves.map((c) => c.composition).filter(Boolean))];
        rows.push([name, String(curves.length), comps.slice(0, 6).join(", ") + (comps.length > 6 ? " …" : "")]);
      } catch (e) {
        rows.push([name, "err", String(e).slice(0, 60)]);
      }
    }
    out.textContent = "";
    const table = el("table", "width:100%;border-collapse:collapse;font-size:11px;");
    const head = table.insertRow();
    ["property pair", "#curves", "compositions already in Starrydata"].forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      th.style.cssText = "text-align:left;border-bottom:1px solid #1d5fa0;padding:2px 6px;color:#9fc0e2;";
      head.appendChild(th);
    });
    let total = 0;
    rows.forEach(([a, b, c]) => {
      const tr = table.insertRow();
      [a, b, c].forEach((v) => {
        const td = tr.insertCell();
        td.textContent = v;
        td.style.cssText = "padding:2px 6px;border-bottom:1px solid rgba(255,255,255,.1);vertical-align:top;";
      });
      total += parseInt(b, 10) || 0;
    });
    out.appendChild(el("div", "margin:6px 0 4px;font-weight:600;",
      total ? `${sid}: ${total} curve(s) already in Starrydata` : `${sid}: nothing found in the checked pairs`));
    out.appendChild(table);
  }

  function buildPanel() {
    if (document.getElementById("ald-sd-panel")) return;
    const panel = el("div",
      "position:fixed;bottom:16px;right:16px;z-index:2147483000;width:380px;max-height:70vh;overflow:auto;" +
      "background:#0d2544;color:#eaf2fb;border:1px solid #1d5fa0;border-radius:10px;" +
      "font:12px/1.45 -apple-system,Segoe UI,sans-serif;box-shadow:0 10px 30px rgba(0,0,0,.4);");
    panel.id = "ald-sd-panel";

    const header = el("div", "display:flex;align-items:center;gap:8px;padding:8px 12px;cursor:pointer;" +
      "border-bottom:1px solid rgba(255,255,255,.15);font-weight:700;");
    header.appendChild(el("span", "color:#c89211;", "★"));
    header.appendChild(el("span", "", "AutoLineDigitizer helper"));
    badge = el("span", "margin-left:auto;background:#1d5fa0;border-radius:9px;padding:1px 8px;font-weight:600;",
      String(records.length));
    badge.title = "write-API calls recorded this session";
    header.appendChild(badge);
    panel.appendChild(header);

    panelBody = el("div", "padding:10px 12px;display:none;");
    panel.appendChild(panelBody);
    header.onclick = () => {
      panelBody.style.display = panelBody.style.display === "none" ? "block" : "none";
    };

    // --- coverage section ---
    panelBody.appendChild(el("div", "font-weight:700;margin-bottom:4px;color:#9fc0e2;", "1 · Paper coverage (read API)"));
    const sidRow = el("div", "display:flex;gap:6px;margin-bottom:6px;");
    const sidInput = el("input", "flex:1;padding:4px 6px;border-radius:6px;border:1px solid #1d5fa0;background:#122f52;color:#eaf2fb;");
    sidInput.placeholder = "SID (e.g. S123456)";
    sidInput.value = detectSID();
    const checkBtn = el("button", "padding:4px 10px;border-radius:6px;border:0;background:#1d5fa0;color:#fff;cursor:pointer;", "Check");
    sidRow.appendChild(sidInput);
    sidRow.appendChild(checkBtn);
    panelBody.appendChild(sidRow);
    const pairBox = el("div", "max-height:110px;overflow:auto;border:1px solid rgba(255,255,255,.15);border-radius:6px;padding:4px 6px;margin-bottom:6px;");
    pairBox.textContent = "Loading property-pair list…";
    panelBody.appendChild(pairBox);
    const covOut = el("div", "margin-bottom:10px;");
    panelBody.appendChild(covOut);

    const pairChecks = [];
    origFetch(`${BULK}/graph_list.json`).then((r) => r.json()).then((gl) => {
      pairBox.textContent = "";
      (gl.graphs || []).slice(0, 25).forEach((p, i) => {
        const lab = el("label", "display:block;cursor:pointer;");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = i < 4;   // top-4 pairs by default (~13MB each — mind the network)
        cb.style.marginRight = "6px";
        lab.appendChild(cb);
        lab.appendChild(document.createTextNode(`${p.prop_x} – ${p.prop_y}  (${p.count})`));
        pairBox.appendChild(lab);
        pairChecks.push([cb, p]);
      });
    }).catch(() => { pairBox.textContent = "Could not load graph_list.json"; });

    checkBtn.onclick = () => {
      const sid = sidInput.value.trim().toUpperCase();
      if (!sid) { covOut.textContent = "Enter the paper's SID first."; return; }
      const pairs = pairChecks.filter(([cb]) => cb.checked).map(([, p]) => p);
      if (!pairs.length) { covOut.textContent = "Select at least one property pair."; return; }
      coverageCheck(sid, pairs, covOut);
    };

    // --- recorder section ---
    panelBody.appendChild(el("div", "font-weight:700;margin:8px 0 4px;color:#9fc0e2;", "2 · Write-API recorder (passive)"));
    panelBody.appendChild(el("div", "color:#c3d6ea;margin-bottom:6px;",
      "Do ONE manual data entry (paper → figure → axes → points → sample). Every write request the site makes is recorded here. Then download the log and send it to the AutoLineDigitizer side."));
    const btnRow = el("div", "display:flex;gap:6px;");
    const dlBtn = el("button", "padding:4px 10px;border-radius:6px;border:0;background:#c89211;color:#0d2544;font-weight:700;cursor:pointer;", "Download log");
    const clrBtn = el("button", "padding:4px 10px;border-radius:6px;border:1px solid #1d5fa0;background:transparent;color:#9fc0e2;cursor:pointer;", "Clear");
    btnRow.appendChild(dlBtn);
    btnRow.appendChild(clrBtn);
    panelBody.appendChild(btnRow);

    dlBtn.onclick = () => {
      const blob = new Blob([JSON.stringify({ recorded_at: new Date().toISOString(),
        origin: location.origin, calls: records }, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "starrydata_write_api_log.json";
      a.click();
      URL.revokeObjectURL(a.href);
    };
    clrBtn.onclick = () => {
      records.length = 0;
      sessionStorage.removeItem(LOG_KEY);
      updateBadge();
    };

    document.body.appendChild(panel);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildPanel);
  } else {
    buildPanel();
  }
})();
