# -*- coding: utf-8 -*-
"""
kmds_vocab.py — the KMDS property vocabulary, for showing the curator what a
digitized figure IS in KMDS terms ("temperature vs electrical conductivity")
before anything is uploaded.

Mirrors Starrydata3's vocabulary walk and property matching (starrydata3/
manage.py + app.py) so the app's preview and the server's ingest agree. Only
honest matches: a name that isn't in the vocabulary returns None and the UI
shows it as non-KMDS — nothing is self-mapped to look conformant.
"""
import json
import os
import re
from functools import lru_cache
from typing import Dict, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.environ.get(
    "KMDS_SCHEMA", os.path.join(SCRIPT_DIR, "kmds_v15.2.4_nullable.json"))

_WRAPPER_KEYS = {"value", "unit", "uncertainty", "measurement"}

# Kept in sync with starrydata3.app._PROP_SYNONYMS — every target is a term
# that actually exists in the vocabulary.
_SYNONYMS = {
    "temp": "temperature",
    "seebeck": "Seebeck coefficient", "thermopower": "Seebeck coefficient",
    "seebeck coeff": "Seebeck coefficient",
    "figure of merit": "ZT", "thermoelectric figure of merit": "ZT",
    "permittivity": "dielectric constant", "relative permittivity": "dielectric constant",
    "dielectric loss": "loss tangent", "loss factor": "loss tangent",
    "tan delta": "loss tangent", "dissipation factor": "loss tangent",
    "t g": "glass transition temperature", "tg": "glass transition temperature",
    # battery axis wordings -> extension terms (see kmds_vocab_extensions.json)
    "amphrs": "capacity", "amp hrs": "capacity", "amp hours": "capacity",
    "ah": "capacity", "mah": "capacity", "amp hr": "capacity",
    "volts": "voltage", "volt": "voltage", "cell voltage": "voltage", "discharge capacity": "capacity",
    "charge capacity": "capacity",
    "ic": "incremental capacity", "dq dv": "incremental capacity",
    "dqdv": "incremental capacity", "power output": "output power",
    "delta t": "temperature difference",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _vocab_triples(schema_path):
    """(name, category, unit) triples from the KMDS materials property tree."""
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    try:
        proot = schema["properties"]["materials"]["items"]["properties"]["property"]
    except KeyError:
        return []
    out = []

    def _cprops(node):
        p = node.get("properties")
        return p if isinstance(p, dict) else {}

    def collect(node, category):
        for name, child in _cprops(node).items():
            if not isinstance(child, dict) or name in _WRAPPER_KEYS:
                continue
            cp = _cprops(child)
            val = cp.get("value")
            vtype = val.get("type") if isinstance(val, dict) else None
            is_array = vtype == "array" or (isinstance(vtype, list) and "array" in vtype)
            if isinstance(val, dict) and is_array and "items" in val:
                collect(val["items"], name)
            elif not cp or set(cp).issubset(_WRAPPER_KEYS):
                vnode = cp.get("value") if isinstance(cp.get("value"), dict) else {}
                unit = child.get("unit") or vnode.get("unit") or vnode.get("examples") or ""
                out.append((name, category, unit if isinstance(unit, str) else ""))
            else:
                collect(child, name)

    collect(proot, "")
    seen, uniq = set(), []
    for name, cat, unit in out:
        if name and name not in seen and name not in _WRAPPER_KEYS:
            seen.add(name)
            uniq.append((name, cat, unit))
    return uniq


EXTENSIONS_PATH = os.path.join(SCRIPT_DIR, "kmds_vocab_extensions.json")


@lru_cache(maxsize=1)
def _tables() -> Tuple[Dict[str, str], Dict[str, str], set]:
    """(normalized name -> canonical term, canonical term -> unit,
    extension term names). Extensions are locally-defined terms the official
    KMDS schema lacks (see kmds_vocab_extensions.json) — matched like any
    vocabulary term but reported as extensions, never as official KMDS."""
    norm_map, units, ext = {}, {}, set()
    try:
        for name, _cat, unit in _vocab_triples(SCHEMA_PATH):
            norm_map[_norm(name)] = name
            units[name] = unit
    except Exception:  # noqa: BLE001 — no vocab file: match() just returns None
        pass
    try:
        with open(EXTENSIONS_PATH, encoding="utf-8") as f:
            for e in json.load(f).get("extensions") or []:
                name = (e.get("name") or "").strip()
                if not name or _norm(name) in norm_map:
                    continue
                norm_map[_norm(name)] = name
                units[name] = e.get("unit") or ""
                ext.add(name)
    except Exception:  # noqa: BLE001 — extensions are optional
        pass
    return norm_map, units, ext


def available() -> bool:
    return bool(_tables()[0])


def is_extension(term: str) -> bool:
    return term in _tables()[2]


def add_extensions(entries) -> list:
    """Append new extension terms to kmds_vocab_extensions.json (deduped
    against the whole vocabulary) and reload. Entries: [{name, category,
    unit, ...}]. Extra keys (e.g. added_by) are kept for provenance.
    Returns the names actually added."""
    norm_map = _tables()[0]
    fresh = []
    seen = set()
    for e in entries or []:
        name = (e.get("name") or "").strip()
        if not name or _norm(name) in norm_map or _norm(name) in seen:
            continue
        seen.add(_norm(name))
        fresh.append({"name": name,
                      "category": (e.get("category") or "property").strip()
                      + (" (extension)" if "(extension)" not in (e.get("category") or "") else ""),
                      "unit": (e.get("unit") or "").strip(),
                      **({"added_by": e["added_by"]} if e.get("added_by") else {})})
    if not fresh:
        return []
    try:
        with open(EXTENSIONS_PATH, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:  # noqa: BLE001
        doc = {"extensions": []}
    doc.setdefault("extensions", []).extend(fresh)
    with open(EXTENSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    _tables.cache_clear()
    return [e["name"] for e in fresh]


def match(name: str) -> Optional[str]:
    """Axis/OCR property name -> canonical KMDS term, or None if not a KMDS
    property. A trailing '(unit)' is split off and used only to disambiguate
    single-letter symbols ('T (K)' -> temperature). Same rules as Starrydata3's
    ingest: exact normalized match, hand synonym, unique substring hit."""
    norm_map = _tables()[0]
    if not norm_map:
        return None
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", (name or "").strip())
    base, unit = (m.group(1), m.group(2)) if m else (name or "", "")
    n = _norm(base)
    if not n:
        return None
    if n in norm_map:
        return norm_map[n]
    syn = _SYNONYMS.get(n)
    if syn:
        return syn
    # ΔT is temperature DIFFERENCE — check before the bare-T rule, because
    # normalization strips the Δ and would otherwise leave just 't'
    if re.match(r"^\s*(Δ|∆|[Dd]elta)\s*T\s*$", base.strip()):
        return norm_map.get("temperature difference")
    # symbol + unit disambiguation: 'T' is temperature ONLY when the unit says so
    if n == "t" and re.match(r"^\s*(K|°\s*C|deg\s*C|℃)\s*$", unit):
        return "temperature"
    hits = [canon for k, canon in norm_map.items() if n == k or (len(n) > 4 and n in k)]
    return hits[0] if len(set(hits)) == 1 else None


_UNIT_TEX = [(r"\^\{\\circ\}\s*C", "°C"), (r"\\circ", "°"), (r"\\Omega", "Ω"),
             (r"\\cdot", "·"), (r"\\times", "×"), (r"\\mu", "µ"), (r"\\Delta", "Δ")]


def unit_of(term: str) -> str:
    """The property's canonical KMDS unit, LaTeX cleaned for display."""
    u = _tables()[1].get(term, "") or ""
    if u.lower() in ("none", "-"):
        return ""
    for pat, rep in _UNIT_TEX:
        u = re.sub(pat, rep, u)
    u = re.sub(r"\^\{([^}]*)\}", r"^\1", u)
    u = re.sub(r"_\{([^}]*)\}", r"_\1", u)
    u = u.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", u).strip()
