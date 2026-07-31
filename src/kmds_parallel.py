# -*- coding: utf-8 -*-
"""
kmds_parallel.py — Parallel KMDS extraction (5 focused section calls + 1 translation).

Replaces the single monolithic ~16K-token Claude call in extract_paper.py with
5 concurrent, focused section calls (asyncio.gather) merged in Python, followed
by one English→Japanese translation pass. Target: ~5 min → under ~90 sec.

Design (approved):
  - asyncio + anthropic.AsyncAnthropic
  - Sonnet 4.6 for ALL calls (section + translation)
  - PDF document block carries cache_control ephemeral, 1-hour TTL
  - Pure 5-way gather (no cache pre-warming)
  - Universal ground rules are copied VERBATIM from extraction_prompt.md at the
    top of every section prompt (extracted from the file at runtime — no drift)
  - figures/graphs/tables merge under metadata.publication.figures[]/.graphs[]/.tables[]
  - If any section fails: save partials + continue; never crash the whole run

Public entry point:
  await extract_kmds_parallel(pdf_path, output_dir, base_name=None,
                              prompt_path="extraction_prompt.md")
Returns a summary dict (same en_path/ja_path/input_tokens/output_tokens/elapsed_sec
shape extract_paper.py already expects, plus per-section detail).
"""

import os
import re
import json
import time
import base64
import asyncio
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False


MODEL = "claude-sonnet-4-6"          # extraction calls: Sonnet 4.6
LIGHT_MODEL = "claude-haiku-4-5-20251001"  # 3x cheaper input+output than Sonnet
TRANSLATION_MODEL = "claude-haiku-4-5-20251001"  # JA translation is mechanical — Haiku is ~3x faster

# Per-section model. core (the id namespaces gate every cross-reference) and
# data_sources (vision + the densest relational linking) stay on Sonnet; the
# three materials detail sections are focused read-and-fill jobs where Haiku
# holds up — any that fails (API error / unparseable JSON) is retried once on
# the full model before the run gives up on it.
SECTION_MODELS = {
    "materials_chem": LIGHT_MODEL,
    "materials_process": LIGHT_MODEL,
    "materials_property": LIGHT_MODEL,
}
SECTION_MAX_TOKENS = 12000           # headroom for publication + many-figure papers
TRANSLATION_MAX_TOKENS = 32000       # JA output ≈ EN size (records with passages run large)


# ===================================================================
# Verbatim rule extraction from extraction_prompt.md (no paraphrase)
# ===================================================================

def _slice(text: str, start_marker: str, end_marker: str) -> str:
    """Return text from start_marker up to (not including) end_marker, verbatim."""
    i = text.find(start_marker)
    if i == -1:
        return ""
    j = text.find(end_marker, i + len(start_marker))
    if j == -1:
        j = len(text)
    return text[i:j].strip()


def load_prompt_blocks(prompt_path: str) -> Dict[str, str]:
    """Pull the verbatim rule/overview blocks out of extraction_prompt.md.

    These are sliced (not retyped) so the section prompts can never drift from
    the source specification.
    """
    txt = open(prompt_path, "r", encoding="utf-8").read()
    return {
        # Universal ground rules — prepended verbatim to EVERY section call.
        "ground_rules": _slice(txt, "### Project intent & ground rules",
                               "### What counts as a material vs a sample"),
        # Material-vs-sample distinction — for publication + materials calls.
        "material_vs_sample": _slice(txt, "### What counts as a material vs a sample",
                                     "### Populate structured conditions"),
        # "Fill structured conditions" rules — for process/property calls.
        "conditions": _slice(txt, "### Populate structured conditions",
                             "### Schema structure overview"),
        # Schema navigation map — inlined into every section call.
        "overview": _slice(txt, "### Schema structure overview",
                           "### Output format"),
        # Translation rules — for the JA pass.
        "translation": _slice(txt, "### Output format (two files: English + Japanese)",
                              "### Schema extension candidates"),
    }


# ===================================================================
# Section-specific instructions + explicit output skeletons
# ===================================================================

def _skel(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# Section instructions (what to extract). The authoritative STRUCTURE comes from
# the per-section JSON Schema fragment appended at call time (build_section_schemas).
#
# The pipeline is TWO-PHASE because the schema is relational: publication.samples[]
# defines the s-NNN namespace that graphs/tables/passages/conclusions reference, and
# materials[] ids are referenced from samples[].components[].references. Phase 1
# ("core") establishes both namespaces in a single call; phase-2 calls receive that
# namespace verbatim so their cross-references stay consistent.
PHASE1_KEY = "core"

SUB_PROMPTS: Dict[str, str] = {
    # PHASE 1 — record core + id namespaces (samples s-NNN, materials material_NN)
    "core": (
        "## For THIS call only (PHASE 1 — record core and id namespaces)\n"
        "Extract:\n"
        "1. `metadata` — all six required fields plus `rights`, and `metadata.publication` "
        "with the bibliography (DOI, title, authors[] with roles/affiliations, journal, "
        "year, dates, volume/issue/pages, `abstract` copied from the printed abstract, "
        "type, open access status, and `references` = the DOIs printed in the reference "
        "list — skip entries without a printed DOI), `scope` (paradigms, purposes, "
        "approaches, conclusions[], classifications, comments), and `samples[]`.\n"
        "2. Top-level `system` (null unless a `name` enum member genuinely applies).\n"
        "3. Top-level `materials` — a NAMESPACE ONLY: one `{id, name}` entry per distinct "
        "material system/composition originally studied in this paper, ids `material_01`, "
        "`material_02`, … in order of appearance. Do NOT fill chemical "
        "information/structure/process/property here — later calls do that.\n"
        "\n"
        "Cross-linking rules (the heart of this schema — do not skip):\n"
        "- `samples[].components[]`: for every component that corresponds to a "
        "`materials[]` entry, add `{\"scheme\": \"KMDS material\", \"id\": "
        "\"material_NN\"}` to its `references`; fill `role` and `fraction value`/"
        "`fraction unit` when the paper states them.\n"
        "- `scope.conclusions[]`: each entry is `{text, samples}` — list the `sample "
        "local id`s the conclusion concerns.\n"
        "Do NOT output figures/graphs/tables/`text passages` keys — a separate call "
        "handles those; omit them entirely."
    ),
    # PHASE 2 — figures (graphs NESTED) + tables + text passages, cross-linked to s-NNN
    "data_sources": (
        "## For THIS call only (PHASE 2 — data sources: figures, tables, text passages)\n"
        "Using the sample-id namespace given below, extract the paper's data sources:\n"
        "- `figures[]` — every figure. Nest each figure's graphs INSIDE its `graphs[]` "
        "array (NEVER as a separate top-level array). For each graph fill: `axes[]` "
        "(axis, quantity.term, unit in TeX notation, reference ticks, scale, SI "
        "conversion), `samples` = the `sample local id`s plotted in that graph (from the "
        "GIVEN namespace), `structure` (brief), `description` (objective detail; mention "
        "series values in prose, not as data), and `digitization` as \"\".\n"
        "- `tables[]` — headers, rows, `samples` (from the namespace), structure, "
        "description.\n"
        "- `text passages[]` — body-text passages that state numeric data or key "
        "conditions (peak values quoted in abstract/results, Methods conditions): "
        "`passage local id` p-001…, `passage name`, `section`, `text` (verbatim), "
        "`samples`.\n"
        "Use ONLY sample ids from the given namespace. Return top-level keys `figures`, "
        "`tables`, and `text passages`. Do NOT digitize plotted curves."
    ),
    # PHASE 2 — composition + doping + purity + structure per material
    "materials_chem": (
        "## For THIS call only (PHASE 2)\n"
        "Extract ONLY each material's `chemical information` and `structure`. Use "
        "EXACTLY the material `id`s and names given below — same order, no new ids, no "
        "renames, no extra entries for precursors or literature materials. Do NOT "
        "extract `process` or `property` here — omit those keys. Return a top-level "
        "`materials` array."
    ),
    # PHASE 2 — process[] per material
    "materials_process": (
        "## For THIS call only (PHASE 2)\n"
        "Extract ONLY each material's `process` (synthesis/processing steps such as ball "
        "milling, arc melting, spark plasma sintering, encapsulated melting). For every "
        "process type you select, FILL its structured condition sub-fields from Methods — "
        "do not leave it empty. Use EXACTLY the material `id`s given below. Return a "
        "top-level `materials` array, each item with `id` and `process` only."
    ),
    # PHASE 2 — measured properties + measurement conditions
    "materials_property": (
        "## For THIS call only (PHASE 2)\n"
        "Extract ONLY each material's measured `property` values (with the measurement "
        "conditions the schema nests inside it). Record only numbers stated EXPLICITLY in "
        "text/tables/captions — do NOT digitize plotted curves. Use EXACTLY the material "
        "`id`s given below. Return a top-level `materials` array, each item with `id` and "
        "`property` only."
    ),
}

# Extra rule blocks each section should also receive (beyond ground rules + overview).
_SECTION_EXTRA = {
    "core": ("material_vs_sample",),
    "data_sources": (),
    "materials_chem": ("material_vs_sample",),
    "materials_process": ("conditions",),
    "materials_property": ("conditions",),
}

# Per-section output budget (data_sources carries figures+tables+passages).
_SECTION_MAX = {"data_sources": 16000}


# ===================================================================
# Per-section JSON Schema slicing (so each call conforms to the real schema)
# ===================================================================

_REF_RE = re.compile(r'"\$ref":\s*"#/\$defs/([^"]+)"')


def _refs_in(obj) -> set:
    return set(_REF_RE.findall(json.dumps(obj)))


def _def_closure(defs: Dict[str, Any], roots) -> set:
    """All $defs transitively referenced from `roots` (a set/iterable of names)."""
    seen, stack = set(), list(roots)
    while stack:
        d = stack.pop()
        if d in seen or d not in defs:
            continue
        seen.add(d)
        stack.extend(_refs_in(defs[d]))
    return seen


def build_section_schemas(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Slice the full KMDS schema into a focused sub-schema per section call, so the
    model conforms to the REAL field names / value types / enums instead of a
    hand-written skeleton. Each fragment carries only the relevant top-level
    properties plus the $defs they transitively reference.
    """
    defs = schema.get("$defs", {})
    metaprops = schema["properties"]["metadata"]["properties"]
    sysprop = schema["properties"].get("system")
    mitems = schema["properties"]["materials"]["items"]
    mprops = mitems.get("properties", {})

    # `process` (~53k tok) and `measurement` (~50k tok) are the huge taxonomies.
    # The dense cross-references would pull them into every closure, so keep each
    # only in the section that actually owns it (the others get a dangling $ref,
    # which is fine — the fragment is guidance for the model, not API-validated).
    HEAVY = {"process", "measurement"}

    def _pick_defs(names, owned_heavy=()):
        cl = _def_closure(defs, names)
        cl = {d for d in cl if d not in HEAVY or d in owned_heavy}
        return {k: defs[k] for k in sorted(cl) if k in defs}

    def _materials_fragment(keys, owned_heavy=()):
        sel = {k: mprops[k] for k in keys if k in mprops}
        return {
            "$defs": _pick_defs(_refs_in(sel), owned_heavy=owned_heavy),
            "properties": {
                "materials": {"type": "array", "items": {
                    "type": "object", "properties": sel}},
            },
        }

    out: Dict[str, Dict[str, Any]] = {}

    # core (phase 1): metadata (incl. publication ref) + system + materials id/name
    # namespace. Omit figure/graph/table/axis/passage defs (data_sources owns them).
    core_props = {"metadata": {"type": "object",
                               "properties": {k: metaprops[k] for k in metaprops}}}
    if sysprop is not None:
        core_props["system"] = sysprop
    core_props["materials"] = {"type": "array", "items": {
        "type": "object", "required": ["id", "name"],
        "properties": {k: mprops[k] for k in ("id", "name") if k in mprops}}}
    core_defs = _pick_defs(_refs_in(core_props))
    for drop in ("figure", "graph", "table", "axis", "passage"):
        core_defs.pop(drop, None)
    out["core"] = {"$defs": core_defs, "properties": core_props}

    out["materials_chem"] = _materials_fragment(["id", "name", "chemical information", "structure"])
    out["materials_process"] = _materials_fragment(["id", "process"], owned_heavy=("process",))
    out["materials_property"] = _materials_fragment(["id", "property"], owned_heavy=("measurement",))

    # data_sources (phase 2): figures (graphs nested inside) + tables + text passages,
    # all merged under metadata.publication later.
    ds_props = {
        "figures": {"type": "array", "items": {"$ref": "#/$defs/figure"}},
        "tables": {"type": "array", "items": {"$ref": "#/$defs/table"}},
        "text passages": {"type": "array", "items": {"$ref": "#/$defs/passage"}},
    }
    out["data_sources"] = {"$defs": _pick_defs({"figure", "table", "passage"}),
                           "properties": ds_props}
    return out


# ===================================================================
# Deterministic axes[].quantity.ref filler (schema property vocabulary)
# ===================================================================

# structural / bookkeeping keys that are not property names themselves
_NON_TERM_KEYS = {
    "value", "uncertainty", "measurement", "unit", "notes", "comments",
    "description", "alias", "examples", "properties", "items", "type",
    "required", "enum", "additionalProperties",
}


def _build_property_ref_index(schema: Dict[str, Any]) -> Dict[str, str]:
    """Map normalized property-name/alias -> unique JSON Pointer into the KMDS
    materials property vocabulary (for axes[].quantity.ref). Ambiguous terms
    (multiple pointers, e.g. 'temperature') are dropped."""
    try:
        prop_root = schema["properties"]["materials"]["items"]["properties"]["property"]
    except (KeyError, TypeError):
        return {}
    base = "#/properties/materials/items/properties/property"
    found: Dict[str, set] = {}

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def _add(term: str, ptr: str):
        if term and len(term) > 1:
            found.setdefault(_norm(term), set()).add(ptr)

    def _walk(node: Any, ptr: str):
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            for k, v in props.items():
                if not isinstance(v, dict):
                    continue
                child = f"{ptr}/properties/{k}"
                if k not in _NON_TERM_KEYS:
                    _add(k, child)
                    alias = v.get("alias")
                    if isinstance(alias, str):
                        for a in re.split(r"[;,]", alias):
                            _add(a, child)
                _walk(v, child)
        items = node.get("items")
        if isinstance(items, dict):
            _walk(items, f"{ptr}/items")

    _walk(prop_root, base)
    return {t: next(iter(ps)) for t, ps in found.items() if len(ps) == 1}


def fill_axis_refs(record: Dict[str, Any], schema: Dict[str, Any]) -> int:
    """Fill axes[].quantity.ref by matching quantity.term against the schema's
    property vocabulary. Deterministic, fills only unique matches. Returns count."""
    index = _build_property_ref_index(schema)
    if not index:
        return 0
    n = 0
    pub = ((record.get("metadata") or {}).get("publication") or {})
    for fig in (pub.get("figures") or []):
        if not isinstance(fig, dict):
            continue
        for g in (fig.get("graphs") or []):
            if not isinstance(g, dict):
                continue
            for ax in (g.get("axes") or []):
                if not isinstance(ax, dict):
                    continue
                q = ax.get("quantity")
                if not isinstance(q, dict) or q.get("ref"):
                    continue
                term = q.get("term")
                if not isinstance(term, str):
                    continue
                ptr = index.get(re.sub(r"\s+", " ", term.strip().lower()))
                if ptr:
                    q["ref"] = ptr
                    n += 1
    return n


def repair_record(record: Dict[str, Any], schema: Dict[str, Any],
                  max_rounds: int = 6) -> List[str]:
    """Deterministic, schema-guided cleanup of common model slip-ups, in place:
    - snake_case keys renamed to the schema's spaced keys ('jar_material' ->
      'jar material') when the spaced key is what the schema defines;
    - other unknown keys moved into a sibling notes/comments array when the
      schema has one, else dropped (logged);
    - {'value': x, ...} wrappers flattened where the schema wants a bare scalar;
    - missing REQUIRED keys added as null when the schema allows null.
    Never invents data. Returns a log of the repairs made."""
    try:
        from jsonschema import Draft202012Validator
    except Exception:
        return []
    log: List[str] = []
    v = Draft202012Validator(schema)
    for _ in range(max_rounds):
        try:
            errors = sorted(v.iter_errors(record), key=lambda e: list(e.path))
        except Exception:
            break
        if not errors:
            break
        changed = False
        for e in errors:
            obj: Any = record
            path = list(e.path)
            try:
                for p in path:
                    obj = obj[p]
            except (KeyError, IndexError, TypeError):
                continue
            loc = "/".join(str(p) for p in path) or "(root)"

            if e.validator == "additionalProperties" and isinstance(obj, dict):
                allowed = set((e.schema.get("properties") or {}).keys())
                for k in [k for k in list(obj.keys()) if k not in allowed]:
                    spaced = k.replace("_", " ")
                    if spaced in allowed and spaced not in obj:
                        obj[spaced] = obj.pop(k)
                        log.append(f"[{loc}] renamed '{k}' -> '{spaced}'")
                    elif ("comments" in allowed or "notes" in allowed):
                        tgt = "comments" if "comments" in allowed else "notes"
                        arr = obj.get(tgt)
                        if not isinstance(arr, list):
                            arr = obj[tgt] = []
                        arr.append(f"{k}: {json.dumps(obj.pop(k), ensure_ascii=False)}")
                        log.append(f"[{loc}] moved unknown key '{k}' into {tgt}[]")
                    else:
                        val = obj.pop(k)
                        log.append(f"[{loc}] dropped unknown key '{k}' "
                                   f"(was: {json.dumps(val, ensure_ascii=False)[:80]})")
                    changed = True

            elif e.validator == "type" and isinstance(obj, dict) and "value" in obj and path:
                types = e.validator_value if isinstance(e.validator_value, list) else [e.validator_value]
                if any(t in types for t in ("number", "integer", "string")):
                    parent: Any = record
                    for p in path[:-1]:
                        parent = parent[p]
                    parent[path[-1]] = obj.get("value")
                    log.append(f"[{loc}] flattened {{'value': …}} wrapper to bare scalar")
                    changed = True

            elif e.validator == "required" and isinstance(obj, dict):
                m = re.match(r"'(.+?)' is a required property", e.message)
                if m and m.group(1) not in obj:
                    name = m.group(1)
                    ps = (e.schema.get("properties") or {}).get(name) or {}
                    ptypes = ps.get("type")
                    ptypes = ptypes if isinstance(ptypes, list) else [ptypes]
                    if "null" in ptypes or ps == {}:
                        obj[name] = None
                        log.append(f"[{loc}] added missing required '{name}' = null")
                        changed = True
        if not changed:
            break
    return log


def validate_record(record: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate a merged KMDS record against the full schema. Returns a list of
    human-readable violation strings (empty = valid). No-op if jsonschema is
    unavailable."""
    try:
        from jsonschema import Draft202012Validator
    except Exception:
        return []
    try:
        v = Draft202012Validator(schema)
        msgs = []
        for e in sorted(v.iter_errors(record), key=lambda e: list(e.path)):
            path = "/".join(str(p) for p in e.path) or "(root)"
            msgs.append(f"[{path}] {e.message}")
        return msgs
    except Exception as e:
        return [f"(validator error: {type(e).__name__}: {e})"]


# ===================================================================
# Helpers
# ===================================================================

def _encode_pdf_b64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _parse_json_block(text: str) -> Optional[Any]:
    """Pull the first JSON object out of a model response (fenced or bare)."""
    fence = re.search(r"```(?:json[a-z_]*)?\s*\n([\s\S]*?)```", text)
    blob = fence.group(1) if fence else text
    i = blob.find("{")
    j = blob.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        return json.loads(blob[i:j + 1])
    except json.JSONDecodeError:
        return None


def _usage(resp) -> Dict[str, int]:
    u = resp.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


# ===================================================================
# One focused section call
# ===================================================================

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


async def _gemini_generate(content: List[Dict[str, Any]], model: str,
                           max_tokens: int) -> Dict[str, Any]:
    """Send the (Anthropic-block-shaped) content to Gemini's REST API.

    Lets any `model` starting with "gemini" run through the same pipeline —
    e.g. the free-tier gemini-3.5-flash for zero-cost extraction. Uses raw
    httpx (already a dependency of the anthropic SDK), converts text /
    image / PDF blocks to Gemini parts, drops cache_control (Gemini caches
    implicitly, and free-tier tokens cost nothing), and honors the server's
    retryDelay on 429 — free-tier TPM limits make that routine, not an error.
    Gemini 3.x spends "thinking" tokens from the same output budget, so the
    cap gets generous headroom. Returns {"text", <usage fields>}.
    """
    import httpx
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    parts: List[Dict[str, Any]] = []
    for b in content:
        if b["type"] == "text":
            parts.append({"text": b["text"]})
        elif b["type"] == "image":
            parts.append({"inline_data": {"mime_type": b["source"]["media_type"],
                                          "data": b["source"]["data"]}})
        elif b["type"] == "document":
            parts.append({"inline_data": {"mime_type": "application/pdf",
                                          "data": b["source"]["data"]}})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": min(65536, max_tokens + 24000)}}
    async with httpx.AsyncClient(timeout=600) as hc:
        for _ in range(5):
            r = await hc.post(GEMINI_URL.format(model=model), params={"key": key}, json=body)
            if r.status_code == 429:
                delay = 30.0
                try:
                    for det in r.json()["error"].get("details", []):
                        if "RetryInfo" in det.get("@type", ""):
                            delay = float(det["retryDelay"].rstrip("s")) + 1
                except Exception:  # noqa: BLE001 — malformed error body
                    pass
                print(f"   ⏳ Gemini rate limit — retrying in {min(delay, 90):.0f}s")
                await asyncio.sleep(min(delay, 90))
                continue
            r.raise_for_status()
            d = r.json()
            cand = (d.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "")
                           for p in (cand.get("content") or {}).get("parts", []))
            um = d.get("usageMetadata", {})
            return {"text": text,
                    "stop_reason": cand.get("finishReason"),
                    "input_tokens": um.get("promptTokenCount", 0),
                    "output_tokens": (um.get("candidatesTokenCount", 0)
                                      + um.get("thoughtsTokenCount", 0)),
                    "cache_read_tokens": um.get("cachedContentTokenCount", 0),
                    "cache_creation_tokens": 0}
        raise RuntimeError("Gemini: still rate-limited after 5 retries")


# ===================================================================
# Local backend (KMDS Foundry step 1): the lab model server via Ollama's
# NATIVE API. Closed-access papers never leave the network — the paper
# reaches the model as MinerU markdown, never as a PDF.
# Empirically verified against Ollama 0.20.4: think:false works only on
# /api/chat; glm ignores grammar-constrained "format" (qwen enforces it),
# so structure is validated client-side by the existing parse/repair loop.
# ===================================================================

LOCAL_PREFIX = "local:"
# context CAP (prompt+output) for local calls; per-call num_ctx is computed
# from the actual prompt size and only capped here. 49152 = glm's own pin.
LOCAL_NUM_CTX = int(os.environ.get("ALD_LOCAL_KMDS_NUM_CTX", "49152"))
_CHARS_PER_TOKEN = 3.2          # conservative for scientific English + JSON


def _local_text_model() -> str:
    """env override → the model configured in Settings → glm default."""
    m = os.environ.get("ALD_LOCAL_KMDS_TEXT_MODEL")
    if m:
        return m
    try:
        from llm_backend import _cfg
        m = _cfg("ALD_LOCAL_LLM_MODEL", "local_llm_model")
    except Exception:  # noqa: BLE001
        m = ""
    return m or "glm-4.7-flash-48k:latest"


def _local_vision_model() -> str:
    return os.environ.get("ALD_LOCAL_KMDS_VISION_MODEL", "qwen2.5vl:32b")


def _is_local(model: str) -> bool:
    return (model or "").startswith(LOCAL_PREFIX)


def local_kmds_available() -> bool:
    """A local model server is configured (llm_backend settings/env)."""
    try:
        from llm_backend import local_configured
        return local_configured()
    except Exception:  # noqa: BLE001
        return False


def kmds_backend_available() -> bool:
    return ANTHROPIC_AVAILABLE or local_kmds_available()


def default_model() -> str:
    """Honor the user's backend choice: an explicit llm_backend="local"
    pin wins even when an Anthropic key exists (closed-access workflows
    depend on this); otherwise Claude when a key is available, else the
    local server when configured."""
    try:
        from llm_backend import resolve_backend
        pinned_local = resolve_backend() == "local"
    except Exception:  # noqa: BLE001
        pinned_local = False
    if pinned_local and local_kmds_available():
        return LOCAL_PREFIX + _local_text_model()
    if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        return MODEL
    if local_kmds_available():
        return LOCAL_PREFIX + _local_text_model()
    return MODEL


def _compact_schema(fragment: Optional[Dict[str, Any]]
                    ) -> Optional[Dict[str, Any]]:
    """Shrink a section schema for small-context local models: drop the
    prose keys ('description', 'examples', ...) that carry most of the
    bytes while keeping structure, field names, types, and enums — the
    parts that actually constrain the output."""
    if fragment is None:
        return None
    _DROP = {"description", "examples", "$comment", "markdownDescription"}

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in _DROP}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(fragment)


def _local_native_url() -> str:
    from llm_backend import local_url
    base = local_url()
    return base[:-3] if base.endswith("/v1") else base


def _split_blocks(content: List[Dict[str, Any]]):
    text_parts: List[str] = []
    images: List[str] = []
    for b in content:
        if b["type"] == "text":
            text_parts.append(b["text"])
        elif b["type"] == "image":
            images.append(b["source"]["data"])
        elif b["type"] == "document":
            raise RuntimeError("local model cannot read PDFs — MinerU "
                               "markdown is required for the local path")
    return text_parts, images


def _est_tokens(content: List[Dict[str, Any]]) -> int:
    """Rough prompt-token estimate: chars/3.2 + ~1500/image."""
    chars = sum(len(b.get("text", "")) for b in content if b["type"] == "text")
    n_img = sum(1 for b in content if b["type"] == "image")
    return int(chars / _CHARS_PER_TOKEN) + 1500 * n_img


async def _local_generate(content: List[Dict[str, Any]], model: str,
                          max_tokens: int,
                          num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """Send Anthropic-block-shaped content to the local model server.

    model arrives WITHOUT the "local:" prefix. Prefers Ollama's NATIVE
    /api/chat (think:false works there; /v1 needs reasoning_effort). A
    404/405 means an OpenAI-compatible server (vLLM/LM Studio) — fall
    back to /v1/chat/completions. Vision support is discovered, not
    guessed: images are sent, and a 400 blaming them retries without
    (with a warning) — so any vision model name works.
    """
    import httpx
    text_parts, images = _split_blocks(content)
    msg: Dict[str, Any] = {"role": "user", "content": "\n\n".join(text_parts)}
    if images:
        msg["images"] = images
    payload: Dict[str, Any] = {
        "model": model, "messages": [msg], "stream": False, "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0,
                    "num_ctx": num_ctx or LOCAL_NUM_CTX},
    }
    url = _local_native_url() + "/api/chat"
    async with httpx.AsyncClient(timeout=3600) as hc:
        r = await hc.post(url, json=payload)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think", None)      # model has no thinking switch
            r = await hc.post(url, json=payload)
        if r.status_code == 400 and images and (
                "image" in r.text.lower() or "vision" in r.text.lower()
                or "multimodal" in r.text.lower()):
            print(f"   ⚠ {model} rejected images — retrying text-only "
                  f"({len(images)} crops dropped)")
            payload["messages"][0].pop("images", None)
            r = await hc.post(url, json=payload)
        if r.status_code in (404, 405):     # OpenAI-compatible server
            return await _local_generate_openai(hc, text_parts, images,
                                               model, max_tokens)
        r.raise_for_status()
        d = r.json()
    return {"text": (d.get("message") or {}).get("content") or "",
            "stop_reason": d.get("done_reason"),
            "prompt_eval": d.get("prompt_eval_count", 0),
            "input_tokens": d.get("prompt_eval_count", 0),
            "output_tokens": d.get("eval_count", 0),
            "cache_read_tokens": 0, "cache_creation_tokens": 0}


async def _local_generate_openai(hc, text_parts: List[str],
                                 images: List[str], model: str,
                                 max_tokens: int) -> Dict[str, Any]:
    """Fallback for OpenAI-compatible local servers (vLLM, LM Studio)."""
    from llm_backend import local_url
    parts: List[Dict[str, Any]] = [
        {"type": "text", "text": "\n\n".join(text_parts)}]
    parts += [{"type": "image_url",
               "image_url": {"url": f"data:image/png;base64,{im}"}}
              for im in images]
    r = await hc.post(local_url() + "/chat/completions", json={
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": parts}]})
    r.raise_for_status()
    d = r.json()
    m = (d.get("choices") or [{}])[0].get("message") or {}
    text = m.get("content") or m.get("reasoning") \
        or m.get("reasoning_content") or ""
    u = d.get("usage") or {}
    return {"text": text, "stop_reason": None,
            "prompt_eval": u.get("prompt_tokens", 0),
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "cache_read_tokens": 0, "cache_creation_tokens": 0}


def _confidence_report(record: Dict[str, Any], paper_text: str
                       ) -> Dict[str, Any]:
    """Groundedness audit: which extracted strings literally occur in the
    paper's markdown? Purely deterministic — no model call. Fields that
    don't appear verbatim aren't necessarily wrong (units get normalized,
    tables reflow), but a LOW ratio flags a hallucinating extraction and
    tells the curator where to look first."""
    def _norm(t: str) -> str:
        # unify the unicode variants PDFs and markdown disagree on
        t = t.translate(str.maketrans({"\u2212": "-", "\u2013": "-",
                                       "\u2014": "-", "\u2011": "-",
                                       "\u2019": "'", "\u2018": "'",
                                       "\u201c": '"', "\u201d": '"'}))
        t = t.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\u00ad", "")
        return re.sub(r"\s+", " ", t.lower())

    hay = _norm(paper_text)
    checked: List[tuple] = []
    # fields that are paraphrase BY DESIGN (summaries, descriptions) can
    # never match verbatim — auditing them measures paraphrasing, not
    # hallucination (calibrated: Claude's gold record scores ~18% on them)
    _PARAPHRASE = ("summary", "description", "structure", "comment",
                   "conclusion", "overview", "note", "data name")

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            s = node.strip()
            if any(w in path.lower() for w in _PARAPHRASE):
                return
            # only prose-like values are checkable: long enough to be
            # non-accidental, and not ids/enums/units
            if len(s) >= 12 and sum(c.isalpha() for c in s) >= 8:
                needle = _norm(s)
                checked.append((path, s[:80], needle in hay))

    walk(record, "$")
    unverified = [{"path": p, "value": v} for p, v, ok in checked if not ok]
    return {"checked_fields": len(checked),
            "verified_fields": len(checked) - len(unverified),
            "grounded_ratio": (round(1 - len(unverified) / len(checked), 3)
                               if checked else None),
            "unverified": unverified[:80]}


_AGENDA_PROMPT = """You are indexing a materials-science paper. From the paper text below, list EVERY distinct entity of two kinds:

1. MATERIALS — substances/compounds/phases studied or used (e.g. 'Fe3Al2Si3 (τ1 phase)', 'ε-FeSi', 'LiFePO4'). Chemical identity, not specimens.
2. SAMPLES — the physical specimens that were made/measured. Distinct synthesis routes, dopings, or types are DIFFERENT samples (e.g. 'n-type, process (A)' and 'n-type, process (B)' are two). Use the paper's own naming.

Be exhaustive but do not invent: every entry must be traceable to the text. Papers typically have 1-8 materials and 1-10 samples.

Output ONLY this JSON in a ```json code block:
{"materials": [{"name": "..."}], "samples": [{"name": "...", "description": "<=15 words"}]}

## PAPER
"""


async def _local_agenda(paper_text: str, model: str) -> Optional[Dict[str, Any]]:
    """Foundry agenda pass: one narrow local call enumerating the paper's
    samples + materials. Small models miss entities when asked to fill a
    whole schema, but are near-ceiling on 'list what exists' — the result
    is injected into the core extraction as an authoritative candidate
    list. Returns None on any failure (extraction proceeds without it)."""
    try:
        g = await _local_generate(
            [{"type": "text", "text": _AGENDA_PROMPT + paper_text}],
            model, 1500)
        agenda = _parse_json_block(g["text"])
        if not isinstance(agenda, dict):
            return None
        mats = [m for m in (agenda.get("materials") or [])
                if isinstance(m, dict) and (m.get("name") or "").strip()]
        smps = [s for s in (agenda.get("samples") or [])
                if isinstance(s, dict) and (s.get("name") or "").strip()]
        if not (mats or smps):
            return None
        return {"materials": mats[:12], "samples": smps[:14],
                "tokens": g["output_tokens"]}
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠ agenda pass failed ({type(e).__name__}: {e}) — "
              f"continuing without it")
        return None


def _agenda_block(agenda: Dict[str, Any]) -> str:
    mats = "\n".join(f"  - {m['name']}" for m in agenda["materials"])
    smps = "\n".join(f"  - {s['name']}"
                     + (f" — {s['description']}" if s.get("description") else "")
                     for s in agenda["samples"])
    return (
        "## VERIFIED ENTITY LIST (authoritative — from a dedicated index pass)\n"
        "This paper contains exactly these entities. Your output MUST include "
        "every one of them (and no invented extras):\n"
        f"MATERIALS ({len(agenda['materials'])}):\n{mats}\n"
        f"SAMPLES ({len(agenda['samples'])}):\n{smps}\n\n"
    )


def _finalize_record(record: Dict[str, Any]) -> List[str]:
    """Deterministic clerical completion — the violation classes a model
    (especially a local one) fumbles but code fixes perfectly: required
    KMDS bookkeeping fields, sample ids, and mechanical type coercions.
    Fills ONLY what is missing; a Claude record passes through untouched."""
    log: List[str] = []
    if not isinstance(record, dict):
        return log
    meta = record.get("metadata")
    if not isinstance(meta, dict):
        return log
    pub = meta.get("publication")
    pub = pub if isinstance(pub, dict) else {}
    title = pub.get("title") if isinstance(pub.get("title"), str) else ""

    def fill(key, value):
        if not meta.get(key):
            meta[key] = value
            log.append(f"metadata.{key} filled")

    fill("data name", (title[:80] or "KMDS record"))
    fill("data classification", ["EB0103"])
    dc = meta.get("data classification")
    if isinstance(dc, list):
        codes = [c for c in dc if isinstance(c, str)
                 and re.fullmatch(r"[A-Z]{2}\d{4}", c)]
        topics = [c for c in dc if isinstance(c, str) and c not in codes]
        if topics:
            meta["data classification"] = codes or ["EB0103"]
            kws = meta.setdefault("keywords", [])
            if isinstance(kws, list):
                kws.extend(t for t in topics if t not in kws)
            log.append(f"{len(topics)} free-text classification(s) moved "
                       f"to keywords")
    fill("data generation date", time.strftime("%Y-%m-%d"))
    doi = pub.get("DOI") if isinstance(pub.get("DOI"), str) else ""
    fill("data source", f"extracted from DOI {doi}" if doi
         else "AutoLineDigitizer extraction")
    if not isinstance(meta.get("contributor"), dict) or not meta["contributor"]:
        meta["contributor"] = {"name": "AutoLineDigitizer (local extraction)",
                               "affiliation": "-", "email address": "-"}
        log.append("metadata.contributor filled")
    if not meta.get("keywords"):
        kws = pub.get("author keywords")
        if not (isinstance(kws, list) and kws):
            kws = [w.strip(",:;()").lower() for w in title.split()
                   if len(w) > 5][:5]
        if kws:
            meta["keywords"] = kws
            log.append("metadata.keywords filled from "
                       + ("author keywords" if pub.get("author keywords")
                          else "title"))

    # scope + type: snap free text onto the schema's enums; move what
    # cannot be coerced honestly into keywords rather than inventing codes
    scope = pub.get("scope")
    if isinstance(scope, dict):
        cls = scope.get("classifications")
        if isinstance(cls, list):
            loose = [c for c in cls if isinstance(c, str)]
            if loose:
                scope["classifications"] = [c for c in cls
                                            if not isinstance(c, str)]
                kws = meta.setdefault("keywords", [])
                if isinstance(kws, list):
                    kws.extend(t for t in loose if t not in kws)
                log.append(f"{len(loose)} loose classification(s) -> keywords")
        cls2 = scope.get("classifications")
        if isinstance(cls2, list):
            keep = []
            moved = 0
            for c in cls2:
                if isinstance(c, dict) and c.get("scheme") and c.get("code"):
                    keep.append(c)
                else:
                    label = (c.get("name") if isinstance(c, dict)
                             else c if isinstance(c, str) else "")
                    if label:
                        kws = meta.setdefault("keywords", [])
                        if isinstance(kws, list) and label not in kws:
                            kws.append(label)
                    moved += 1
            if moved:
                scope["classifications"] = keep
                log.append(f"{moved} schemeless classification(s) -> keywords")
        for lk in ("approaches", "purposes", "perspectives"):
            aps = scope.get(lk)
            if isinstance(aps, list):
                fixed = [a.get("text") if isinstance(a, dict) and a.get("text")
                         else a for a in aps]
                if fixed != aps:
                    scope[lk] = [a for a in fixed if isinstance(a, str)]
                    log.append(f"{lk} objects flattened to text")
        pars = scope.get("paradigms")
        if isinstance(pars, list):
            allowed = {"experiment", "theory", "simulation", "data-driven"}
            kept = [x for x in pars if isinstance(x, str) and x in allowed]
            if kept != pars:
                scope["paradigms"] = kept or ["experiment"]
                log.append("paradigms snapped to schema enum")
    t = pub.get("type")
    if isinstance(t, str):
        enum = ["original research", "review", "commentary", "preprint",
                "supplementary materials", "other"]
        tl = t.strip().lower()
        if tl not in enum:
            snap = next((e for e in enum if e.split()[0] in tl), "other")
            pub["type"] = snap
            log.append(f"publication.type {t!r} snapped to {snap!r}")

    # mechanical shape coercions
    y = pub.get("year")
    if isinstance(y, str):
        m = re.search(r"(19|20)\d{2}", y)
        if m:
            pub["year"] = int(m.group(0))
            log.append("publication.year coerced to integer")
    j = pub.get("journal")
    if isinstance(j, str) and j.strip():
        pub["journal"] = {"name": j.strip()}
        log.append("publication.journal wrapped into object")
    authors = pub.get("authors")
    if isinstance(authors, list):
        cleaned = [a for a in authors if a and a != {}]
        if len(cleaned) != len(authors):
            log.append(f"{len(authors) - len(cleaned)} empty author "
                       f"stub(s) dropped")
            authors = pub["authors"] = cleaned
        for i, a in enumerate(authors):
            if isinstance(a, str):
                a = authors[i] = {"given name": " ".join(a.split()[:-1]),
                                  "family name": (a.split() or [""])[-1]}
                log.append(f"authors[{i}] split into given/family")
            elif isinstance(a, dict) and not a.get("family name"):
                full = a.get("name") or a.get("given name") or ""
                if isinstance(full, str) and full.strip():
                    parts = full.split()
                    a["family name"] = parts[-1]
                    a["given name"] = " ".join(parts[:-1]) or a.get(
                        "given name") or ""
                    a.pop("name", None)
                    log.append(f"authors[{i}] family name derived")
    smps = pub.get("samples")
    if isinstance(smps, list):
        taken = {s.get("sample local id") for s in smps
                 if isinstance(s, dict)
                 and isinstance(s.get("sample local id"), str)}
        n = 0
        for s in smps:
            if not isinstance(s, dict):
                continue
            sid_v = s.get("sample local id")
            if sid_v is not None and not isinstance(sid_v, str):
                s["sample local id"] = None       # model emitted an object
                sid_v = None
            if (isinstance(sid_v, str) and sid_v
                    and not re.fullmatch(r"s-\d{3}", sid_v)):
                # the model wrote the NAME into the id slot
                if not s.get("name"):
                    s["name"] = sid_v
                    log.append(f"sample name recovered from id slot: "
                               f"{sid_v[:40]!r}")
                s["sample local id"] = None
                taken.discard(sid_v)
        for s in smps:
            if isinstance(s, dict) and not s.get("sample local id"):
                n += 1
                sid = f"s-{n:03d}"
                while sid in taken:
                    n += 1
                    sid = f"s-{n:03d}"
                s["sample local id"] = sid
                taken.add(sid)
                log.append(f"sample id {sid} assigned")
    return log


_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>()\[\]{},;]+)")


def _backfill_bibliography(record: Dict[str, Any], paper_text: str) -> List[str]:
    """Deterministic Stage-1 mining: DOI and title are IN the markdown —
    never leave them to the model. Fills metadata.publication.DOI (regex)
    and .title (first markdown heading) only where the extraction left
    them empty. Returns a log of what was filled."""
    log: List[str] = []
    if not isinstance(record, dict):
        return log
    meta = record.setdefault("metadata", {})
    if not isinstance(meta, dict):        # model emitted a scalar — replace
        meta = record["metadata"] = {}
    pub = meta.setdefault("publication", {})
    if not isinstance(pub, dict):
        pub = meta["publication"] = {}

    def _empty(v) -> bool:
        return not (isinstance(v, str) and v.strip())

    if _empty(pub.get("DOI")):
        m = _DOI_RE.search(paper_text)
        if m:
            doi = m.group(1).rstrip(".")
            pub["DOI"] = doi
            log.append(f"DOI <- {doi} (regex from paper text)")
    if _empty(pub.get("title")):
        jname = ""
        j = pub.get("journal")
        if isinstance(j, dict):
            jname = (j.get("name") or "").lower()
        elif isinstance(j, str):
            jname = j.lower()
        best = ""
        for line in paper_text.splitlines()[:120]:
            t = line.strip()
            if not t.startswith("#"):
                continue
            t = t.lstrip("# ").strip()
            if len(t) < 15 or (jname and t.lower() == jname):
                continue
            if len(t.split()) >= 6:          # real titles are wordy
                best = t
                break
            best = best or t
        if best:
            pub["title"] = best
            log.append(f"title <- {best[:60]!r} (markdown heading)")
    return log


async def extract_one_section(pdf_b64: str, section_key: str, client,
                              blocks: Dict[str, str],
                              section_schema: Optional[Dict[str, Any]] = None,
                              model: str = MODEL,
                              context_digest: Optional[str] = None,
                              paper_text: Optional[str] = None,
                              paper_figures: Optional[List[Dict[str, Any]]] = None,
                              extra_context: Optional[str] = None) -> Dict[str, Any]:
    """One focused Claude call for a single KMDS section. Never raises.

    paper_text: MinerU-extracted markdown of the paper. When given, this call
    sends the markdown INSTEAD of the PDF (cheaper, reading-order-clean).
    paper_figures: MinerU figure/table crops ({"index","page","label","png"}).
    data_sources needs to SEE figures — it gets markdown + labeled crops when
    available, else the raw PDF.
    """
    t0 = time.time()

    # ---- Cost-aware content layout ----------------------------------------
    # The paper-independent instruction (ground rules + schema fragment, the
    # BULK of the input at ~15-80k tokens/section) goes FIRST with a 1h cache:
    # its prefix is byte-identical across papers, so extraction N+1 within the
    # hour READS it (~0.1x) instead of re-WRITING it (~2x). The paper goes
    # after it with the default 5-minute cache (phase 2 fires seconds after
    # phase 1). Per-paper text (namespace digest, crop notes) stays uncached
    # at the very end so it can never invalidate the stable prefix.
    extras = "\n\n".join(blocks[name] for name in _SECTION_EXTRA[section_key] if blocks.get(name))
    static_instruction = (
        blocks["ground_rules"]                       # verbatim universal rules, at the top
        + "\n\n" + blocks["overview"]                 # verbatim schema overview
        + (("\n\n" + extras) if extras else "")       # section-relevant rule blocks
        + "\n\n" + SUB_PROMPTS[section_key]           # section-specific instruction
    )
    if _is_local(model):
        # small-context models: strip schema prose (descriptions/examples)
        # — structure, names, types, and enums survive
        section_schema = _compact_schema(section_schema)
    static_no_schema = static_instruction
    if section_schema is not None:
        schema_json = json.dumps(section_schema, ensure_ascii=False)
        static_instruction += (
            "\n\n## OUTPUT SCHEMA — authoritative\n"
            "Your JSON for THIS section MUST conform EXACTLY to the JSON Schema below.\n"
            "- Use its EXACT field names, value TYPES, and enum/pattern values. A field "
            "typed `number` takes a bare number (e.g. 350), NOT a `{value, unit}` object — "
            "only use an object where the schema defines an object.\n"
            "- `additionalProperties` is false everywhere — NEVER add a key that is not in "
            "the schema. If the paper states something the schema cannot hold, put it in the "
            "nearest `notes`/`comments` (if the schema has one) — never invent a new key.\n"
            "- Fill what the paper states; use null (or omit optional keys) when it is "
            "silent. Output ONLY the keys this section is responsible for.\n"
            "```json\n" + schema_json + "\n```"
        )

    tail = ""
    if extra_context:
        tail += extra_context
    if context_digest:
        tail += (
            "## Namespace from Phase 1 (authoritative)\n"
            "These ids were assigned by the phase-1 extraction of THIS paper. Use them "
            "EXACTLY when cross-referencing — do not invent, rename, drop, or re-order "
            "ids:\n```json\n" + context_digest + "\n```\n\n"
        )

    crops = paper_figures if (section_key == "data_sources" and paper_figures
                              and 0 < len(paper_figures) <= MAX_FIGURE_CROPS) else None
    if crops and _is_local(model) and not _local_vision_model():
        crops = None                     # vision disabled — captions only
    # local models can never fall back to the raw PDF — markdown always wins
    # (data_sources then runs from captions alone when crops are unusable)
    use_markdown = paper_text is not None and (section_key != "data_sources"
                                               or crops or _is_local(model))

    figure_blocks: List[Dict[str, Any]] = []
    if use_markdown:
        paper_block = {
            "type": "text",
            "text": ("## PAPER (MinerU-extracted markdown, reading order)\n\n" + paper_text),
            "cache_control": {"type": "ephemeral"},   # 5m — reused within this run only
        }
        if crops:
            tail += (
                "## Attached figure/table crops\n"
                "The images above are crops of every figure/chart/table in the paper, in "
                "order. Each crop k corresponds to the `*[FIGURE k …]*` or `*[TABLE crop "
                "k …]*` marker at its position in the markdown — use the markers to match "
                "each image to its caption and page.\n\n"
            )
            for fig in crops:
                figure_blocks.append({"type": "text",
                                      "text": f"FIGURE {fig['index']} ({fig['label']}) — "
                                              f"page {fig['page']}:"})
                figure_blocks.append({"type": "image",
                                      "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.standard_b64encode(fig["png"]).decode()}})
    else:
        paper_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
            "cache_control": {"type": "ephemeral"},   # 5m — reused within this run only
        }

    tail += "Output the JSON in a single ```json code block and nothing else."
    content = [
        {"type": "text", "text": static_instruction,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},  # stable across papers
        paper_block,
        *figure_blocks,
        {"type": "text", "text": tail},
    ]

    base = {"key": section_key, "fragment": None, "raw": None,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0}
    try:
        if model.startswith("gemini"):
            g = await _gemini_generate(content, model,
                                       _SECTION_MAX.get(section_key, SECTION_MAX_TOKENS))
            raw = g.pop("text")
            g.pop("stop_reason", None)
            base.update(g)
        elif _is_local(model):
            mt = _SECTION_MAX.get(section_key, SECTION_MAX_TOKENS)
            if section_key == "data_sources":
                mt = 24000        # caption-rich papers overflow 16k output
            est = _est_tokens(content)
            # the abort decision uses the TEXT-ONLY estimate: crops are
            # dropped automatically when the model rejects them, so images
            # must never make a text call look impossible
            est_text = _est_tokens([b for b in content if b["type"] == "text"])
            if est_text + mt + 512 > LOCAL_NUM_CTX and section_schema is not None:
                # even compacted, the fragment can outgrow the window —
                # drop it rather than let the server truncate silently
                print(f"   ⚠ {section_key}: ~{est//1000}k-token prompt over "
                      f"the {LOCAL_NUM_CTX//1024}k ctx cap — dropping the "
                      f"schema fragment (repair pass normalizes afterwards)")
                content[0] = {**content[0], "text": static_no_schema +
                    "\n\n(No schema fragment fits this context — use exact "
                    "KMDS field names; a schema-repair pass runs on your "
                    "output.)"}
                est = _est_tokens(content)
                est_text = _est_tokens([b for b in content if b["type"] == "text"])
            if est_text + mt + 512 > LOCAL_NUM_CTX:
                raise RuntimeError(
                    f"prompt ~{est_text} tokens exceeds the local context cap "
                    f"{LOCAL_NUM_CTX} — raise ALD_LOCAL_KMDS_NUM_CTX or "
                    f"shorten the paper")
            needed = int(est * 1.15) + mt + 512
            # bucket num_ctx so consecutive calls reuse the loaded model
            # instead of forcing an Ollama reload on every context change
            buckets = [16384, 32768, 49152, 65536, 98304, 131072]
            num_ctx = next((b for b in buckets
                            if b >= needed and b <= LOCAL_NUM_CTX),
                           LOCAL_NUM_CTX)
            g = await _local_generate(content, model[len(LOCAL_PREFIX):],
                                      mt, num_ctx=num_ctx)
            pec = g.pop("prompt_eval", 0)
            # compare against the TEXT-ONLY estimate: image tokens vary
            # wildly per model (and text models skip them entirely), and
            # char-based estimates overshoot — 0.5 is the safe line
            if pec and est_text > 4000 and pec < est_text * 0.5:
                # the server evaluated far fewer tokens than we sent —
                # silent truncation; a "successful" answer would be built
                # on a partial prompt, so fail the section loudly instead
                raise RuntimeError(
                    f"server truncated the prompt (evaluated {pec} of "
                    f"~{est_text} tokens) — raise ALD_LOCAL_KMDS_NUM_CTX")
            raw = g.pop("text")
            g.pop("stop_reason", None)
            base.update(g)
        else:
            resp = await client.messages.create(
                model=model,
                max_tokens=_SECTION_MAX.get(section_key, SECTION_MAX_TOKENS),
                messages=[{"role": "user", "content": content}],
            )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            base.update(_usage(resp))
    except Exception as e:  # noqa: BLE001 — never crash the whole run
        base.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                     "elapsed_sec": time.time() - t0})
        return base

    frag = _parse_json_block(raw)
    base.update({
        "ok": frag is not None,
        "fragment": frag,
        "raw": raw,
        "error": None if frag is not None else "no JSON block parsed from response",
        "elapsed_sec": time.time() - t0,
    })
    return base


# ===================================================================
# Merge section fragments into a single KMDS dict
# ===================================================================

_EMPTY = (None, "", [], {})


def merge_sections(results: List[Dict[str, Any]]) -> (Dict[str, Any], List[str]):
    """Combine section fragments into one KMDS root. First non-null wins on conflict."""
    by_key = {r["key"]: r for r in results if r}
    warnings: List[str] = []
    merged: Dict[str, Any] = {}

    # metadata (+ publication), system, and the materials namespace come from phase 1
    core_frag = (by_key.get(PHASE1_KEY) or {}).get("fragment") or {}
    merged["metadata"] = core_frag.get("metadata") if isinstance(core_frag.get("metadata"), dict) else {}
    if "system" in core_frag:
        merged["system"] = core_frag.get("system")

    # phase-1 output must not carry data-source keys (data_sources owns them)
    pub_seed = merged["metadata"].get("publication")
    if isinstance(pub_seed, dict):
        for k in ("figures", "graphs", "tables", "text passages"):
            if k in pub_seed:
                pub_seed.pop(k)
                warnings.append(f"phase 1 emitted publication.{k} — dropped (data_sources owns it)")

    # materials: seed with the phase-1 namespace (authoritative ids/names/order),
    # then union detail fields from the three materials_* calls by id.
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for m in (core_frag.get("materials") or []):
        if isinstance(m, dict) and m.get("id"):
            by_id[m["id"]] = {"id": m["id"], **({"name": m["name"]} if m.get("name") else {})}
            order.append(m["id"])
    for key in ("materials_chem", "materials_process", "materials_property"):
        frag = (by_key.get(key) or {}).get("fragment") or {}
        for m in (frag.get("materials") or []):
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if not mid:
                continue
            if mid not in by_id:
                if order:  # a namespace exists and this id is outside it
                    warnings.append(f"{key}: id {mid} not in phase-1 namespace — added anyway")
                by_id[mid] = {"id": mid}
                order.append(mid)
            tgt = by_id[mid]
            for field, val in m.items():
                if field == "id":
                    continue
                cur = tgt.get(field)
                if field not in tgt or cur in _EMPTY:
                    tgt[field] = val
                elif cur != val and val not in _EMPTY:
                    warnings.append(
                        f"material {mid}.{field}: sections disagree — keeping first "
                        f"non-null value (from an earlier section)")
    merged["materials"] = [by_id[mid] for mid in order]

    # figures / tables / text passages → metadata.publication.*
    ds_frag = (by_key.get("data_sources") or {}).get("fragment") or {}
    if ds_frag:
        if not isinstance(merged.get("metadata"), dict):
            merged["metadata"] = {}
        pub_obj = merged["metadata"].setdefault("publication", {})
        if isinstance(pub_obj, dict):
            for k in ("figures", "tables", "text passages"):
                if ds_frag.get(k):
                    pub_obj[k] = ds_frag[k]
            # publication has NO `graphs` key — graphs live inside figures[].graphs.
            # If the model emitted a top-level graphs array anyway, never propagate it.
            if ds_frag.get("graphs"):
                warnings.append(
                    f"data_sources emitted a top-level graphs array "
                    f"({len(ds_frag['graphs'])}) — dropped; graphs must nest inside figures")

    return merged, warnings


def _namespace_digest(core_fragment: Optional[Dict[str, Any]]) -> Optional[str]:
    """Compact JSON digest of the phase-1 id namespaces, handed to phase-2 calls."""
    frag = core_fragment or {}
    pub = ((frag.get("metadata") or {}).get("publication") or {})
    samples = [{"sample local id": s.get("sample local id"), "name": s.get("name")}
               for s in (pub.get("samples") or [])
               if isinstance(s, dict) and s.get("sample local id")]
    mats = [{"id": m.get("id"), "name": m.get("name")}
            for m in (frag.get("materials") or [])
            if isinstance(m, dict) and m.get("id")]
    if not samples and not mats:
        return None
    return json.dumps({"samples": samples, "materials": mats},
                      ensure_ascii=False, indent=1)


# ===================================================================
# Translation pass (EN → JA)
# ===================================================================

async def translate_kmds(en_dict: Dict[str, Any], client,
                         translation_rules: str,
                         model: str = TRANSLATION_MODEL) -> Dict[str, Any]:
    """One LLM call: translate natural-language values to Japanese. Never raises.

    model may be an Anthropic id (streamed, with a Sonnet fallback) or a
    gemini one (free tier, client unused — pass None)."""
    t0 = time.time()
    en_json = json.dumps(en_dict, ensure_ascii=False, indent=2)
    instruction = (
        "You are translating a COMPLETED KMDS JSON record from English to Japanese. "
        "This is a translation, not a re-extraction — the two files MUST be structurally "
        "identical. Apply these rules from the extraction specification VERBATIM:\n\n"
        + translation_rules
        + "\n\nReturn ONLY the Japanese JSON in a single ```json code block — same keys, "
          "enums, numbers, units, sample/material ids, formulas, and identifiers "
          "byte-for-byte; translate only the natural-language free-text values listed "
          "above; a field null in English stays null.\n\nEnglish KMDS JSON:\n```json\n"
        + en_json + "\n```"
    )

    out = {"ja": None, "raw": None, "input_tokens": 0, "output_tokens": 0,
           "cache_read_tokens": 0, "cache_creation_tokens": 0}

    async def _stream(model_id):
        # 32k max_tokens exceeds the SDK's 10-minute non-streaming limit —
        # stream and accumulate instead.
        async with client.messages.stream(
            model=model_id,
            max_tokens=TRANSLATION_MAX_TOKENS,
            messages=[{"role": "user", "content": [{"type": "text", "text": instruction}]}],
        ) as s:
            return await s.get_final_message()

    try:
        if model.startswith("gemini"):
            g = await _gemini_generate([{"type": "text", "text": instruction}],
                                       model, TRANSLATION_MAX_TOKENS)
            raw = g.pop("text")
            stop_reason = g.pop("stop_reason", None)
            out.update(g)
        else:
            try:
                resp = await _stream(model)
            except Exception:  # noqa: BLE001
                # Fall back to the extraction model (e.g. no Haiku access).
                resp = await _stream(MODEL)
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            stop_reason = getattr(resp, "stop_reason", None)
            out.update(_usage(resp))
    except Exception as e:  # noqa: BLE001
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "elapsed_sec": time.time() - t0})
        return out

    ja = _parse_json_block(raw)
    err = None
    if ja is None:
        err = ("translation truncated at max_tokens — record too large"
               if stop_reason in ("max_tokens", "MAX_TOKENS")
               else "no JSON block parsed from translation")
    out.update({
        "ok": ja is not None,
        "ja": ja,
        "raw": raw,
        "error": err,
        "elapsed_sec": time.time() - t0,
    })
    return out


async def translate_record_file(en_path: str, ja_path: str,
                                prompt_path: str = "extraction_prompt.md",
                                model: str = TRANSLATION_MODEL) -> Dict[str, Any]:
    """Standalone EN→JA translation of a saved KMDS record (for running in the
    background after the English record is already shown). Never raises."""
    try:
        blocks = load_prompt_blocks(prompt_path)
        with open(en_path, "r", encoding="utf-8") as f:
            en_dict = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "ja_path": None}
    client_cm = (contextlib.nullcontext() if model.startswith("gemini")
                 else AsyncAnthropic())
    async with client_cm as client:
        tr = await translate_kmds(en_dict, client, blocks["translation"], model=model)
    if tr.get("ok"):
        with open(ja_path, "w", encoding="utf-8") as f:
            json.dump(tr["ja"], f, indent=2, ensure_ascii=False)
        tr["ja_path"] = ja_path
    else:
        tr["ja_path"] = None
    return tr


# ===================================================================
# Top-level orchestrator
# ===================================================================

MAX_FIGURE_CROPS = 40  # above this, data_sources falls back to the raw PDF


def _mineru_paper_markdown(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Paper -> LLM-ready markdown + figure crops. Prefers the full MinerU
    CLI when installed (LaTeX formulas, HTML tables, OCR for scans — see
    text_extract.find_mineru_cli); falls back to the bundled lightweight
    layout+text-layer path. Returns None on total unavailability — never raises."""
    try:
        from mineru_layout.text_extract import (pdf_to_markdown, pdf_to_markdown_full,
                                                pdf_to_markdown_remote, find_mineru_cli)
        url = os.environ.get("ALD_MINERU_URL")
        if url:
            res = pdf_to_markdown_remote(pdf_path, url, include_references=True,
                                         return_figures=True)
            if res:
                res["engine"] = "mineru-remote"
                return res
            print("   ⚠ remote MinerU server failed — trying local paths")
        cli = find_mineru_cli()
        if cli:
            res = pdf_to_markdown_full(pdf_path, cli, include_references=True,
                                       return_figures=True)
            if res:
                res["engine"] = "mineru-full"
                return res
            print("   ⚠ full-MinerU CLI failed — trying lightweight extraction")
        from pdf_figures import mineru_available, _load_mineru
        if not mineru_available():
            return None
        res = pdf_to_markdown(pdf_path, detector=_load_mineru(),
                              include_references=True, return_figures=True)
        if res:
            res["engine"] = "lightweight"
        return res
    except Exception as e:  # noqa: BLE001 — markdown is an optimization, not a requirement
        print(f"   ⚠ MinerU text extraction failed ({type(e).__name__}: {e}) — raw-PDF fallback")
        return None


async def extract_kmds_parallel(pdf_path: str, output_dir: str,
                                base_name: Optional[str] = None,
                                prompt_path: str = "extraction_prompt.md",
                                model: str = MODEL,
                                schema_path: Optional[str] = None,
                                use_mineru_text: bool = True,
                                translate: bool = True) -> Dict[str, Any]:
    """Run the two-phase KMDS extraction (+ optional JA translation).

    translate=False skips the JA pass so callers can show the English record
    immediately and run translate_record_file() in the background."""
    is_gemini = model.startswith("gemini")
    is_local = _is_local(model)
    if is_gemini:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            return {"_error": "GEMINI_API_KEY / GOOGLE_API_KEY not set"}
    elif is_local:
        if not local_kmds_available():
            return {"_error": "local model server not configured "
                              "(Settings → Local model server / ALD_LOCAL_LLM_URL)"}
    else:
        if not ANTHROPIC_AVAILABLE:
            return {"_error": "anthropic SDK not installed. pip install anthropic"}
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {"_error": "ANTHROPIC_API_KEY not set"}
    if not os.path.exists(prompt_path):
        return {"_error": f"Prompt file not found: {prompt_path}"}
    if base_name is None:
        base_name = Path(pdf_path).stem

    blocks = load_prompt_blocks(prompt_path)
    if not blocks["ground_rules"]:
        return {"_error": f"Could not extract ground-rules block from {prompt_path}"}
    pdf_b64 = _encode_pdf_b64(pdf_path)

    # Load the real KMDS schema and slice a focused sub-schema per section so the
    # model conforms to actual field names / types / enums (not a skeleton).
    if schema_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(here, "kmds_v15.2.4_nullable.json")
    full_schema = None
    section_schemas: Dict[str, Any] = {}
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            full_schema = json.load(f)
        section_schemas = build_section_schemas(full_schema)
        print(f"⤷ Using KMDS schema: {os.path.basename(schema_path)} "
              f"({len(section_schemas)} section sub-schemas)")
    except Exception as e:
        print(f"⚠ schema not loaded ({e}); falling back to skeleton-free prompts.")

    t0 = time.time()

    # MinerU text extraction: LLM-ready markdown for the text-based calls
    # (core + materials_*). data_sources always keeps the PDF (needs vision).
    paper_text = None
    paper_figures = None
    md_info = _mineru_paper_markdown(pdf_path) if use_mineru_text else None
    if md_info and md_info.get("markdown"):
        paper_text = md_info["markdown"]
        paper_figures = md_info.get("figures") or None
        md_path = os.path.join(output_dir, f"{base_name}_paper.md")
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(paper_text)
        except OSError:
            md_path = None
        print(f"⤷ MinerU [{md_info.get('engine', '?')}]: {md_info['n_pages']} pages → "
              f"{len(paper_text)//1000}k chars markdown "
              f"+ {len(paper_figures or [])} figure/table crops "
              f"({md_info['n_blocks']} text blocks){' -> ' + md_path if md_path else ''}")
        if paper_figures and len(paper_figures) > MAX_FIGURE_CROPS:
            print(f"   ⚠ {len(paper_figures)} crops > {MAX_FIGURE_CROPS} cap — "
                  f"data_sources will use the raw PDF instead")
    elif use_mineru_text:
        print("⤷ MinerU text unavailable (weights missing or scanned PDF) — raw PDF for all calls")

    if is_local and paper_text is None:
        return {"_error": "local KMDS needs MinerU markdown (scanned PDF or "
                          "MinerU unavailable) — the local model cannot read PDFs"}

    # Gemini/local runs need no Anthropic client (nor its API key).
    client_cm = (AsyncAnthropic() if not (is_gemini or is_local)
                 else contextlib.nullcontext())
    async with client_cm as client:
        # FOUNDRY AGENDA (local only): one narrow entity-index call whose
        # result is injected into the core prompt as an authoritative list —
        # small models are near-ceiling on "list what exists" but drop
        # entities when filling a whole schema in one shot.
        agenda = None
        if is_local:
            print("⤷ Foundry agenda: indexing samples + materials (local)...")
            agenda = await _local_agenda(paper_text, model[len(LOCAL_PREFIX):])
            if agenda:
                print(f"   ✓ agenda: {len(agenda['materials'])} materials, "
                      f"{len(agenda['samples'])} samples "
                      f"({agenda['tokens']} tok)")
        agenda_ctx = _agenda_block(agenda) if agenda else None

        # PHASE 1 — one call establishes the record core + id namespaces (samples,
        # materials). The paper block is cached (1h TTL) so phase 2 rides the cache.
        print(f"⤷ KMDS phase 1: core record + id namespaces ({model})...")
        core_res = await extract_one_section(
            pdf_b64, PHASE1_KEY, client, blocks,
            section_schema=section_schemas.get(PHASE1_KEY), model=model,
            paper_text=paper_text, extra_context=agenda_ctx)

        # Corrective retry: if the core still dropped agenda entities, tell
        # it exactly that and take the better of the two attempts.
        if agenda and core_res["ok"]:
            def _counts(frag):
                if not isinstance(frag, dict):
                    return 0, 0
                meta = frag.get("metadata")
                pub = meta.get("publication") if isinstance(meta, dict) else None
                pub = pub if isinstance(pub, dict) else {}
                smps = pub.get("samples")
                mats = frag.get("materials")
                return (len(smps) if isinstance(smps, list) else 0,
                        len(mats) if isinstance(mats, list) else 0)
            s1, m1 = _counts(core_res["fragment"])
            want_s, want_m = len(agenda["samples"]), len(agenda["materials"])
            if s1 < want_s or m1 < want_m:
                print(f"   ↻ core missed entities (samples {s1}/{want_s}, "
                      f"materials {m1}/{want_m}) — corrective retry")
                retry = await extract_one_section(
                    pdf_b64, PHASE1_KEY, client, blocks,
                    section_schema=section_schemas.get(PHASE1_KEY), model=model,
                    paper_text=paper_text,
                    extra_context=(agenda_ctx +
                        "\nYOUR PREVIOUS ATTEMPT MISSED ENTITIES from the "
                        "verified list. Include EVERY listed sample in "
                        "metadata.publication.samples and EVERY listed "
                        "material in materials[].\n"))
                if retry["ok"]:
                    s2, m2 = _counts(retry["fragment"])
                    if s2 + m2 > s1 + m1:
                        for fld in ("input_tokens", "output_tokens"):
                            retry[fld] += core_res[fld]
                        core_res = retry
        results = [core_res]
        mark = "✓" if core_res["ok"] else "✗"
        extra = "" if core_res["ok"] else f" — {core_res['error']}"
        print(f"   {mark} {core_res['key']:<20} out={core_res['output_tokens']:>5} tok  "
              f"({core_res['elapsed_sec']:.1f}s){extra}")

        digest = _namespace_digest(core_res.get("fragment")) if core_res["ok"] else None
        if digest is None:
            print("   ⚠ no phase-1 namespace — phase 2 runs without cross-reference context")

        # PHASE 2 — 4 concurrent detail calls, each carrying the phase-1 namespace.
        # materials_* run on the light model (see SECTION_MODELS); a caller-chosen
        # non-default model overrides the map for every section.
        phase2_keys = [k for k in SUB_PROMPTS if k != PHASE1_KEY]
        if is_local:
            # data_sources needs eyes: route it to the local VISION model when
            # crops will actually attach (same gate as extract_one_section —
            # >MAX_FIGURE_CROPS means no crops, so vision buys nothing)
            usable_crops = bool(paper_figures
                                and 0 < len(paper_figures) <= MAX_FIGURE_CROPS)
            vision = _local_vision_model()
            sec_model = {k: ((LOCAL_PREFIX + vision)
                             if k == "data_sources" and usable_crops and vision
                             else model)
                         for k in phase2_keys}
        else:
            sec_model = {k: (SECTION_MODELS.get(k, model) if model == MODEL else model)
                         for k in phase2_keys}
        n_light = sum(1 for m in sec_model.values() if m != model)
        mode_note = ("sequential — free-tier TPM" if is_gemini
                     else "sequential — one local GPU" if is_local
                     else f"concurrent{f'; {n_light} on Haiku' if n_light else ''}")
        print(f"⤷ KMDS phase 2: firing {len(phase2_keys)} focused section calls "
              f"({mode_note})...")
        p2_coros = (extract_one_section(pdf_b64, key, client, blocks,
                                        section_schema=section_schemas.get(key),
                                        model=sec_model[key],
                                        context_digest=digest, paper_text=paper_text,
                                        paper_figures=paper_figures)
                    for key in phase2_keys)
        if is_gemini or is_local:
            # gemini: 4-wide bursts blow the free-tier TPM cap.
            # local: one GPU — concurrent requests only queue server-side
            # and interleave badly with model swapping.
            p2 = [await c for c in p2_coros]
        else:
            p2 = list(await asyncio.gather(*p2_coros))
        # Local safety net: a section that answered but produced no
        # parseable JSON gets ONE explicit do-over (temp-0 runs still vary
        # on GPUs; the reminder usually lands).
        if is_local:
            for i, r in enumerate(p2):
                if not r["ok"] and "no JSON block" in (r["error"] or ""):
                    print(f"   ↻ {r['key']}: output was not valid JSON — one retry")
                    p2[i] = await extract_one_section(
                        pdf_b64, r["key"], client, blocks,
                        section_schema=section_schemas.get(r["key"]),
                        model=sec_model[r["key"]], context_digest=digest,
                        paper_text=paper_text, paper_figures=paper_figures,
                        extra_context=("REMINDER: your ENTIRE response must be "
                                       "a single ```json code block. No prose, "
                                       "no explanations.\n\n"))
        # Light-model safety net: retry a failed Haiku section once on Sonnet.
        for i, r in enumerate(p2):
            if not r["ok"] and sec_model.get(r["key"], model) != model:
                print(f"   ↻ {r['key']}: failed on light model ({r['error']}) — "
                      f"retrying on {model}")
                p2[i] = await extract_one_section(
                    pdf_b64, r["key"], client, blocks,
                    section_schema=section_schemas.get(r["key"]), model=model,
                    context_digest=digest, paper_text=paper_text,
                    paper_figures=paper_figures)
        results.extend(p2)
        sec_wall = time.time() - t0
        for r in p2:
            mark = "✓" if r["ok"] else "✗"
            extra = "" if r["ok"] else f" — {r['error']}"
            print(f"   {mark} {r['key']:<20} out={r['output_tokens']:>5} tok  "
                  f"cache_read={r['cache_read_tokens']:>6}  ({r['elapsed_sec']:.1f}s){extra}")

        merged, warnings = merge_sections(results)
        for w in warnings:
            print(f"   ⚠ merge: {w}")

        # Deterministic post-passes: schema-guided repair of mechanical model
        # slip-ups, then axes[].quantity.ref from the schema's own vocabulary.
        repair_log: List[str] = []
        if full_schema is not None:
            repair_log = repair_record(merged, full_schema)
            if repair_log:
                print(f"   ✓ schema repair: {len(repair_log)} fixes applied")
            n_refs = fill_axis_refs(merged, full_schema)
            if n_refs:
                print(f"   ✓ filled {n_refs} axes quantity.ref from schema vocabulary")

        # Deterministic bibliography backfill: DOI/title live verbatim in the
        # paper — regex beats any model at this, and small local models miss
        # them more often than Claude does. Runs AFTER repair_record so
        # {'value': ...}-wrapped fields are already flattened to scalars.
        try:
            if paper_text:
                for line in _backfill_bibliography(merged, paper_text):
                    print(f"   ✓ backfill: {line}")
            fin_log = _finalize_record(merged)
            if fin_log:
                print(f"   ✓ clerical completion: {len(fin_log)} fields "
                      f"({', '.join(fin_log[:4])}{'…' if len(fin_log) > 4 else ''})")
        except Exception as e:  # noqa: BLE001 — cosmetics must never kill a run
            import traceback
            print(f"   ⚠ post-processing error (record saved as-is): "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()

        en_path = os.path.join(output_dir, f"{base_name}.json")
        with open(en_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved EN: {en_path}")

        # Validate against the full schema and save a conformance report.
        n_violations = None
        if full_schema is not None:
            viol = validate_record(merged, full_schema)
            n_violations = len(viol)
            val_path = os.path.join(output_dir, f"{base_name}_validation.txt")
            with open(val_path, "w", encoding="utf-8") as f:
                f.write(f"KMDS schema validation — {base_name}\n")
                f.write(f"schema: {os.path.basename(schema_path)}\n")
                f.write(f"violations: {n_violations}\n\n")
                f.write("\n".join(viol) if viol else "VALID — conforms to the schema.")
                if repair_log:
                    f.write("\n\nautomatic schema repairs applied before validation:\n")
                    f.write("\n".join(repair_log))
            mark = "✓ VALID" if n_violations == 0 else f"⚠ {n_violations} violation(s)"
            print(f"   schema validation: {mark}  -> {val_path}")

        # Groundedness audit (deterministic): which extracted prose strings
        # literally occur in the paper? Small local models hallucinate more
        # than Claude — this quantifies it per run and shows the curator
        # where to look. Written for every backend so runs are comparable.
        confidence = None
        if paper_text:
            confidence = _confidence_report(merged, paper_text)
            conf_path = os.path.join(output_dir, f"{base_name}_confidence.json")
            with open(conf_path, "w", encoding="utf-8") as f:
                json.dump(confidence, f, indent=2, ensure_ascii=False)
            if confidence["grounded_ratio"] is not None:
                print(f"   groundedness: {confidence['grounded_ratio']:.0%} of "
                      f"{confidence['checked_fields']} prose fields verbatim "
                      f"in the paper -> {conf_path}")

        # per-section raw dump for debugging
        dbg_path = os.path.join(output_dir, "_parallel_sections_raw.json")
        with open(dbg_path, "w", encoding="utf-8") as f:
            json.dump({r["key"]: {"ok": r["ok"], "error": r["error"],
                                  "fragment": r["fragment"]} for r in results},
                      f, indent=2, ensure_ascii=False)

        # translation pass (sequential, after gather; skippable for fast UI)
        # Local runs skip it: the Haiku translation model needs the Claude API.
        if translate and is_local:
            translate = False
            print("   ⤷ JA translation skipped — needs the Claude API "
                  "(local extraction run)")
        if translate:
            tr_model = model if is_gemini else TRANSLATION_MODEL
            print(f"⤷ Translating EN → JA (1 call, {tr_model})...")
            tr = await translate_kmds(merged, client, blocks["translation"],
                                      model=tr_model)
        else:
            tr = {"ok": None, "skipped": True, "error": None, "ja": None,
                  "input_tokens": 0, "output_tokens": 0,
                  "cache_read_tokens": 0, "cache_creation_tokens": 0}

    ja_path = None
    if tr["ok"]:
        ja_path = os.path.join(output_dir, f"{base_name}_ja.json")
        with open(ja_path, "w", encoding="utf-8") as f:
            json.dump(tr["ja"], f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved JA: {ja_path}")
    elif tr.get("skipped"):
        print("   ⤷ JA translation deferred (run in background by the caller)")
    else:
        print(f"   ⚠ JA translation failed: {tr['error']} (EN saved, continuing)")

    wall = time.time() - t0
    total_in = sum(r["input_tokens"] for r in results) + tr["input_tokens"]
    total_out = sum(r["output_tokens"] for r in results) + tr["output_tokens"]
    total_cr = sum(r["cache_read_tokens"] for r in results) + tr.get("cache_read_tokens", 0)
    total_cc = sum(r["cache_creation_tokens"] for r in results) + tr.get("cache_creation_tokens", 0)

    summary = {
        "mode": "two-phase",
        "model": model,
        "mineru_text": paper_text is not None,
        "n_schema_repairs": len(repair_log),
        "en_path": en_path,
        "ja_path": ja_path,
        "n_sections_ok": sum(1 for r in results if r["ok"]),
        "n_sections": len(results),
        "n_schema_violations": n_violations,
        "confidence": ({"grounded_ratio": confidence["grounded_ratio"],
                        "checked_fields": confidence["checked_fields"],
                        "verified_fields": confidence["verified_fields"]}
                       if confidence else None),
        "sections": {r["key"]: {
            "ok": r["ok"], "error": r["error"],
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
            "cache_read_tokens": r["cache_read_tokens"],
            "cache_creation_tokens": r["cache_creation_tokens"],
            "elapsed_sec": round(r["elapsed_sec"], 2),
        } for r in results},
        "translation": {"ok": tr["ok"], "error": tr.get("error"),
                        "input_tokens": tr["input_tokens"], "output_tokens": tr["output_tokens"]},
        "section_wall_sec": round(sec_wall, 2),
        "elapsed_sec": round(wall, 2),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_read_tokens": total_cr,
        "cache_creation_tokens": total_cc,
        "merge_warnings": warnings,
    }
    print(f"   KMDS parallel done in {wall:.1f}s "
          f"(input {total_in:,} + output {total_out:,} tokens; "
          f"cache_read {total_cr:,}, cache_write {total_cc:,})")
    return summary
