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


MODEL = "claude-sonnet-4-6"          # user-specified: Sonnet 4.6 for all calls
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
        "type, open access status), `scope` (paradigms, purposes, approaches, "
        "conclusions[], classifications, comments), and `samples[]`.\n"
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

async def extract_one_section(pdf_b64: str, section_key: str, client,
                              blocks: Dict[str, str],
                              section_schema: Optional[Dict[str, Any]] = None,
                              model: str = MODEL,
                              context_digest: Optional[str] = None,
                              paper_text: Optional[str] = None) -> Dict[str, Any]:
    """One focused Claude call for a single KMDS section. Never raises.

    paper_text: MinerU-extracted markdown of the paper. When given, this call
    sends the markdown INSTEAD of the PDF (cheaper, reading-order-clean) —
    except data_sources, which always gets the PDF because it must SEE the
    figures to describe axes/colors/markers.
    """
    t0 = time.time()

    extras = "\n\n".join(blocks[name] for name in _SECTION_EXTRA[section_key] if blocks.get(name))
    instruction = (
        blocks["ground_rules"]                       # verbatim universal rules, at the top
        + "\n\n" + blocks["overview"]                 # verbatim schema overview
        + (("\n\n" + extras) if extras else "")       # section-relevant rule blocks
        + "\n\n" + SUB_PROMPTS[section_key]           # section-specific instruction
    )

    if context_digest:
        instruction += (
            "\n\n## Namespace from Phase 1 (authoritative)\n"
            "These ids were assigned by the phase-1 extraction of THIS paper. Use them "
            "EXACTLY when cross-referencing — do not invent, rename, drop, or re-order "
            "ids:\n```json\n" + context_digest + "\n```"
        )

    if section_schema is not None:
        schema_json = json.dumps(section_schema, ensure_ascii=False)
        instruction += (
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
            "```json\n" + schema_json + "\n```\n"
            "Output the JSON in a single ```json code block and nothing else."
        )
    else:
        instruction += "\nOutput the JSON in a single ```json code block and nothing else."

    use_markdown = paper_text is not None and section_key != "data_sources"
    if use_markdown:
        paper_block = {
            "type": "text",
            "text": ("## PAPER (MinerU-extracted markdown, reading order; figure "
                     "images not included — captions are)\n\n" + paper_text),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    else:
        paper_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    content = [
        paper_block,
        {"type": "text", "text": instruction,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]

    base = {"key": section_key, "fragment": None, "raw": None,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0}
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=_SECTION_MAX.get(section_key, SECTION_MAX_TOKENS),
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:  # noqa: BLE001 — never crash the whole run
        base.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                     "elapsed_sec": time.time() - t0})
        return base

    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    frag = _parse_json_block(raw)
    base.update(_usage(resp))
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
                         translation_rules: str) -> Dict[str, Any]:
    """One Claude call: translate natural-language values to Japanese. Never raises."""
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
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=TRANSLATION_MAX_TOKENS,
            messages=[{"role": "user", "content": [{"type": "text", "text": instruction}]}],
        )
    except Exception as e:  # noqa: BLE001
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "elapsed_sec": time.time() - t0})
        return out

    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    ja = _parse_json_block(raw)
    out.update(_usage(resp))
    out.update({
        "ok": ja is not None,
        "ja": ja,
        "raw": raw,
        "error": None if ja is not None else "no JSON block parsed from translation",
        "elapsed_sec": time.time() - t0,
    })
    return out


# ===================================================================
# Top-level orchestrator
# ===================================================================

def _mineru_paper_markdown(pdf_path: str) -> Optional[Dict[str, Any]]:
    """MinerU text extraction (layout + reading order + text layer). Returns the
    pdf_to_markdown result dict, or None on any unavailability — never raises."""
    try:
        from mineru_layout.text_extract import pdf_to_markdown
        from pdf_figures import mineru_available, _load_mineru
        if not mineru_available():
            return None
        return pdf_to_markdown(pdf_path, detector=_load_mineru())
    except Exception as e:  # noqa: BLE001 — markdown is an optimization, not a requirement
        print(f"   ⚠ MinerU text extraction failed ({type(e).__name__}: {e}) — raw-PDF fallback")
        return None


async def extract_kmds_parallel(pdf_path: str, output_dir: str,
                                base_name: Optional[str] = None,
                                prompt_path: str = "extraction_prompt.md",
                                model: str = MODEL,
                                schema_path: Optional[str] = None,
                                use_mineru_text: bool = True) -> Dict[str, Any]:
    """Run the two-phase KMDS extraction + translation. Returns a summary dict."""
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
    md_info = _mineru_paper_markdown(pdf_path) if use_mineru_text else None
    if md_info and md_info.get("markdown"):
        paper_text = md_info["markdown"]
        md_path = os.path.join(output_dir, f"{base_name}_paper.md")
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(paper_text)
        except OSError:
            md_path = None
        print(f"⤷ MinerU text: {md_info['n_pages']} pages → {len(paper_text)//1000}k chars "
              f"({md_info['n_blocks']} blocks, {md_info['n_figures']} figures, "
              f"{md_info['n_tables']} tables){' -> ' + md_path if md_path else ''}")
    elif use_mineru_text:
        print("⤷ MinerU text unavailable (weights missing or scanned PDF) — raw PDF for all calls")

    async with AsyncAnthropic() as client:
        # PHASE 1 — one call establishes the record core + id namespaces (samples,
        # materials). The paper block is cached (1h TTL) so phase 2 rides the cache.
        print(f"⤷ KMDS phase 1: core record + id namespaces ({model})...")
        core_res = await extract_one_section(
            pdf_b64, PHASE1_KEY, client, blocks,
            section_schema=section_schemas.get(PHASE1_KEY), model=model,
            paper_text=paper_text)
        results = [core_res]
        mark = "✓" if core_res["ok"] else "✗"
        extra = "" if core_res["ok"] else f" — {core_res['error']}"
        print(f"   {mark} {core_res['key']:<20} out={core_res['output_tokens']:>5} tok  "
              f"({core_res['elapsed_sec']:.1f}s){extra}")

        digest = _namespace_digest(core_res.get("fragment")) if core_res["ok"] else None
        if digest is None:
            print("   ⚠ no phase-1 namespace — phase 2 runs without cross-reference context")

        # PHASE 2 — 4 concurrent detail calls, each carrying the phase-1 namespace.
        phase2_keys = [k for k in SUB_PROMPTS if k != PHASE1_KEY]
        print(f"⤷ KMDS phase 2: firing {len(phase2_keys)} focused section calls (concurrent)...")
        p2 = await asyncio.gather(
            *(extract_one_section(pdf_b64, key, client, blocks,
                                  section_schema=section_schemas.get(key), model=model,
                                  context_digest=digest, paper_text=paper_text)
              for key in phase2_keys)
        )
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

        # per-section raw dump for debugging
        dbg_path = os.path.join(output_dir, "_parallel_sections_raw.json")
        with open(dbg_path, "w", encoding="utf-8") as f:
            json.dump({r["key"]: {"ok": r["ok"], "error": r["error"],
                                  "fragment": r["fragment"]} for r in results},
                      f, indent=2, ensure_ascii=False)

        # translation pass (sequential, after gather)
        print("⤷ Translating EN → JA (1 call)...")
        tr = await translate_kmds(merged, client, blocks["translation"])

    ja_path = None
    if tr["ok"]:
        ja_path = os.path.join(output_dir, f"{base_name}_ja.json")
        with open(ja_path, "w", encoding="utf-8") as f:
            json.dump(tr["ja"], f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved JA: {ja_path}")
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
