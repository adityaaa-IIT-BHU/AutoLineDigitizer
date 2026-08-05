# -*- coding: utf-8 -*-
"""
ncmrd_interview.py — "ask questions → compile to NCMRD" extraction prototype.

Stage A (interview): ONE small LLM call. The model gets the paper markdown +
figure crops and a compact question form (~2k tokens) — plain questions in
plain shapes, the paper's own words and units. No NCMRD keys, no catalogs, no
schema fragments: the ~1.4MB NCMRD schema never enters the prompt.

Stage B (compile): deterministic code builds the full NCMRD record from the
answers — assigns s-NNN/f-NNN/g-NNN/material_NN ids, wires every
cross-reference from the interview's explicit mentions, resolves property /
process / measurement names against indexes built from the schema itself,
and only ever emits keys that exist in the schema. jsonschema validation +
repair_record stay as the final gate.

NOT wired into the desktop app — prototype entry point is
extract_ncmrd_interview(). The production pipeline stays ncmrd_parallel.
"""

import asyncio
import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ncmrd_parallel import (MODEL, LIGHT_MODEL, ANTHROPIC_AVAILABLE,
                           _mineru_paper_markdown, _parse_json_block, _usage,
                           _build_property_ref_index, repair_record,
                           fill_axis_refs)

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None

INTERVIEW_MAX_TOKENS = 32000  # exhaustive passages need room; >16k requires streaming

# ---------------------------------------------------------------------------
# Stage A — the question form
# ---------------------------------------------------------------------------

INTERVIEW_FORM = """\
{
  "paper": {"title": "", "doi": "", "journal": "", "year": null, "article_number": "",
            "abstract_summary": "2-4 sentence summary",
            "authors": [{"given_name": "", "family_name": "",
                         "affiliations": ["affiliation name, country"]}],
            "keywords": ["5-8 topical keywords"]},
  "reference_dois": ["DOI of every numbered reference that has one, in citation order"],
  "materials": [
    {"mid": "m1", "name": "chemical formula or name as written",
     "composition": [{"constituent": "element or compound", "concentration": 33.3}],
     "composition_unit": "at.% | wt.% | mol",
     "structure": {"crystal_system": "", "space_group": "", "lattice_parameter_nm": null,
                   "phase_name": ""}}
  ],
  "samples": [
    {"sid": "s1", "label": "sample name as used in the paper", "made_of": ["m1"],
     "role": "e.g. thermoelectric material",
     "mixing_type": "stoichiometric compound | doped compound | composite | solution",
     "description": "1-2 sentences: how it was made, what distinguishes it"}
  ],
  "process_steps": [
    {"applies_to": ["m1"], "method": "method name, e.g. ball milling / hot pressing",
     "inputs": ["starting materials fed into this step"],
     "params": [{"name": "e.g. milling time / atmosphere / equipment", "value": 35,
                 "unit": "h (empty string if none / value is text)"}]}
  ],
  "property_values": [
    {"of": "m1", "property": "property name, e.g. band gap energy",
     "value": 0.53, "unit": "eV", "conditions": "e.g. at 300 K",
     "technique": "measurement technique, e.g. infrared spectroscopy",
     "instrument": "instrument model if stated"}
  ],
  "figures": [
    {"fig": "1", "name": "Figure 1", "caption_summary": "1 sentence",
     "n_panels": 4,
     "graphs": [
       {"panel": "a", "what": "electrical conductivity vs temperature",
        "x": {"name": "Temperature", "unit": "K", "ticks": [300, 500, 700, 900],
              "scale": "linear"},
        "y": {"name": "Electrical conductivity", "unit": "10^{4} S m^{-1}",
              "ticks": [0, 10, 20], "scale": "linear"},
        "samples": ["s1", "s2"],
        "description": "2-4 sentences: every series with its marker/color and trend"}
     ]}
  ],
  "tables": [{"label": "1", "caption": "", "samples": ["s1"]}],
  "passages": [
    {"section": "abstract | introduction | results | discussion | methods",
     "name": "short title for this passage",
     "text": "quote of a factual statement worth keeping (1-2 sentences, max 40 words)",
     "samples": ["s1"]}
  ],
  "conclusions": [{"text": "one key finding of the paper", "samples": ["s1"]}],
  "data_name": "short dataset name: main material + property focus"
}"""

INTERVIEW_INSTRUCTION = """\
You are reading ONE materials-science paper (markdown below; every figure is \
attached as a cropped image matching the *[FIGURE k]* markers). Fill in this \
QUESTION FORM about it. Rules:
- Plain language, the paper's OWN words, values and units EXACTLY as printed.
- Do NOT invent information. Use null / [] / "" when the paper is silent.
- ids: give every material an "mid" (m1, m2, ...) and every sample an "sid" \
(s1, s2, ...) and use those ids for EVERY cross-reference (made_of, \
applies_to, of, samples). A material is a distinct composition; a sample is a \
physical specimen studied (usually one per composition).
- Be EXHAUSTIVE on structure: every sample/composition in the paper, every \
graph panel of every figure, every numeric property value stated in the \
text, every reference DOI. Be CONCISE in prose: 12-15 passages max, each a \
short quote; keep the whole answer well under 20,000 tokens.
- Read the attached figure crops to answer the figures questions: name every \
plotted series and which sample it is (legend), and describe trends.

FORM (answer with the same structure, in one ```json code block, nothing else):
""" + INTERVIEW_FORM


async def conduct_interview(paper_text: str, paper_figures: Optional[List[Dict]],
                            client, model: str = LIGHT_MODEL) -> Dict[str, Any]:
    """One LLM call: paper + crops + question form -> plain answers dict."""
    t0 = time.time()
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": INTERVIEW_INSTRUCTION,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},  # identical across papers
        {"type": "text", "text": "## PAPER (MinerU markdown)\n\n" + paper_text,
         "cache_control": {"type": "ephemeral"}},
    ]
    for fig in paper_figures or []:
        content.append({"type": "text",
                        "text": f"FIGURE {fig['index']} ({fig['label']}) — page {fig['page']}:"})
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/png",
                                   "data": base64.standard_b64encode(fig["png"]).decode()}})
    content.append({"type": "text", "text": "Answer the form now, one ```json block."})

    out = {"answers": None, "raw": None, "error": None,
           "input_tokens": 0, "output_tokens": 0,
           "cache_read_tokens": 0, "cache_creation_tokens": 0}
    try:
        # 32k max_tokens exceeds the SDK's non-streaming limit — stream + collect.
        async with client.messages.stream(
                model=model, max_tokens=INTERVIEW_MAX_TOKENS,
                messages=[{"role": "user", "content": content}]) as s:
            resp = await s.get_final_message()
    except Exception as e:  # noqa: BLE001
        out.update({"error": f"{type(e).__name__}: {e}", "elapsed_sec": time.time() - t0})
        return out
    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    out.update(_usage(resp))
    answers = _parse_json_block(raw)
    if answers is None and "{" in raw:  # salvage an unfenced / trailing-text answer
        try:
            answers = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except ValueError:
            pass
    out.update({"answers": answers, "raw": raw, "elapsed_sec": time.time() - t0})
    if out["answers"] is None:
        out["error"] = ("interview truncated at max_tokens"
                        if getattr(resp, "stop_reason", None) == "max_tokens"
                        else "no JSON block parsed from interview response")
    return out


# ---------------------------------------------------------------------------
# Stage B — the compiler
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _match(name: str, index: Dict[str, Any]) -> Optional[Any]:
    """Exact-normalized, then unique-substring match into an index."""
    n = _norm(name)
    if not n:
        return None
    if n in index:
        return index[n]
    hits = [v for k, v in index.items() if n in k or k in n]
    return hits[0] if len(hits) == 1 else None


def _catalog_index(schema: Dict, def_name: str) -> Dict[str, Tuple[str, Dict]]:
    """{normalized method/technique name: (exact key, its param sub-schema)}."""
    node = schema["$defs"][def_name]
    props = (node.get("items") or node).get("properties") or {}
    return {_norm(k): (k, (v.get("properties") or {}))
            for k, v in props.items() if k not in ("M_in", "M_out")}


# plain-language names papers use -> vocabulary terms (extended as papers reveal them)
_PROPERTY_SYNONYMS = {
    "hole concentration": "carrier concentration",
    "electron concentration": "carrier concentration",
    "figure of merit zt": "zt",
    "thermoelectric figure of merit": "zt",
}
_QUALIFIERS = re.compile(r"^(peak|average|maximum|minimum|max|min|room temperature|rt)\s+")


def _property_paths(schema: Dict) -> Dict[str, List[str]]:
    """{normalized property term: path segments below materials[].property}.

    Two shapes exist in the vocabulary: plain leaves
    (['electrical property', 'band gap energy']) and value-array members
    (['thermoelectric property', 'value[]', 'ZT'])."""
    out: Dict[str, List[str]] = {}
    for term, ptr in _build_property_ref_index(schema).items():
        segs = ptr.split("/properties/")
        try:
            i = segs.index("property")
        except ValueError:
            continue
        tail = segs[i + 1:]
        if not tail:
            continue
        path = []
        for s in tail:
            if s.endswith("/items"):
                path.append(s[:-len("/items")] + "[]")
            else:
                path.append(s)
        out[_norm(term)] = path
    return out


def _match_property(name: str, prop_paths: Dict[str, List[str]]) -> Optional[List[str]]:
    n = _QUALIFIERS.sub("", _norm(name))
    n = _PROPERTY_SYNONYMS.get(n, n)
    return _match(n, prop_paths)


def _si_conversion(unit: str) -> str:
    m = re.match(r"^\s*10\^?\{?(-?\d+)\}?", str(unit or ""))
    return f"value * 1e{m.group(1)}" if m else "n/a"


_SCALES = {"linear", "log", "log10", "ln", "reciprocal"}


def _axis(kind: str, ax: Optional[Dict]) -> Dict[str, Any]:
    ax = ax or {}
    comments = []
    ticks = [t for t in (ax.get("ticks") or []) if isinstance(t, (int, float))]
    labels = [str(t) for t in (ax.get("ticks") or []) if not isinstance(t, (int, float))]
    if labels:
        comments.append("categorical ticks: " + ", ".join(labels))
    scale = ax.get("scale") or "linear"
    if scale not in _SCALES:
        comments.append(f"scale: {scale}")
        scale = None
    return {"axis": kind,
            "quantity": {"term": ax.get("name")},
            "unit": ax.get("unit") or "",
            "reference ticks": ticks,
            "scale": scale,
            "SI conversion": _si_conversion(ax.get("unit")),
            "comments": comments}


def compile_interview(iv: Dict[str, Any], schema: Dict[str, Any]
                      ) -> Tuple[Dict[str, Any], List[str]]:
    """Deterministically build a full NCMRD record from interview answers."""
    log: List[str] = []
    prop_paths = _property_paths(schema)
    proc_idx = _catalog_index(schema, "process")
    meas_idx = _catalog_index(schema, "measurement")

    mids = {m.get("mid"): f"material_{i+1:02d}" for i, m in enumerate(iv.get("materials") or [])}
    sids = {s.get("sid"): f"s-{i+1:03d}" for i, s in enumerate(iv.get("samples") or [])}
    mid_name = {m.get("mid"): m.get("name") for m in iv.get("materials") or []}
    # models sometimes cite a MATERIAL where a sample is expected — map it to
    # the sample(s) made of that material.
    mid_to_sids = {}
    for s in iv.get("samples") or []:
        for m in s.get("made_of") or []:
            mid_to_sids.setdefault(m, []).append(sids[s.get("sid")])

    def sample_refs(lst):  # interview sids/mids -> NCMRD sample ids
        out, dropped = [], []
        for x in lst or []:
            if x in sids:
                out.append(sids[x])
            elif x in mid_to_sids:
                out.extend(mid_to_sids[x])
            else:
                dropped.append(x)
        if dropped:
            log.append(f"dropped unknown sample ref(s): {dropped}")
        return list(dict.fromkeys(out))

    # ---- materials --------------------------------------------------------
    materials = []
    for m in iv.get("materials") or []:
        entry: Dict[str, Any] = {"id": mids[m.get("mid")], "name": m.get("name")}
        comp = m.get("composition") or []
        if comp:
            vals = []
            for c in comp:
                conc = c.get("concentration")
                if not isinstance(conc, (int, float)):  # symbolic ('x', '1-x')
                    if conc is not None:
                        log.append(f"symbolic concentration {conc!r} for "
                                   f"{c.get('constituent')!r} in {m.get('name')!r} -> null")
                    conc = None
                vals.append({"constituent": c.get("constituent"), "concentration": conc})
            entry["chemical information"] = {"composition": {
                "value": vals, "unit": m.get("composition_unit") or "at.%"}}
        st = m.get("structure") or {}
        cry = {}
        if st.get("space_group"):
            cry["space group"] = st["space_group"]
        if st.get("crystal_system"):
            cry["crystal system"] = st["crystal_system"]
        if st.get("lattice_parameter_nm") is not None:
            cry["lattice parameter"] = {"a": {"value": st["lattice_parameter_nm"]}}
        struct: Dict[str, Any] = {}
        if cry:
            struct["crystallography"] = cry
        if st.get("phase_name"):
            struct["phase"] = {"data": [{"id": "phase_01", "name": st["phase_name"]}]}
        if struct:
            entry["structure"] = struct
        materials.append(entry)
    mat_by_id = {m["id"]: m for m in materials}

    # ---- process steps (grouped per material) -----------------------------
    expanded_steps = []
    for step in iv.get("process_steps") or []:
        parts = re.split(r"\s+(?:and|then|followed by)\s+|\s*\+\s*",
                         str(step.get("method") or ""))
        for part in [p for p in parts if p.strip()] or [step.get("method")]:
            expanded_steps.append({**step, "method": part})
    for step in expanded_steps:
        hit = _match(step.get("method"), proc_idx)
        if hit is None:
            log.append(f"unresolved process method: {step.get('method')!r}")
            continue
        key, params_schema = hit
        param_idx = {_norm(k): k for k in params_schema}
        body: Dict[str, Any] = {}
        for p in step.get("params") or []:
            field = _match(p.get("name"), param_idx)
            if field is None:
                log.append(f"unresolved {key} param: {p.get('name')!r}")
                continue
            val = p.get("value")
            body[field] = val
        kstep: Dict[str, Any] = {key: body}
        if step.get("inputs"):
            kstep["M_in"] = [{"name": n, "function": "precursor"} for n in step["inputs"]]
        for mid in step.get("applies_to") or list(mids):
            mat = mat_by_id.get(mids.get(mid, ""))
            if mat is not None:
                mat.setdefault("process", []).append(json.loads(json.dumps(kstep)))

    # ---- property values --------------------------------------------------
    for pv in iv.get("property_values") or []:
        path = _match_property(pv.get("property"), prop_paths)
        mat = mat_by_id.get(mids.get(pv.get("of"), ""))
        if path is None or mat is None:
            log.append(f"unresolved property: {pv.get('property')!r} of {pv.get('of')!r}")
            continue
        tech = _match(pv.get("technique"), meas_idx)
        measurement = None
        if tech is not None:
            tkey, tschema = tech
            tbody: Dict[str, Any] = {}
            if pv.get("instrument") and "instrument" in tschema:
                tbody["instrument"] = pv["instrument"]
            measurement = [{tkey: tbody}]
        elif pv.get("technique"):
            log.append(f"unresolved technique: {pv.get('technique')!r}")

        prop = mat.setdefault("property", {})
        if "[]" in "".join(path):  # value-array shape, e.g. thermoelectric property
            group = path[0]
            leaf = path[-1]
            arr_field = path[1].rstrip("[]") or "value"
            gnode = prop.setdefault(group, {})
            item: Dict[str, Any] = {leaf: {"value": pv.get("value")}}
            m_temp = re.search(r"(\d+(?:\.\d+)?)\s*K\b", str(pv.get("conditions") or ""))
            if m_temp:
                item["temperature"] = {"value": float(m_temp.group(1))}
            gnode.setdefault(arr_field, []).append(item)
            if measurement:
                gnode.setdefault("measurement", []).extend(measurement)
        else:
            group, leaf = path[0], path[-1]
            node: Dict[str, Any] = {"value": pv.get("value")}
            if measurement:
                node["measurement"] = measurement
            prop.setdefault(group, {})[leaf] = node

    # ---- publication ------------------------------------------------------
    p = iv.get("paper") or {}
    publication: Dict[str, Any] = {
        "DOI": p.get("doi"), "title": p.get("title"),
        "journal": ({"name": p.get("journal")} if p.get("journal") else None),
        "year": p.get("year"), "article number": p.get("article_number"),
        "abstract": p.get("abstract_summary"),
        "authors": [{"given name": a.get("given_name"), "family name": a.get("family_name"),
                     "affiliations": [{"name": x} for x in (a.get("affiliations") or [])]}
                    for a in (p.get("authors") or [])],
        "references": [d for d in (iv.get("reference_dois") or []) if d],
        "samples": [], "figures": [], "text passages": [],
        "scope": {"conclusions": [{"text": c.get("text"),
                                   "samples": sample_refs(c.get("samples"))}
                                  for c in (iv.get("conclusions") or [])]},
    }

    for s in iv.get("samples") or []:
        refs = [{"scheme": "NCMRD material", "id": mids[m]}
                for m in (s.get("made_of") or []) if m in mids]
        publication["samples"].append({
            "sample local id": sids[s.get("sid")],
            "name": s.get("label"),
            "description": s.get("description"),
            "mixing type": s.get("mixing_type"),
            "components": [{"name": mid_name.get(m, s.get("label")),
                            "role": s.get("role"),
                            "references": [{"scheme": "NCMRD material", "id": mids[m]}]}
                           for m in (s.get("made_of") or []) if m in mids] or
                          [{"name": s.get("label"), "role": s.get("role"),
                            "references": refs}],
        })

    g_no = 0
    for fi, f in enumerate(iv.get("figures") or []):
        graphs = []
        for g in f.get("graphs") or []:
            g_no += 1
            graphs.append({
                "graph local id": f"g-{g_no:03d}",
                "graph name": g.get("what"),
                "subfigure label": g.get("panel"),
                "caption summary": f.get("caption_summary"),
                "axes": [_axis("x", g.get("x")), _axis("y", g.get("y"))],
                "samples": sample_refs(g.get("samples")),
                "structure": None,
                "description": g.get("description"),
                "digitization": "",
            })
        publication["figures"].append({
            "figure local id": f"f-{fi+1:03d}",
            "figure name": f.get("name") or f"Figure {f.get('fig')}",
            "structure": (f"{f['n_panels']} panels" if f.get("n_panels") else None),
            "description": f.get("caption_summary"),
            "graphs": graphs,
        })

    if iv.get("tables"):
        publication["tables"] = [{"table local id": f"t-{i+1:03d}",
                                  "table name": f"Table {t.get('label')}",
                                  "description": t.get("caption"),
                                  "samples": sample_refs(t.get("samples"))}
                                 for i, t in enumerate(iv["tables"])]

    for i, ps in enumerate(iv.get("passages") or []):
        publication["text passages"].append({
            "passage local id": f"p-{i+1:03d}",
            "passage name": ps.get("name"),
            "section": ps.get("section"),
            "text": ps.get("text"),
            "samples": sample_refs(ps.get("samples")),
        })

    record = {
        "metadata": {
            "data name": iv.get("data_name"),
            # MSIT classification code cannot be read from the paper — default to
            # the materials-engineering code the production pipeline also uses.
            "data classification": ["EB0103"],
            "data generation date": time.strftime("%Y-%m-%d"),
            "data source": "publication",
            "contributor": {"name": "Extracted, Auto",
                            "affiliation": "AutoLineDigitizer interview-compile prototype",
                            "email address": "noreply@autolinedigitizer.example"},
            "keywords": p.get("keywords") or [],
            "embargo": None,
            "rights": None,
            "publication": publication,
        },
        "system": None,
        "materials": materials,
    }

    # Final gates shared with the production pipeline: mechanical repair against
    # the real schema, then vocabulary refs for axes.
    log.extend(repair_record(record, schema))
    n_refs = fill_axis_refs(record, schema)
    if n_refs:
        log.append(f"filled {n_refs} axis quantity.ref")
    return record, log


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def extract_ncmrd_interview(pdf_path: str, output_dir: str,
                                 base_name: Optional[str] = None,
                                 model: str = LIGHT_MODEL,
                                 schema_path: Optional[str] = None) -> Dict[str, Any]:
    """Interview → compile → validate → save. Returns a summary dict."""
    if not ANTHROPIC_AVAILABLE or AsyncAnthropic is None:
        return {"_error": "anthropic SDK not installed"}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"_error": "ANTHROPIC_API_KEY not set"}
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "ncmrd_v15.2.4_nullable.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    t0 = time.time()
    md = _mineru_paper_markdown(pdf_path)
    if not md:
        return {"_error": "MinerU markdown unavailable (scanned PDF?) — interview needs text"}
    print(f"⤷ interview: paper ready ({md.get('engine')}, "
          f"{len(md['markdown'])//1000}k chars, {len(md.get('figures') or [])} crops)")

    async with AsyncAnthropic() as client:
        res = await conduct_interview(md["markdown"], md.get("figures"), client, model=model)
    if res.get("error"):
        os.makedirs(output_dir, exist_ok=True)
        if res.get("raw"):
            with open(os.path.join(output_dir, f"{base_name}_interview_raw.txt"),
                      "w", encoding="utf-8") as f:
                f.write(res["raw"])
        return {"_error": res["error"]}
    print(f"   ✓ interview answered: out={res['output_tokens']} tok "
          f"({res['elapsed_sec']:.1f}s, {model})")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{base_name}_interview.json"), "w",
              encoding="utf-8") as f:
        json.dump(res["answers"], f, indent=2, ensure_ascii=False)

    record, log = compile_interview(res["answers"], schema)
    for line in log:
        print(f"   · compile: {line}")

    n_violations = None
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(schema)
        n_violations = sum(1 for _ in v.iter_errors(record))
    except ImportError:
        pass

    en_path = os.path.join(output_dir, f"{base_name}.json")
    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"   ✅ compiled NCMRD record: {en_path}")
    print(f"   schema violations: {n_violations}   |   {elapsed:.1f}s total   |   "
          f"in={res['input_tokens']} cache_w={res['cache_creation_tokens']} "
          f"cache_r={res['cache_read_tokens']} out={res['output_tokens']} tok")
    return {"en_path": en_path, "n_violations": n_violations,
            "elapsed_sec": elapsed, "compile_log": log,
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "cache_creation_tokens": res["cache_creation_tokens"],
            "cache_read_tokens": res["cache_read_tokens"], "model": model}


if __name__ == "__main__":
    import sys
    asyncio.run(extract_ncmrd_interview(sys.argv[1], sys.argv[2]))
