# -*- coding: utf-8 -*-
"""Tests for the schema-guided repair pass and axis quantity.ref filler."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

jsonschema = pytest.importorskip("jsonschema")

from ncmrd_parallel import repair_record, validate_record, fill_axis_refs  # noqa: E402


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "materials": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "property": {
                        "type": ["object", "null"],
                        "properties": {
                            "electrical property": {
                                "type": ["object", "null"],
                                "properties": {
                                    "band gap energy": {
                                        "type": ["object", "null"],
                                        "properties": {
                                            "value": {"type": ["number", "null"]},
                                        },
                                        "additionalProperties": False,
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                    },
                    "step": {
                        "type": ["object", "null"],
                        "required": ["ball size"],
                        "properties": {
                            "ball size": {"type": ["number", "null"]},
                            "jar material": {"type": ["string", "null"]},
                        },
                        "additionalProperties": False,
                    },
                    "notes": {"type": ["array", "null"],
                              "items": {"type": ["string", "null"]}},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def test_repair_snakecase_rename_and_wrapper_flatten():
    rec = {"materials": [{
        "step": {"jar_material": "steel", "ball size": 5},
        "property": {"electrical property": {"band gap energy": {
            "value": {"value": 0.53, "uncertainty": None}}}},
    }]}
    log = repair_record(rec, SCHEMA)
    assert validate_record(rec, SCHEMA) == []
    assert rec["materials"][0]["step"]["jar material"] == "steel"
    bge = rec["materials"][0]["property"]["electrical property"]["band gap energy"]
    assert bge["value"] == 0.53
    assert any("renamed" in l for l in log)
    assert any("flattened" in l for l in log)


def test_repair_adds_nullable_required_and_moves_unknown_to_notes():
    rec = {"materials": [{
        "step": {"jar material": "steel"},      # missing required 'ball size'
        "unknown_thing": "provenance text",     # item level has notes[] sibling
    }]}
    log = repair_record(rec, SCHEMA)
    assert validate_record(rec, SCHEMA) == []
    assert rec["materials"][0]["step"]["ball size"] is None
    assert any("unknown_thing" in n for n in rec["materials"][0]["notes"])
    assert any("added missing required" in l for l in log)


def test_repair_drops_unknown_when_no_notes_available():
    rec = {"materials": [{"step": {"ball size": 3, "rogue": 1}}]}
    log = repair_record(rec, SCHEMA)
    assert validate_record(rec, SCHEMA) == []
    assert "rogue" not in rec["materials"][0]["step"]
    assert any("dropped" in l or "moved" in l for l in log)


def test_fill_axis_refs_unique_terms_only():
    schema = {
        "properties": {"materials": {"items": {"properties": {"property": {
            "properties": {
                "electrical property": {"properties": {
                    "electrical conductivity": {"description": "x"},
                    "temperature": {"description": "cond"},
                }},
                "thermophysical property": {"properties": {
                    "temperature": {"description": "cond"},   # ambiguous
                }},
            }}}}}},
    }
    rec = {"metadata": {"publication": {"figures": [{
        "figure local id": "f-001",
        "graphs": [{"graph local id": "g-001", "axes": [
            {"axis": "x", "quantity": {"term": "Temperature"}},
            {"axis": "y", "quantity": {"term": "Electrical Conductivity"}},
        ]}],
    }]}}}
    n = fill_axis_refs(rec, schema)
    axes = rec["metadata"]["publication"]["figures"][0]["graphs"][0]["axes"]
    assert n == 1
    assert "ref" not in axes[0]["quantity"]          # ambiguous -> untouched
    assert axes[1]["quantity"]["ref"].endswith("electrical conductivity")
