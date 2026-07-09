# -*- coding: utf-8 -*-
"""Unit tests for kmds_editor (pure logic, no flet)."""
import copy

from kmds_editor import (flatten_record, leaf_to_text, text_to_leaf,
                         apply_edit, apply_text_edits)

RECORD = {
    "metadata": {
        "data name": "test",
        "keywords": ["a", "b"],
        "embargo": None,
        "publication": {"DOI": "10.1/x", "figures": [
            {"graphs": [{"axes": [{"reference ticks": [1.0, 2.0, 3.0]}]}]},
        ]},
    },
    "system": {"n_cells": 3, "calibrated": True},
}


def test_flatten_finds_all_leaves():
    rows = flatten_record(copy.deepcopy(RECORD))
    labels = {label for _, label, _ in rows}
    assert "metadata.data name" in labels
    assert "metadata.keywords" in labels            # scalar list = one leaf
    assert "metadata.embargo" in labels
    assert "metadata.publication.figures[0].graphs[0].axes[0].reference ticks" in labels
    assert "system.n_cells" in labels
    assert len(rows) == 7


def test_leaf_text_round_trip():
    assert leaf_to_text("plain") == "plain"
    assert leaf_to_text(None) == ""
    assert leaf_to_text(3) == "3"
    assert leaf_to_text(True) == "true"
    assert leaf_to_text(["a", "b"]) == '["a", "b"]'
    assert text_to_leaf('["a", "b"]', ["a"]) == (["a", "b"], None)
    assert text_to_leaf("3.5", 3) == (3.5, None)
    assert text_to_leaf("false", True) == (False, None)


def test_string_fields_stay_verbatim():
    # strings must never be JSON-parsed: "3" stays a string, spaces survive
    assert text_to_leaf("3", "old") == ("3", None)
    assert text_to_leaf(" spaced ", "old") == (" spaced ", None)


def test_null_field_accepts_text_number_or_empty():
    assert text_to_leaf("", None) == (None, None)
    assert text_to_leaf("42", None) == (42, None)
    assert text_to_leaf("free text", None) == ("free text", None)


def test_bad_json_keeps_original_and_reports():
    value, err = text_to_leaf("not-a-number", 5)
    assert value == 5 and err is not None


def test_apply_edit_nested():
    rec = copy.deepcopy(RECORD)
    apply_edit(rec, ("metadata", "publication", "figures", 0, "graphs", 0,
                     "axes", 0, "reference ticks"), [9, 8])
    assert rec["metadata"]["publication"]["figures"][0]["graphs"][0][
        "axes"][0]["reference ticks"] == [9, 8]


def test_apply_text_edits_mixed():
    rec = copy.deepcopy(RECORD)
    rows = [
        (("metadata", "data name"), "test", "renamed"),
        (("system", "n_cells"), 3, "7"),
        (("system", "calibrated"), True, "not json"),   # error, keeps original
        (("metadata", "embargo"), None, ""),            # untouched null
    ]
    errors = apply_text_edits(rec, rows)
    assert rec["metadata"]["data name"] == "renamed"
    assert rec["system"]["n_cells"] == 7
    assert rec["system"]["calibrated"] is True
    assert len(errors) == 1 and "calibrated" in errors[0]
