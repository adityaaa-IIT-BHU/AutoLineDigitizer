# -*- coding: utf-8 -*-
"""Flatten a NCMRD record into editable leaf fields and apply edits back.

The NCMRD paper record is a deeply nested JSON document (~400 scalar leaves).
The desktop app renders each leaf as an editable text field; this module owns
the pure logic — flattening the record into (path, label, value) rows and
parsing edited text back with the original type respected — so it can be
unit-tested without flet.
"""
import json

SCALAR_TYPES = (str, int, float, bool, type(None))


def is_leaf(value):
    """A leaf is a scalar, or a list made only of scalars (e.g. keywords,
    reference ticks). Lists containing dicts/lists are containers."""
    if isinstance(value, SCALAR_TYPES):
        return True
    if isinstance(value, list):
        return all(isinstance(x, SCALAR_TYPES) for x in value)
    return False


def flatten_record(record):
    """-> list of (path_tuple, label, value) for every editable leaf.

    path_tuple holds dict keys / list indexes needed to reach the leaf;
    label is the human-readable dotted path (e.g. "metadata.keywords",
    "materials[0].name").
    """
    out = []

    def walk(obj, path, label):
        if path and is_leaf(obj):
            out.append((tuple(path), label, obj))
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, path + [k], f"{label}.{k}" if label else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, path + [i], f"{label}[{i}]")

    walk(record, [], "")
    return out


def leaf_to_text(value):
    """Render a leaf value for a text field. Strings appear as-is; other
    types (numbers, bools, null, scalar lists) as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def text_to_leaf(text, original):
    """Parse edited text back into a value, respecting the original type.

    Returns (value, error): error is None on success, else a message and the
    original value is returned unchanged.

    Rules: string fields stay strings verbatim; empty text on a non-string
    field means null; anything else must parse as JSON.
    """
    if isinstance(original, str):
        return text, None
    stripped = text.strip()
    if stripped == "":
        return None, None
    try:
        return json.loads(stripped), None
    except ValueError:
        if original is None:
            # a null field freely accepts plain text
            return text, None
        return original, (f"not valid JSON for a "
                          f"{type(original).__name__} field: {text!r}")


def apply_edit(record, path, value):
    """Set the leaf at path (from flatten_record) to value, in place."""
    obj = record
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def apply_text_edits(record, rows):
    """Apply many text edits at once.

    rows: iterable of (path_tuple, original_value, new_text).
    Returns a list of error strings (empty if everything applied cleanly);
    fields that fail to parse keep their original value.
    """
    errors = []
    for path, original, text in rows:
        if leaf_to_text(original) == text:
            continue                       # untouched
        value, err = text_to_leaf(text, original)
        if err:
            label = ".".join(str(p) for p in path)
            errors.append(f"{label}: {err}")
            continue
        apply_edit(record, path, value)
    return errors
