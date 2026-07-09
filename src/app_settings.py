# -*- coding: utf-8 -*-
"""Per-user app settings — currently the shared Anthropic API key.

The settings file lives in the user data directory (never inside a repo
checkout, so a dev run can't accidentally commit a key). The environment
variable ANTHROPIC_API_KEY always wins over the saved key, so power users
and CI keep their existing workflow.
"""
import json
import os
import sys


def get_settings_path():
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        data_dir = os.path.join(os.path.expanduser("~"), "Library",
                                "Application Support", "AutoLineDigitizer")
    elif getattr(sys, "frozen", False) and sys.platform == "win32":
        data_dir = os.path.join(os.environ.get("LOCALAPPDATA",
                                               os.path.expanduser("~")),
                                "AutoLineDigitizer")
    else:
        data_dir = os.path.join(os.path.expanduser("~"), ".autolinedigitizer")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "settings.json")


def read_settings(path=None):
    try:
        with open(path or get_settings_path(), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def load_saved_api_key(path=None):
    """Export the saved key as ANTHROPIC_API_KEY unless one is already set.

    Returns True if a key is available (from either source) afterwards.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    key = (read_settings(path).get("anthropic_api_key") or "").strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
        return True
    return False


def save_api_key(key, path=None):
    """Persist the key (0600 perms) and activate it for this process.

    An empty key clears the saved one (the env var, if set externally,
    is left alone).
    """
    path = path or get_settings_path()
    cfg = read_settings(path)
    key = (key or "").strip()
    cfg["anthropic_api_key"] = key
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
