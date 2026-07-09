# -*- coding: utf-8 -*-
"""Unit tests for app_settings (no flet/torch required)."""
import json
import os

from app_settings import load_saved_api_key, save_api_key, read_settings


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = str(tmp_path / "settings.json")
    save_api_key("sk-ant-test123", path=p)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test123"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert load_saved_api_key(path=p) is True
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test123"


def test_env_var_wins_over_saved_key(tmp_path, monkeypatch):
    p = str(tmp_path / "settings.json")
    save_api_key("sk-ant-saved", path=p)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert load_saved_api_key(path=p) is True
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-env"


def test_empty_key_clears_saved(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = str(tmp_path / "settings.json")
    save_api_key("sk-ant-old", path=p)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    save_api_key("", path=p)
    assert read_settings(p)["anthropic_api_key"] == ""
    assert load_saved_api_key(path=p) is False


def test_missing_file_is_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert load_saved_api_key(path=str(tmp_path / "nope.json")) is False


def test_file_permissions_are_private(tmp_path):
    p = str(tmp_path / "settings.json")
    save_api_key("sk-ant-perm", path=p)
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"
    assert json.load(open(p))["anthropic_api_key"] == "sk-ant-perm"
