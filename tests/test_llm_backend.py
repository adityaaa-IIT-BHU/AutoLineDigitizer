# -*- coding: utf-8 -*-
"""llm_backend + VLMVerifier local path, tested against a fake
OpenAI-compatible server (vLLM/Ollama stand-in) running on a thread —
no GPU, no network, no API keys needed."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import llm_backend
from llm_backend import LLMBackend, resolve_backend


class _FakeOpenAIServer(BaseHTTPRequestHandler):
    """Answers /models and /chat/completions like vLLM; records requests."""
    canned_text = "{}"
    requests_seen = []

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send({"data": [{"id": "fake/served-model"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        type(self).requests_seen.append(body)
        if self.path.endswith("/chat/completions"):
            self._send({"choices": [{"message": {
                "role": "assistant", "content": type(self).canned_text}}]})
        else:
            self.send_error(404)

    def _send(self, obj):
        payload = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # silence test output
        pass


@pytest.fixture()
def fake_server(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAIServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("ALD_LOCAL_LLM_URL", url)
    monkeypatch.setenv("ALD_LLM_BACKEND", "local")
    monkeypatch.delenv("ALD_LOCAL_LLM_MODEL", raising=False)
    # isolate from the user's real settings.json (saved model/url would leak)
    monkeypatch.setattr(llm_backend, "get_setting",
                        lambda name, default="", path=None: default)
    _FakeOpenAIServer.requests_seen = []
    _FakeOpenAIServer.canned_text = "{}"
    yield url
    server.shutdown()
    server.server_close()


def test_resolve_backend_prefers_explicit_local(fake_server):
    assert resolve_backend() == "local"


def test_resolve_backend_none_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("ALD_LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("ALD_LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_backend, "get_setting",
                        lambda name, default="", path=None: default)
    if llm_backend.ANTHROPIC_SDK_AVAILABLE:
        # SDK installed: old UX kept — backend resolves, key error at call time
        assert resolve_backend() == "anthropic"
    else:
        assert resolve_backend() is None


def test_local_chat_converts_blocks_and_autodetects_model(fake_server):
    _FakeOpenAIServer.canned_text = "hello from local"
    backend = LLMBackend()
    out = backend.chat(
        system="sys prompt",
        model="claude-opus-4-8",   # must be replaced by the served model
        max_tokens=99,
        blocks=[
            {"type": "text", "text": "look at this"},
            {"type": "image", "source": {"type": "base64",
                "media_type": "image/png", "data": "aGk="}},
        ],
    )
    assert out == "hello from local"
    req = _FakeOpenAIServer.requests_seen[-1]
    assert req["model"] == "fake/served-model"
    # local backend doubles the budget (min 2048) for thinking-style models
    assert req["max_tokens"] == 2048
    assert req["messages"][0] == {"role": "system", "content": "sys prompt"}
    user = req["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "look at this"}
    assert user[1]["image_url"]["url"] == "data:image/png;base64,aGk="


def test_local_chat_honors_configured_model(fake_server, monkeypatch):
    monkeypatch.setenv("ALD_LOCAL_LLM_MODEL", "my/model")
    LLMBackend().chat(system="s", blocks=[{"type": "text", "text": "t"}])
    assert _FakeOpenAIServer.requests_seen[-1]["model"] == "my/model"


def test_document_blocks_rejected_on_local(fake_server):
    backend = LLMBackend()
    with pytest.raises(ValueError, match="MinerU"):
        backend.chat(system="s", blocks=[
            {"type": "document", "source": {"type": "base64",
                "media_type": "application/pdf", "data": "aGk="}}])


def test_vlm_verifier_end_to_end_local(fake_server):
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    from vlm_verifier import VLMVerifier
    _FakeOpenAIServer.canned_text = json.dumps({
        "x_axis": {"name": "Temperature", "unit": "K", "is_log": False},
        "y_axis": {"name": "ZT", "unit": "", "is_log": False},
        "notes": "",
    })
    v = VLMVerifier()
    assert v.backend_name == "local"
    img = np.full((60, 80, 3), 255, dtype=np.uint8)
    parsed = v.read_axis_properties(img)
    assert parsed["x_axis"]["name"] == "Temperature"
    # the request really carried the chart as an image block
    user = _FakeOpenAIServer.requests_seen[-1]["messages"][1]["content"]
    assert any(p.get("type") == "image_url" for p in user)


def test_parse_response_handles_arrays_and_fences():
    from vlm_verifier import VLMVerifier
    arr = VLMVerifier._parse_response(
        '```json\n[{"label": "a"}, {"label": "b"}]\n```')
    assert [x["label"] for x in arr] == ["a", "b"]
    arr2 = VLMVerifier._parse_response(
        'Sure! Here is the result:\n[{"is_property": true}] Done.')
    assert arr2[0]["is_property"] is True
    obj = VLMVerifier._parse_response('noise {"k": 1} trailing')
    assert obj == {"k": 1}
