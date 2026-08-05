# -*- coding: utf-8 -*-
"""NCMRD Foundry local backend, tested against a configurable fake Ollama.
Covers the review findings: dynamic num_ctx, silent-truncation guard,
schema compaction, image/think 400-retries, OpenAI fallback, backfill
type safety. No GPU, no network."""
import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ncmrd_parallel as kp
import llm_backend


class _FakeOllama(BaseHTTPRequestHandler):
    requests_seen = []
    canned = "```json\n{\"metadata\": {}}\n```"
    reject_think = False          # 400 the first think:false request
    reject_images = False         # 400 any request carrying images
    native_missing = False        # 404 /api/chat (OpenAI-only server)
    prompt_eval = 11              # reported prompt_eval_count

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests_seen.append((self.path, body))
        c = type(self)
        if self.path == "/api/chat":
            if c.native_missing:
                return self._send(404, {"error": "not found"})
            if c.reject_think and "think" in body:
                return self._send(400, {"error": "model does not support thinking"})
            if c.reject_images and body["messages"][0].get("images"):
                return self._send(400, {"error": "model does not support images"})
            return self._send(200, {
                "message": {"role": "assistant", "content": c.canned},
                "done_reason": "stop",
                "prompt_eval_count": c.prompt_eval, "eval_count": 7})
        if self.path.endswith("/chat/completions"):
            return self._send(200, {
                "choices": [{"message": {"role": "assistant",
                                         "content": c.canned}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
        self._send(404, {"error": "nope"})

    def _send(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ALD_LOCAL_LLM_URL",
                       f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(llm_backend, "get_setting",
                        lambda name, default="", path=None: default)
    f = _FakeOllama
    f.requests_seen = []
    f.canned = "```json\n{\"metadata\": {}}\n```"
    f.reject_think = f.reject_images = f.native_missing = False
    f.prompt_eval = 11
    yield f
    server.shutdown()
    server.server_close()


BLOCKS = {"ground_rules": "R", "overview": "O",
          "material_vs_sample": "", "conditions": "", "translation": ""}


def test_default_model_honors_local_pin(fake, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ALD_LLM_BACKEND", "local")
    if kp.ANTHROPIC_AVAILABLE:
        assert kp.default_model().startswith(kp.LOCAL_PREFIX)
    monkeypatch.setenv("ALD_LLM_BACKEND", "anthropic")
    if kp.ANTHROPIC_AVAILABLE:
        assert kp.default_model() == kp.MODEL


def test_local_generate_sends_dynamic_ctx(fake):
    out = asyncio.run(kp._local_generate(
        [{"type": "text", "text": "hello"}], "m", 500, num_ctx=12345))
    _, body = fake.requests_seen[-1]
    assert body["options"]["num_ctx"] == 12345
    assert body["options"]["num_predict"] == 500
    assert body["think"] is False
    assert out["prompt_eval"] == 11


def test_think_400_retries_without_think(fake):
    fake.reject_think = True
    out = asyncio.run(kp._local_generate(
        [{"type": "text", "text": "hi"}], "m", 100))
    assert out["output_tokens"] == 7
    paths = [b for p, b in fake.requests_seen if p == "/api/chat"]
    assert "think" in paths[0] and "think" not in paths[1]


def test_images_400_retries_without_images(fake):
    fake.reject_images = True
    blocks = [{"type": "text", "text": "look"},
              {"type": "image", "source": {"type": "base64",
                  "media_type": "image/png", "data": "aGk="}}]
    out = asyncio.run(kp._local_generate(blocks, "not-a-vl-name", 100))
    assert out["output_tokens"] == 7
    reqs = [b for p, b in fake.requests_seen if p == "/api/chat"]
    assert reqs[0]["messages"][0].get("images") == ["aGk="]
    assert "images" not in reqs[-1]["messages"][0]


def test_images_kept_when_server_accepts(fake):
    blocks = [{"type": "text", "text": "look"},
              {"type": "image", "source": {"type": "base64",
                  "media_type": "image/png", "data": "aGk="}}]
    asyncio.run(kp._local_generate(blocks, "llava:13b", 100))
    _, body = fake.requests_seen[-1]
    assert body["messages"][0]["images"] == ["aGk="]   # no name heuristic


def test_openai_fallback_on_404(fake):
    fake.native_missing = True
    out = asyncio.run(kp._local_generate(
        [{"type": "text", "text": "hi"}], "vllm-model", 100))
    assert out["text"].startswith("```json")
    assert any(p.endswith("/chat/completions") for p, _ in fake.requests_seen)


def test_local_generate_rejects_pdf(fake):
    with pytest.raises(RuntimeError, match="MinerU"):
        asyncio.run(kp._local_generate(
            [{"type": "document", "source": {"type": "base64",
                "media_type": "application/pdf", "data": "aGk="}}], "m", 100))


def test_compact_schema_strips_prose():
    frag = {"properties": {"x": {"type": "string",
                                "description": "long prose " * 100,
                                "examples": ["a"],
                                "enum": ["p", "q"]}}}
    c = kp._compact_schema(frag)
    assert "description" not in c["properties"]["x"]
    assert "examples" not in c["properties"]["x"]
    assert c["properties"]["x"]["enum"] == ["p", "q"]
    assert kp._compact_schema(None) is None
    assert "description" in frag["properties"]["x"]    # original untouched


def test_section_drops_schema_when_over_budget(fake, monkeypatch):
    # cap must cover the output budget (12k for core) but not the schema
    monkeypatch.setattr(kp, "LOCAL_NUM_CTX", 26000)
    fake.prompt_eval = 900
    big_schema = {"properties": {"metadata": {
        "description": "x" * 40000, "filler": "y" * 80000}}}
    res = asyncio.run(kp.extract_one_section(
        "PDF", "core", None, BLOCKS, section_schema=big_schema,
        model="local:m", paper_text="short paper"))
    assert res["ok"], res["error"]
    _, body = fake.requests_seen[-1]
    assert "y" * 100 not in body["messages"][0]["content"]  # schema dropped
    assert "repair pass" in body["messages"][0]["content"]


def test_section_fails_loud_on_server_truncation(fake, monkeypatch):
    monkeypatch.setattr(kp, "LOCAL_NUM_CTX", 60000)
    fake.prompt_eval = 50           # server claims tiny prompt => truncated
    res = asyncio.run(kp.extract_one_section(
        "PDF", "core", None, BLOCKS, section_schema=None,
        model="local:m", paper_text="word " * 4000))
    assert not res["ok"]
    assert "truncated" in res["error"]


def test_data_sources_local_markdown_without_crops(fake):
    fake.canned = "```json\n{\"figures\": []}\n```"
    res = asyncio.run(kp.extract_one_section(
        "PDF", "data_sources", None, BLOCKS,
        model="local:m", paper_text="markdown here", paper_figures=None))
    assert res["ok"], res["error"]
    _, body = fake.requests_seen[-1]
    assert "markdown here" in body["messages"][0]["content"]


def test_backfill_type_safety():
    md = "# A Great Paper Title Of Substance\ndoi 10.1234/abc.def"
    r1 = {"metadata": {"publication": "not a dict"}}
    log = kp._backfill_bibliography(r1, md)
    assert r1["metadata"]["publication"]["DOI"] == "10.1234/abc.def"
    r2 = {"metadata": {"publication": {"DOI": {"value": "x"},
                                      "title": 42}}}
    kp._backfill_bibliography(r2, md)     # non-str values: no crash
    assert kp._backfill_bibliography("nonsense", md) == []
    assert any("title" in x for x in log)


def test_confidence_report_flags_ungrounded():
    record = {"metadata": {"publication": {
        "title": "Thermoelectric properties of FAST materials studied here",
        "abstract": "completely invented sentence that is nowhere in the text",
        "caption summary": "paraphrase fields are excluded from the audit",
    }}, "id": "x"}
    paper = ("We report the Thermoelectric properties of FAST materials "
             "studied here in detail.")
    rep = kp._confidence_report(record, paper)
    assert rep["checked_fields"] == 2          # summary field excluded
    assert rep["verified_fields"] == 1
    assert rep["unverified"][0]["path"].endswith("abstract")


def test_finalize_record_closes_clerical_violations():
    rec = {"metadata": {"publication": {
        "title": "A Sufficiently Long Paper Title", "DOI": "10.1/x",
        "year": "2020", "journal": "ACS Materials",
        "authors": [{}, {"name": "Aiko Tanaka"}],
        "samples": [{"name": "s1"}, {"name": "s2",
                     "sample local id": "s-001"}],
    }}}
    log = kp._finalize_record(rec)
    meta = rec["metadata"]
    pub = meta["publication"]
    assert meta["data classification"] == ["EB0103"]
    assert meta["contributor"]["name"]
    assert pub["year"] == 2020
    assert pub["journal"] == {"name": "ACS Materials"}
    assert pub["authors"] == [{"given name": "Aiko",
                               "family name": "Tanaka"}]
    ids = [s["sample local id"] for s in pub["samples"]]
    assert len(set(ids)) == 2 and "s-001" in ids
    assert kp._finalize_record({"metadata": {}}) is not None   # no crash
    assert log
