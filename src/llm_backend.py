# -*- coding: utf-8 -*-
"""llm_backend.py — one chat client, two interchangeable backends:

  "anthropic"  the Claude API (needs the anthropic SDK + an API key)
  "local"      any OpenAI-compatible server on the lab network — vLLM,
               Ollama, LM Studio… — so closed-access papers never leave
               the building.

Configuration (settings.json via app_settings; the env var always wins):

  llm_backend      / ALD_LLM_BACKEND      "anthropic" | "local".
                                          Unset = auto: anthropic when a key
                                          is available, else local when a
                                          server URL is configured.
  local_llm_url    / ALD_LOCAL_LLM_URL    e.g. "http://192.168.1.50:8000/v1"
  local_llm_model  / ALD_LOCAL_LLM_MODEL  served model id; blank = ask the
                                          server (GET /models, first entry)
  local_llm_key    / ALD_LOCAL_LLM_KEY    bearer token, only if the server
                                          requires one (vLLM --api-key)

Callers hand over Anthropic-style content blocks (text / base64 image);
the local path converts them to the OpenAI chat format. PDF "document"
blocks have no local equivalent — the local KMDS path feeds MinerU
markdown instead of raw PDFs.
"""
import json
import os

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

try:
    from app_settings import get_setting
except ImportError:          # standalone use outside the app
    def get_setting(name, default="", path=None):
        return default

_LOCAL_TIMEOUT = (10, 600)   # connect, read — a 32B model on one GPU is slow


def _cfg(env_name, setting_name):
    return (os.environ.get(env_name) or get_setting(setting_name) or "").strip()


def local_url():
    return _cfg("ALD_LOCAL_LLM_URL", "local_llm_url").rstrip("/")


def local_configured():
    return bool(local_url())


def anthropic_ready():
    """SDK importable and a key reachable (env or saved settings)."""
    return ANTHROPIC_SDK_AVAILABLE and bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or get_setting("anthropic_api_key"))


def resolve_backend(explicit=None):
    """Return "anthropic", "local", or None when neither is usable."""
    choice = (explicit or _cfg("ALD_LLM_BACKEND", "llm_backend")).lower()
    if choice == "anthropic":
        return "anthropic" if ANTHROPIC_SDK_AVAILABLE else None
    if choice == "local":
        return "local" if local_configured() else None
    if anthropic_ready():
        return "anthropic"
    if local_configured():
        return "local"
    # SDK present but no key yet: keep the old UX (fail at call time with
    # a clear message) rather than silently disabling every AI button.
    return "anthropic" if ANTHROPIC_SDK_AVAILABLE else None


def backend_available():
    return resolve_backend() is not None


class LLMBackend:
    """Chat once, in whichever backend is active.

    chat() takes an Anthropic-style content-block list and returns the
    model's text. Claude model names are honored on the anthropic backend
    and transparently replaced by the served model on the local one, so
    callers never need to branch.
    """

    def __init__(self, backend=None, api_key=None, verify_ssl=True):
        self.backend = resolve_backend(backend)
        if self.backend is None:
            raise RuntimeError(
                "No AI backend available. Either install the anthropic SDK "
                "and set an API key (Settings → Claude API), or point the "
                "app at a local model server (ALD_LOCAL_LLM_URL or "
                "local_llm_url in settings.json).")
        self.verify_ssl = verify_ssl
        self._local_model_cache = None
        if self.backend == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY") \
                or get_setting("anthropic_api_key")
            if not key:
                raise RuntimeError(
                    "Set ANTHROPIC_API_KEY (Settings → Claude API) or "
                    "configure a local model server.")
            if verify_ssl:
                self.client = anthropic.Anthropic(api_key=key)
            else:
                import httpx
                self.client = anthropic.Anthropic(
                    api_key=key, http_client=httpx.Client(verify=False))

    # ---- public ----------------------------------------------------------

    def chat(self, system, blocks, model=None, max_tokens=1024):
        if self.backend == "anthropic":
            return self._chat_anthropic(system, blocks, model, max_tokens)
        return self._chat_local(system, blocks, max_tokens)

    # ---- anthropic -------------------------------------------------------

    def _chat_anthropic(self, system, blocks, model, max_tokens):
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": blocks}],
        )
        return "".join(b.text for b in message.content if b.type == "text")

    # ---- local (OpenAI-compatible) ---------------------------------------

    def _chat_local(self, system, blocks, max_tokens):
        import requests
        url = local_url()
        headers = {"Content-Type": "application/json"}
        token = _cfg("ALD_LOCAL_LLM_KEY", "local_llm_key")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = {
            "model": self._local_model(headers),
            "max_tokens": max_tokens,
            "temperature": 0,     # curation wants determinism, not flair
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._blocks_to_openai(blocks)},
            ],
        }
        resp = requests.post(f"{url}/chat/completions", json=payload,
                             headers=headers, timeout=_LOCAL_TIMEOUT,
                             verify=self.verify_ssl)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content")
        if isinstance(content, list):   # some servers return content parts
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict))
        return content or ""

    def _local_model(self, headers):
        configured = _cfg("ALD_LOCAL_LLM_MODEL", "local_llm_model")
        if configured:
            return configured
        if self._local_model_cache:
            return self._local_model_cache
        import requests
        resp = requests.get(f"{local_url()}/models", headers=headers,
                            timeout=(10, 30), verify=self.verify_ssl)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            raise RuntimeError(
                f"Local server at {local_url()} lists no models — set "
                "local_llm_model / ALD_LOCAL_LLM_MODEL explicitly.")
        self._local_model_cache = data[0]["id"]
        return self._local_model_cache

    @staticmethod
    def _blocks_to_openai(blocks):
        out = []
        for b in blocks:
            kind = b.get("type")
            if kind == "text":
                out.append({"type": "text", "text": b["text"]})
            elif kind == "image":
                src = b["source"]
                uri = f"data:{src['media_type']};base64,{src['data']}"
                out.append({"type": "image_url", "image_url": {"url": uri}})
            elif kind == "document":
                raise ValueError(
                    "PDF document blocks are Anthropic-only; the local "
                    "backend takes MinerU markdown text instead.")
            else:
                raise ValueError(f"Unsupported content block type: {kind!r}")
        return out
