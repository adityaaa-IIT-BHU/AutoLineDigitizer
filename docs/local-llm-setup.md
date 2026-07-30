# Local model server setup (RTX 4090)

AutoLineDigitizer's AI features (✦ Ask Claude, legend labeling, curve
verify, vocab canonicalization) can run against any OpenAI-compatible
server instead of the Claude API, so closed-access figures never leave
the lab network. This sets one up on the 4090 machine.

## Option A — vLLM (Linux / WSL2, recommended)

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92
```

- The 32B AWQ build fits in 24 GB. If it OOMs, drop `--max-model-len`
  to 8192 or fall back to `Qwen/Qwen2.5-VL-7B-Instruct` (no AWQ needed).
- Add `--api-key <token>` if the machine is on a shared subnet; put the
  same token in the app (`local_llm_key` / `ALD_LOCAL_LLM_KEY`).

## Option B — Ollama (native Windows, simplest)

```
ollama pull qwen2.5vl:32b
set OLLAMA_HOST=0.0.0.0
ollama serve
```

Serves at `http://<machine>:11434/v1` (note the `/v1`).

## Point the app at it

Settings → **Local model server**:

- Local server URL: `http://<4090-ip>:8000/v1` (vLLM) or
  `http://<4090-ip>:11434/v1` (Ollama)
- Model id: blank (auto-detected from `GET /models`)
- AI backend: **Local server** — or **Auto** to prefer Claude when an
  API key is set and fall back to local otherwise.

Env vars override settings: `ALD_LLM_BACKEND`, `ALD_LOCAL_LLM_URL`,
`ALD_LOCAL_LLM_MODEL`, `ALD_LOCAL_LLM_KEY`.

## Smoke test from any lab machine

```bash
curl http://<4090-ip>:8000/v1/models
curl http://<4090-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-VL-32B-Instruct-AWQ","max_tokens":20,
       "messages":[{"role":"user","content":"say ok"}]}'
```

Remember to open the port in the machine's firewall (Windows Defender
inbound rule / `ufw allow 8000`) and give the box a static IP or DHCP
reservation so the saved URL keeps working.
