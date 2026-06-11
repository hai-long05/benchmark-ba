# Benchmark — llama.cpp GGUF quantization sweep

Bachelor thesis benchmark, replicates Kurt (2026, [arXiv:2601.14277](https://arxiv.org/abs/2601.14277)) cross-family on Qwen3-4B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct on a server-class x86 CPU.

**Spec:** `docs/superpowers/specs/2026-06-11-benchmark-design.md`
**Plan:** `docs/superpowers/plans/2026-06-11-benchmark-minimal-pipeline.md`

## Setup (once)

```bash
# 1. Build llama.cpp (CPU-only) into a sibling directory
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build --config Release -j

# 2. Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (Mac, ~20 min with LM_EVAL_LIMIT=200)

```bash
source .venv/bin/activate
export LLAMA_CPP_DIR=../llama.cpp
./scripts/00_capture_env.sh
LM_EVAL_LIMIT=200 ./scripts/run_slot.sh Qwen/Qwen3-0.6B Q4_K_S 512 3 hellaswag
ls results/*/results.json
```

Expected: a `results/<run-id>/results.json` row with pp/tg medians, hellaswag acc_norm, env hash, llama.cpp commit, and lm-eval version stamped in.

The smoke test proves the pipeline shape; it does *not* replicate Kurt's numbers — that happens later on the GCP server.

## Layout

| Path | Purpose |
|---|---|
| `scripts/` | One stage per script: 00 env, 10 fetch, 20 quantize, 30 bench, 40 lm_eval, 50 aggregate, plus `run_slot.sh` orchestrator |
| `configs/` | `quants.txt` (13 schemes for the future sweep), `tasks.yaml` (lm-eval task IDs pinned per exposé §4.4) |
| `tests/` | pytest covering `bench_parse.py` aggregation and `50_aggregate.py` merge + env-drift refusal |
| `models/`, `quantized/`, `results/`, `env/` | gitignored runtime artefacts |

## Notes on lm_eval + llama.cpp compatibility

`lm_eval`'s `gguf` backend (v0.4.9.2) talks to `llama-server` via `/v1/completions` and expects the legacy "echo prompt logprobs" behavior. Modern `llama-server` builds (≥ b4000, current here is `b9595`) silently drop that field. `scripts/40_lm_eval.sh` works around this by spawning a thin Python proxy that emulates the legacy behavior via N sequential per-token requests, leveraging `llama-server`'s KV-cache for reuse. Wall-clock is ~1 it/s on Apple M1 Pro CPU, so set `LM_EVAL_LIMIT=<N>` for smoke tests; full-task runs (~10k items) are reserved for the GCP server.

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Six tests: 3 covering bench-JSON aggregation (`bench_parse`), 3 covering the slot-aggregator merge + env-drift refusal.
