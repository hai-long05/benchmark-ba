# Benchmark — llama.cpp GGUF quantization sweep

Bachelor thesis benchmark, replicates Kurt (2026, arXiv:2601.14277) cross-family on Qwen3-4B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct.

Spec: `docs/superpowers/specs/2026-06-11-benchmark-design.md`.

## Smoke test (Mac)

```
./scripts/00_capture_env.sh
./scripts/run_slot.sh Qwen/Qwen3-0.6B Q4_K_S 512
cat results/*/results.json | jq .
```

Expected wall-clock: < 30 min.

## Dependencies

- `llama.cpp` checkout: scripts read `$LLAMA_CPP_DIR` (default `../llama.cpp`).
  Build with `cmake -B build -DGGML_METAL=OFF -DGGML_BLAS=OFF && cmake --build build -j`.
- Python 3.11+: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
