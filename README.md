# Benchmark Reproducibility Package

This repository contains the benchmark pipeline and result artefacts for the bachelor thesis:

> Evaluierung der llama.cpp-Quantisierung für die LLM-Inferenz auf Server-Class-CPUs

The final thesis evaluates llama.cpp GGUF quantization on a server-class CPU. The main analysis is a controlled sweep over Mistral-7B-Instruct-v0.3. Llama-3.1-8B-Instruct is included only as a diagnostic Q4_K_S comparison against Kurt (2026), not as a second full sweep or a successful quantitative replication.

## Scope of the Thesis Results

Main analysis:

- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Quantization set: `F16`, `Q3_K_S`, `Q4_K_S`, `Q5_0`, `Q6_K`, `Q8_0`
- Tasks: HellaSwag, GSM8K, IFEval, TruthfulQA-MC2
- Hardware: Google Cloud `c3-standard-192-metal`, Intel Xeon Platinum 8481C, 96 physical cores, 192 threads
- Backend: llama.cpp commit `be47fb9285779e900915bd8246eb9664110d4ba5`
- Evaluation harness: `lm-evaluation-harness` `0.4.9.2`

Diagnostic comparison:

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Quantization: `Q4_K_S`
- Purpose: diagnostic comparison against Kurt (2026), not a replication claim

## Repository Layout

| Path | Purpose |
|---|---|
| `configs/` | Quantization and task configuration. `quants.txt` is the final thesis quant set. |
| `scripts/` | Pipeline scripts for environment capture, model fetching, quantization, benchmarking, lm-eval, aggregation and bootstrap. |
| `results/` | Result artefacts, raw llama-bench JSONs, lm-eval outputs and per-slot metadata. |
| `results/bootstrap/` | Paired bootstrap confidence intervals used in the thesis. |
| `env/` | Environment metadata used by the pipeline. |
| `env/snapshot/` | Final system and reproducibility snapshot from the GCP benchmark instance. |
| `logs/` | Run logs from smoke tests, sweeps and spot-checks. |
| `tests/` | Unit tests for benchmark JSON parsing and result aggregation. |

Large model artefacts are not included in this repository. GGUF file sizes and SHA-256 hashes are documented in:

- `env/snapshot/gguf_sha256_all.txt`
- `env/snapshot/gguf_sizes_all.txt`

## Final Thesis Configuration

The final quantization set is stored in `configs/quants.txt`:

```text
F16
Q3_K_S
Q4_K_S
Q5_0
Q6_K
Q8_0
```

The final thesis tasks are configured in `configs/tasks.yaml` and interpreted as follows:

| Task | Setting | Reported metric |
|---|---|---|
| HellaSwag | 0-shot, no chat template | `acc_norm` |
| GSM8K | 5-shot, chat template, few-shot as multiturn | `exact_match` with `flexible-extract` |
| IFEval | 0-shot, chat template | mean of prompt-level strict, prompt-level loose, instruction-level strict and instruction-level loose accuracy |
| TruthfulQA-MC2 | 0-shot, no chat template | MC2 accuracy |

The thesis Avg score is the unweighted mean of these four task scores. MMLU is not included in the final thesis Avg.

## Final Result Artefacts

The Mistral main analysis uses these result directories:

```text
results/2026-06-17T13-33-45Z-mistralai_mistral-7b-instruct-v0.3-f16-ctx512/
results/2026-06-18T06-27-15Z-mistralai_mistral-7b-instruct-v0.3-q3_k_s-ctx512/
results/2026-06-18T18-05-51Z-mistralai_mistral-7b-instruct-v0.3-q4_k_s-ctx512/
results/2026-06-19T03-44-57Z-mistralai_mistral-7b-instruct-v0.3-q5_0-ctx512/
results/2026-06-19T16-31-42Z-mistralai_mistral-7b-instruct-v0.3-q6_k-ctx512/
results/2026-06-20T03-57-55Z-mistralai_mistral-7b-instruct-v0.3-q8_0-ctx512/
```

The Llama diagnostic comparison artefacts are stored in:

```text
results/2026-06-16T18-01-37Z-meta-llama_llama-3.1-8b-instruct-q4_k_s-ctx512/
```

This slot is a diagnostic comparison, not a second full sweep and not a successful quantitative replication of Kurt (2026). The original run started with `ctx=512`; HellaSwag, TruthfulQA-MC2 and llama-bench completed under that setting, but GSM8K failed because a prompt exceeded the 512-token context window. The generation-based tasks GSM8K and IFEval were therefore rerun with `LM_EVAL_N_CTX=4096`. The aggregated `results.json` in this slot combines these final artefacts and should be read with that context-window history in mind.

Against Kurt's reported Q4_K_S values, TruthfulQA-MC2 and the prompt-level IFEval components fall within the 2-SE comparison band, while GSM8K and HellaSwag do not. The slot is retained to document the diagnostic comparison and the sensitivity of direct cross-study comparisons to prompting, context and runtime details.

Bootstrap confidence intervals are stored in:

```text
results/bootstrap/Q3_K_S.json
results/bootstrap/Q4_K_S.json
results/bootstrap/Q5_0.json
results/bootstrap/Q6_K.json
results/bootstrap/Q8_0.json
```

Each result slot contains the following artefacts:

- `slot.json`: model, quantization, hashes, tasks, chat-template policy and environment hash
- `bench.json`: aggregated pp512 and tg128 medians and IQRs
- `bench_raw/*.json`: raw llama-bench outputs
- `lm_eval/*.json`: task-level lm-evaluation-harness outputs
- `lm_eval/_group_*_raw/*.jsonl`: per-item or per-prompt samples used for bootstrap
- `results.json`: aggregated slot result where available

## Environment Snapshot

The final environment snapshot is stored in `env/snapshot/`. It includes:

- CPU and NUMA topology: `lscpu.txt`, `lscpu_extended.txt`, `numactl_hardware.txt`
- Kernel, OS and microcode: `uname.txt`, `os_release.txt`, `microcode.txt`
- GCP machine metadata: `gcp_machine_type.txt`, `gcp_zone.txt`
- Transparent Huge Pages status: `thp_enabled.txt`, `hugepages_meminfo.txt`
- llama.cpp build information: `llama_cpp_commit.txt`, `llama_cpp_cmake_flags.txt`
- Python environment: `python_version.txt`, `pip_freeze.txt`
- GGUF model artefact references: `gguf_sha256_all.txt`, `gguf_sizes_all.txt`
- Reproduction command summary: `final_thesis_commands.txt`

These files are intended to document the system state used for the thesis, not to be consumed by the pipeline. The pipeline environment hash is based on the JSON files directly under `env/`.

## Reproduction Guide

The following steps describe how to reproduce the pipeline shape and the final thesis configuration. Full reproduction requires access to the relevant gated Hugging Face models and sufficient CPU time.

### 1. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
git -C ../llama.cpp checkout be47fb9285779e900915bd8246eb9664110d4ba5
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build --config Release -j
```

The final GCP build details are documented in `env/snapshot/llama_cpp_cmake_flags.txt`.

### 2. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The final package set used for the thesis is documented in `env/snapshot/pip_freeze.txt`.

### 3. Capture environment metadata

```bash
export LLAMA_CPP_DIR=../llama.cpp
./scripts/00_capture_env.sh
```

This writes `env/host.json`, `env/llama_cpp.json` and `env/lm_eval.json`. Existing thesis result slots were aggregated against the environment hash derived from those files.

### 4. Run the final Mistral sweep

The final thesis sweep is represented by the Mistral result directories listed above. A reproduction command for the final configuration is:

```bash
QUANTS_FILE=configs/quants.txt \
LM_EVAL_N_THREADS=96 \
LM_EVAL_BATCH=1 \
LM_EVAL_N_CTX=4096 \
BENCH_THREADS=96 \
BENCH_NUMA=distribute \
SKIP_CTX2048_PERF=1 \
numactl --interleave=all \
./scripts/sweep.sh mistralai/Mistral-7B-Instruct-v0.3 \
  hellaswag,gsm8k,ifeval,truthfulqa_mc2 \
  1
```

The authoritative per-slot settings are stored in each `slot.json` and in the corresponding lm-eval logs.

### 5. Run the Llama Q4_K_S diagnostic comparison

The final diagnostic comparison artefacts are stored under `results/2026-06-16T18-01-37Z-meta-llama_llama-3.1-8b-instruct-q4_k_s-ctx512/`. To reproduce a comparable run with Kurt's released GGUFs, set `KURT_GGUFS_DIR` to the directory containing those GGUF files:

The stored artefacts reflect the final task outputs: loglikelihood tasks at the slot context and generation tasks with `LM_EVAL_N_CTX=4096`, after the initial GSM8K context-window failure described above.

```bash
KURT_GGUFS_DIR=/path/to/kurt_ggufs \
LM_EVAL_N_THREADS=96 \
LM_EVAL_BATCH=1 \
LM_EVAL_N_CTX=4096 \
BENCH_THREADS=96 \
numactl --interleave=all \
./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct \
  Q4_K_S \
  512 \
  3 \
  hellaswag,gsm8k,ifeval,truthfulqa_mc2
```

### 6. Run paired bootstrap confidence intervals

```bash
python3 scripts/60_bootstrap.py \
  --baseline-slot results/2026-06-17T13-33-45Z-mistralai_mistral-7b-instruct-v0.3-f16-ctx512 \
  --quant-glob 'results/*-mistralai_mistral-7b-instruct-v0.3-*-ctx512' \
  --tasks hellaswag,gsm8k,ifeval,truthfulqa_mc2 \
  --b 10000 \
  --seed 0xBACE10AD \
  --out-dir results/bootstrap
```

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

The tests cover llama-bench JSON aggregation and result aggregation behaviour.
