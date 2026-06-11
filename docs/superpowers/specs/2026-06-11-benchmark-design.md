# Benchmark Setup — Design

**Date:** 2026-06-11
**Author:** Hai Long Do Pham
**Thesis:** Performance- und genauigkeitsorientierte Quantisierungswahl für 4–8 B-LLMs auf Server-Class-CPU
**Supervisors:** Prof. Dr. Andreas Schmietendorf (HWR Berlin), Kolya Opahle (SAP)
**Submission:** 2026-06-07
**Anchor paper:** Kurt, U. (2026). *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct*. arXiv:2601.14277.

## 1. Purpose

Set up the benchmark infrastructure that will, in a later iteration, sweep 3 model families × 14 quantization configs × 2 context lengths = 84 configuration slots (each ≥ 3 repeats) on a server-class x86 CPU.

This document specifies the **minimal end-to-end pipeline** — one model × one quant × one context length × one accuracy task — which proves the pipeline shape end-to-end and forms the foundation that the sweep, paired-bootstrap statistics, Pareto analysis, and reproducibility package later compose on top of.

**Explicitly out of scope for this iteration:**
- The full sweep (84 slots × ≥ 3 repeats).
- Paired-bootstrap CIs on item level for Quant vs FP16 differences.
- WikiText-2-PPL ↔ Avg-Score correlation analysis.
- Pareto-plot generation.
- Cross-family comparison tables.

These each presuppose the per-slot pipeline this document defines.

## 2. Final Targets (from the exposé, §4.3 + §4.5)

| Item | Value |
|---|---|
| Models | Qwen3-4B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct |
| Quant configs | FP16 baseline + 13 GGUF schemes: Q3_K_S/M/L, Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0 |
| Context lengths | 512 (Kurt-Anker), 2 048 (zweite, praxisnahe Kontextlänge) |
| Accuracy tasks | GSM8K v3 (5-shot, FE), HellaSwag v1 (0-shot, acc_norm), IFEval v4 (0-shot, 4 sub-metrics), MMLU v2 (0-shot, acc), TruthfulQA v3 (0-shot, MC2) |
| Quality (intrinsic) | WikiText-2 perplexity (corpus-aggregated, separate from item-bootstrap) |
| Performance | llama-bench → pp512, tg128 (and tg2048 for the second ctx); ≥ 3 repeats; report median + IQR |
| Decoding | Greedy (`temperature=0`) |
| Tools | llama.cpp (pinned commit), lm-evaluation-harness v0.4.9.2 |
| Hardware (real run) | Server-class x86 CPU on GCP, exact spec captured before measurements (per exposé §4.4) |

## 3. Architecture

The benchmark is a **filesystem-as-database** pipeline. Each stage reads files, writes files, exits. State lives on disk, not in memory and not in a process. Two consequences fall out of this:

1. A half-finished sweep is recoverable — re-running picks up the missing files only.
2. The result tree directly serves as the §4.6 reproducibility package (lm-eval version, llama.cpp commit, GGUF SHA256, hardware spec, raw logs).

```
                ┌─ env/host.json        (CPU, kernel, BIOS, AVX-512, RAM, governor)
capture-env  ─→ ├─ env/llama_cpp.json   (commit + build flags)
                └─ env/lm_eval.json     (version)

fetch-model   → models/<model>-fp16.gguf  (+ .sha256, + .meta.json with HF revision + license)

quantize      → quantized/<model>-<scheme>.gguf  (+ .sha256, + quant_time_s)

run-slot orchestrates one (model, quant, ctx) row:
   ├─ bench         → results/<run-id>/bench.json   (pp/tg, N repeats, median + IQR)
   ├─ lm-eval × T   → results/<run-id>/lm_eval/<task>.json   (incl. -log_samples)
   └─ aggregate     → results/<run-id>/results.json (merged single row, Kurt-table schema)
```

Five processes, four data interfaces, one orchestrator. No daemons, no databases, no DAG runner.

## 4. Directory Layout

```
benchmark/
├── env/                    # captured environment, regenerated per session
│   ├── host.json
│   ├── llama_cpp.json
│   └── lm_eval.json
├── models/                 # FP16 GGUF inputs (gitignored, large)
├── quantized/              # produced by quantize.sh
├── results/<run-id>/       # one timestamped directory per slot
│   ├── slot.json
│   ├── bench.json
│   ├── lm_eval/<task>.json
│   └── results.json
├── scripts/
│   ├── 00_capture_env.sh
│   ├── 10_fetch_model.sh
│   ├── 20_quantize.sh
│   ├── 30_bench.sh
│   ├── 40_lm_eval.sh
│   ├── 50_aggregate.py
│   └── run_slot.sh
├── configs/
│   ├── quants.txt          # 13 schemes (one per line) — for the future sweep
│   └── tasks.yaml          # task names + version + shot counts pinned
├── docs/superpowers/specs/
│   └── 2026-06-11-benchmark-design.md   # this file
└── README.md
```

`.gitignore` excludes `models/`, `quantized/`, and `results/`. The repository tracks scripts, configs, and docs only.

## 5. Components

Each script does one thing. Inputs and outputs are paths. Non-zero exit on any failure.

### 5.1 `00_capture_env.sh`

- **Input:** none.
- **Output:** `env/host.json`, `env/llama_cpp.json`, `env/lm_eval.json`.
- `host.json` collects: `lscpu` output (or `sysctl -a` on macOS), `uname -a`, total RAM, AVX flags, current CPU governor, BIOS string when readable. macOS records that BIOS is N/A; the production run on GCP records the real BIOS.
- `llama_cpp.json` records `git rev-parse HEAD` of the local llama.cpp checkout plus the cmake flags used to build it.
- `lm_eval.json` records `lm_eval --version` (must equal `0.4.9.2` per exposé §4.4 — the script asserts this and exits non-zero on mismatch).
- Idempotent: re-running overwrites cleanly.

### 5.2 `10_fetch_model.sh <hf-id>`

- **Input:** Hugging Face model identifier (e.g. `Qwen/Qwen3-0.6B`).
- **Output:** `models/<safe-name>-fp16.gguf` + `.sha256` + `.meta.json` (HF revision, license, original source format).
- Pins the HF revision (no `main`); converts safetensors → GGUF FP16 using `llama.cpp/convert_hf_to_gguf.py` when needed.
- Skips download/conversion if output exists and SHA matches.

### 5.3 `20_quantize.sh <fp16.gguf> <SCHEME>`

- **Input:** FP16 GGUF path, scheme name (e.g. `Q4_K_S`).
- **Output:** `quantized/<safe-name>-<scheme>.gguf` + `.sha256` + side-car `<gguf>.quant.json` containing `quant_time_s`, `size_mib`, `size_reduction_pct` (vs FP16 input).
- Wraps `llama-quantize <fp16.gguf> <out.gguf> <SCHEME>`.
- Skips if output exists and SHA matches.

### 5.4 `30_bench.sh <gguf> <ctx> [<repeats>]`

- **Input:** quantized GGUF, ctx ∈ {512, 2 048}, default repeats = 3.
- **Output:** `bench.json` with raw N-repeat samples plus aggregated median + IQR for `pp` and `tg`.
- Wraps `llama-bench --output json` with `-p <ctx>` and `-n 128` for tg, run `<repeats>` times. Aggregation is plain numpy (median, IQR) — no statistical machinery this iteration.
- `<repeats>` must be ≥ 2 for IQR to be defined; with `<repeats> = 1` the IQR field is `null` and a warning is recorded. The exposé requires ≥ 3 for the real run.
- If IQR > 5 % of median, append a warning string to `bench.json.warnings` (does not fail the slot).

### 5.5 `40_lm_eval.sh <gguf> <task>`

- **Input:** quantized GGUF, task name (one of: `gsm8k` (5-shot, flexible-extract per Kurt), `hellaswag`, `ifeval`, `mmlu`, `truthfulqa_mc2`, `wikitext`). Final task IDs in `configs/tasks.yaml` are pinned to the exact lm-eval-harness v0.4.9.2 task names; the smoke test only uses `hellaswag` so this is finalized once during the first real-run setup.
- **Output:** `lm_eval/<task>.json` (the harness's `results.json`) plus the harness's per-sample log file.
- Calls `lm_eval --model gguf --model_args pretrained=<gguf>,…` with `--log_samples --output_path …`. Greedy decoding (`temperature=0`).
- Per-task version + shot count comes from `configs/tasks.yaml` (pinned to the exposé § 4.4 versions).
- Per-task standard-error from the harness is preserved verbatim (no aggregation here).

### 5.6 `50_aggregate.py <slot-dir>`

- **Input:** a slot directory containing `slot.json`, `bench.json`, `lm_eval/`.
- **Output:** `results.json` — a single row whose schema mirrors Kurt's Table 2 + Table 3:

```json
{
  "run_id": "...",
  "model": "Qwen3-0.6B",
  "quant": "Q4_K_S",
  "bits_nominal": 4,
  "ctx": 512,
  "size_mib": 412.3,
  "size_reduction_pct": 65.4,
  "quant_time_s": 4.1,
  "pp": {"median": 92.5, "iqr": 1.5, "samples": [...], "ctx": 512},
  "tg": {"median": 4.65, "iqr": 0.15, "samples": [...]},
  "tasks": {
    "hellaswag": {"acc_norm": 0.7187, "acc_norm_stderr": 0.0045}
  },
  "ppl_wikitext2": null,
  "env": {"host_id": "...", "llama_cpp_commit": "...", "lm_eval_version": "0.4.9.2"},
  "warnings": []
}
```

- Refuses to merge if `slot.json.host_id` ≠ current `env/host.json` SHA (surfaces drift loudly).

### 5.7 `run_slot.sh <model-hf-id> <quant> <ctx>`

- Composes 10 → 20 → 30 → 40 → 50 for one (model, quant, ctx) triple.
- Writes `slot.json` first (defines the row identity), then runs each downstream stage if its output file is missing or invalid.
- Returns the failed stage name on non-zero exit. Re-running resumes at the failed stage.

## 6. Data Flow & Idempotency

Every stage's output filename is deterministic from its inputs. A re-run that finds a valid output (file exists + SHA matches) skips the work. Three properties follow:

1. The Mac smoke test and the GCP real run write to the **same** layout — only `env/` content differs.
2. Deleting one corrupt artefact (e.g. `lm_eval/mmlu.json`) and re-running `run_slot.sh` re-executes only that artefact.
3. The future 84-slot sweep is a triple `for` loop around `run_slot.sh` — no extra orchestration logic needed.

`run_slot.sh` records the SHA256 of the `env/` directory it ran under in `slot.json`. The aggregator refuses to merge slots whose env hash disagrees with the current `env/` — so accidental cross-hardware-state aggregation surfaces as a hard error rather than a silent contamination.

## 7. Error Handling

| Failure | Behaviour |
|---|---|
| Tool failure (OOM, hang, corrupt GGUF) | Stage script exits non-zero; partial output deleted; `run_slot.sh` reports the failed stage. Re-run resumes there. |
| Environment drift (env hash mismatch) | Aggregator refuses to merge. Operator must either re-run the slot under current env or accept the mismatch explicitly. |
| Bench noise (IQR > 5 % of median) | Warning appended to `bench.json.warnings` and propagated into `results.json.warnings`. Slot still completes. |

Out of scope this iteration (deferred to the sweep + analysis layer):

- Paired-bootstrap CIs on item level for Quant − FP16 differences (§ 4.4 exposé) — requires the FP16 baseline slot and the comparison slot, not a single slot.
- WikiText-2-PPL ↔ Avg-Score correlation (Pearson and Spearman, n = 13) — requires all 13 quant configs of one model.
- Pareto plots — require the full per-model sweep.

These compose on top of `results/*/results.json` after the per-slot pipeline is proven.

## 8. Smoke Test Plan

The minimal end-to-end run that proves this pipeline works, on the local Mac:

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` (~1.2 GiB FP16, fits Apple Silicon) |
| Quant | `Q4_K_S` (one of Kurt's quality-Pareto points) |
| Context | 512 |
| Bench repeats | 3 |
| lm-eval tasks | `hellaswag` only (~1 k items, ~minutes on CPU) |
| Expected wall-clock | < 30 min |

**Pass criteria:** `results/<run-id>/results.json` exists and contains:

- `quant.gguf_sha256`
- `pp.median`, `pp.iqr`, `pp.samples` (length = 3)
- `tg.median`, `tg.iqr`, `tg.samples` (length = 3)
- `tasks.hellaswag.acc_norm` and `acc_norm_stderr`
- `env.host_id`, `env.llama_cpp_commit`, `env.lm_eval_version == "0.4.9.2"`

The smoke test does **not** prove Kurt's numbers replicate. It proves the pipeline shape is sound. Replication happens later on GCP with the real models.

## 9. Portability: Mac Smoke vs GCP Real Run

| Concern | Mac smoke | GCP real run |
|---|---|---|
| llama.cpp build | `cmake -DGGML_METAL=OFF` (CPU path, matches Linux build) | `cmake -DGGML_BLAS=OFF -DGGML_NATIVE=ON` with AVX-512/BF16 enabled (per Kurt) |
| BIOS / governor | recorded as N/A | recorded; governor pinned to `performance` |
| lm-eval version | 0.4.9.2 | 0.4.9.2 |
| Directory layout | identical | identical |
| Result schema | identical | identical |

The CPU-only llama.cpp build on the Mac means the Mac smoke run uses the same inference path as the GCP run. This is deliberate — Apple Silicon Metal would be a different code path and would not validate the pipeline that runs in production.

## 10. Open Questions Deferred to the Next Iteration

- Exact GCP machine type (the exposé fixes this before measurements; the pipeline does not depend on it).
- llama.cpp commit pin — picked at the moment we run `00_capture_env.sh` for the first real session.
- Sweep driver script — trivial extension; written when this iteration is signed off.
- Statistical layer (paired bootstrap, correlation, Pareto) — designed in its own spec.

## 11. Acceptance for This Iteration

The iteration is done when:

1. All scripts and configs in §4 exist and are committed.
2. The smoke test in §8 runs to completion on the Mac and the pass-criteria are met.
3. `README.md` documents how to run the smoke test in three commands.
