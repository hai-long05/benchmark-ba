# GCP Full Llama-3.1-8B Sweep — Step-by-Step

This guide takes you from "I just provisioned a GCP `c3-standard-192`" to "the full Kurt-equivalent benchmark for Llama-3.1-8B-Instruct (14 quants × 6 tasks at ctx=512 + 14 perf-only slots at ctx=2048) is on disk in ~46 hours of wall-clock."

The pipeline at HEAD has all the changes you need: FP16 baseline path, single-process multi-task lm_eval, wikitext perplexity via `loglikelihood_rolling`, `tg256` for the long-context perf sweep, `PERF_ONLY=1` for ctx=2048 bench-only runs, and a `sweep.sh` driver with GNU parallel support.

The reference machine assumed throughout: **`c3-standard-192`** (192 vCPUs / 96 physical cores, 768 GiB RAM, Intel Xeon Platinum 8481C, AVX-512 + AMX-BF16). This is the same Sapphire-Rapids generation as Kurt's 8488C — direct comparability.

---

## 1. Pre-flight on your laptop (do this BEFORE provisioning the instance)

The instance costs ~€10/hour. Don't pay it to idle while you debug auth.

### 1.1 Hugging Face access (Llama-3.1 is gated)

1. Sign in to https://huggingface.co with the account you'll also use on GCP.
2. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct → **Agree and access repository**. Approval is usually automatic; can take up to 48 h.
3. At https://huggingface.co/settings/tokens create a token with **read** scope. Save it.
4. Verify locally:
   ```bash
   source .venv/bin/activate
   huggingface-cli login   # paste the token
   python3 -c "from huggingface_hub import hf_hub_download; \
     hf_hub_download('meta-llama/Llama-3.1-8B-Instruct', 'config.json'); print('OK')"
   ```
   `OK` = cleared. `awaiting review` = wait. Anything else = stop and fix before going to GCP.

### 1.2 Mac smoke (final sanity check before paying for GCP)

```bash
cd ~/benchmark   # or wherever the repo lives
source .venv/bin/activate
export LLAMA_CPP_DIR=../llama.cpp

LM_EVAL_LIMIT=20 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct F16 512 3 hellaswag,gsm8k
LM_EVAL_LIMIT=20 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 hellaswag,gsm8k
```

You're verifying that:
- Llama auth works (the F16 fetch is ~16 GiB)
- `results.json` for the F16 row shows `"quant": "F16", "bits_nominal": 16, "size_reduction_pct": 0.0`
- HellaSwag acc_norm is in roughly the right ballpark — Q4_K_S near 0.70, F16 slightly higher (broad CIs at limit=20)

If anything fails here, **debug on the Mac**. Don't bring known-broken state to the GCP node.

---

## 2. Provision the GCP instance

### 2.1 Create the VM

```bash
gcloud compute instances create benchmark-c3-192 \
  --zone=europe-west4-a \
  --machine-type=c3-standard-192 \
  --image-family=ubuntu-2404-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-ssd
```

500 GB is enough: FP16 GGUF (~16 GiB) + 13 quants (~70 GiB total) + lm_eval datasets (~3 GiB) + logs ≈ 100 GiB peak. SSD matters — quantization is single-threaded I/O-bound.

If `c3-standard-192` is unavailable in your region, try `c3-standard-176` or `us-central1`. Anything Sapphire-Rapids (c3-*) is fine. Avoid older c2 instances — you lose AMX-BF16 and your numbers diverge from Kurt's.

### 2.2 SSH in

```bash
gcloud compute ssh benchmark-c3-192 --zone=europe-west4-a
```

---

## 3. One-time setup on the box (~25 min)

### 3.1 System packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv python3-pip parallel jq tmux
```

`tmux` is critical — it's how the sweep survives SSH drops.

### 3.2 Set CPU governor to `performance`

GCP defaults to `powersave`. That costs ~15-20% throughput across the entire sweep. Fix once:

```bash
sudo cpupower frequency-set -g performance 2>/dev/null \
  || echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# expect: performance
```

This survives across reboots in some configurations. Re-check after any reboot.

### 3.3 Repo + llama.cpp

```bash
# Repo
git clone <your-repo-url> ~/benchmark   # or scp from your Mac
cd ~/benchmark

# llama.cpp at the same commit you used on Mac
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 1bfbdb134e4b983f7cbbde252d004483e31206a2

# Build with native CPU optimisation (picks up AVX-512 + AMX-BF16 automatically)
cmake -B build -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)
```

Verify the build picked up your CPU's vector extensions:

```bash
grep -oE 'avx512f|avx512bw|avx512_bf16|amx_bf16' /proc/cpuinfo | sort -u
# expect: amx_bf16, avx512_bf16, avx512bw, avx512f
```

If `amx_bf16` is missing, you're not on Sapphire-Rapids — verify the machine type.

### 3.4 Python venv

```bash
cd ~/benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# IFEval needs nltk's 'punkt' corpora at runtime (one-time download, ~3 MB)
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"
```

### 3.5 Hugging Face auth (paste the token from §1.1)

```bash
huggingface-cli login   # paste the token
huggingface-cli whoami  # confirm — should print your username
```

### 3.6 Capture environment metadata

```bash
export LLAMA_CPP_DIR=$HOME/llama.cpp
./scripts/00_capture_env.sh
cat env/host.json | python3 -m json.tool | head -10
```

You should see `"governor": "performance"`, `"cpu_cores": 192`, and the AVX-512 + AMX-BF16 flags. **From this moment on, every slot is hashed against this `env/`. Don't change governor or rebuild llama.cpp without re-running this and accepting that prior slots become un-mergeable.**

---

## 4. GCP smoke test (~10 min, do this BEFORE the full sweep)

You're verifying the GCP box agrees with the Mac on Q4_K_S HellaSwag at limit=200, and that the build actually uses AMX-BF16 (~5x faster than M1 expectations).

### 4.1 Start a tmux session (so the run survives SSH drops)

```bash
tmux new -s smoke
```

Inside tmux:

```bash
cd ~/benchmark
source .venv/bin/activate
export LLAMA_CPP_DIR=$HOME/llama.cpp
mkdir -p logs

LM_EVAL_N_THREADS=88 \
LM_EVAL_BATCH=1 \
LM_EVAL_N_CTX=2048 \
LM_EVAL_LIMIT=200 \
  ./scripts/run_slot.sh \
    meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 \
    hellaswag,gsm8k,ifeval,truthfulqa_mc2 \
  2>&1 | tee logs/smoke_$(date +%Y%m%d_%H%M).log
```

`LM_EVAL_N_THREADS=88` pins llama-cpp-python to 88 physical cores (out of 96), leaving 8 for OS + I/O + tqdm. Single-slot smoke maximises per-slot throughput.

### 4.2 Detach and let it run

`Ctrl-b` then `d` — you're back at the regular shell; the smoke keeps running.

### 4.3 Reattach later

```bash
tmux attach -t smoke
```

### 4.4 Pass criteria (n=200, ±3 percentage points)

```bash
RESULT=$(ls -td results/*-meta-llama_llama-3.1-8b-instruct-q4_k_s-ctx512/results.json | head -1)
python3 <<PY
import json
d = json.load(open("$RESULT"))
print(f"pp512: {d['pp']['median']:.1f} tok/s   (target: > 350 — Kurt: 92.5 on Mac, you have AMX)")
print(f"tg128: {d['tg']['median']:.2f} tok/s   (target: > 30)")
print(f"hellaswag acc_norm:    {d['tasks']['hellaswag']['acc_norm']:.4f}    (target: 0.69-0.76; Kurt full: 0.7279)")
print(f"gsm8k flex-extract:    {d['tasks']['gsm8k']['exact_match_flexible_extract']:.4f}    (target: 0.74-0.81; Kurt full: 0.7733)")
print(f"truthfulqa_mc2:        {d['tasks']['truthfulqa_mc2']['acc']:.4f}    (target: 0.50-0.57; Kurt full: 0.5340)")
PY
```

If **all five** pass, you're cleared for the full sweep. **If any miss, stop** — re-running 14 broken slots wastes ~46 hours of GCP time. Tell me the specific failure.

### 4.5 Pre-fetch all 14 quants while smoke runs

In a second SSH session (no tmux needed — it's just I/O):

```bash
cd ~/benchmark
source .venv/bin/activate
export LLAMA_CPP_DIR=$HOME/llama.cpp

# F16 already fetched by the smoke. Do the other 13 in the background.
SAFE=meta-llama_llama-3.1-8b-instruct
for SCHEME in Q3_K_S Q3_K_M Q3_K_L Q4_0 Q4_1 Q4_K_M Q5_0 Q5_1 Q5_K_S Q5_K_M Q6_K Q8_0; do
  ./scripts/20_quantize.sh "models/${SAFE}-fp16.gguf" "$SCHEME"
done
```

(Q4_K_S already done by the smoke.) This hammers one core for ~25 min in parallel with the smoke's 88 cores. Saves 25 min off the real sweep's warm-up phase.

---

## 5. Run the full Llama-3.1-8B sweep

### 5.1 Concurrency layout for `c3-standard-192`

96 physical cores. Three layouts; pick by goal:

| Layout | Concurrent slots | n_threads/slot | RAM/slot × N | Best for |
|---|---|---|---|---|
| **8 × 12** | 8 | 12 | ~80 GB | **Recommended** — overall throughput |
| 12 × 8 | 12 | 8 | ~120 GB | Maximises tail-end completion (fewer slots stragglers at the end) |
| 4 × 24 | 4 | 24 | ~40 GB | Single-slot wall-clock matters more |

768 GiB RAM means you're nowhere near memory-bound — pick by core sharing efficiency. **Use 8 × 12.**

### 5.2 Start the sweep in tmux

```bash
tmux new -s sweep
```

Inside tmux:

```bash
cd ~/benchmark
source .venv/bin/activate
export LLAMA_CPP_DIR=$HOME/llama.cpp
mkdir -p logs

# Sanity: env/ matches the post-governor-change state. If env/ is from before
# §3.2 you'll get drift errors at aggregate time. Re-run if unsure:
./scripts/00_capture_env.sh

LM_EVAL_N_THREADS=12 \
LM_EVAL_BATCH=1 \
LM_EVAL_N_CTX=2048 \
  ./scripts/sweep.sh meta-llama/Llama-3.1-8B-Instruct \
    hellaswag,gsm8k,ifeval,mmlu,truthfulqa_mc2,wikitext \
    8 \
  2>&1 | tee logs/sweep_llama_$(date +%Y%m%d_%H%M).log
```

Detach: `Ctrl-b` then `d`. Reattach: `tmux attach -t sweep`.

### 5.3 What `sweep.sh` does

1. **Warm-up (sequential, ~5 min if §4.5 was done):** confirms FP16 GGUF + 13 quants exist; runs whichever are missing. F16 reuses the FP16 GGUF directly — no quantize step.
2. **Accuracy phase (parallel-8, ~46 h):** runs all 14 (model, quant, ctx=512) slots, each running all 6 lm_eval tasks in **one Python process per fewshot group** (saves model-load overhead). Slots are independent; GNU parallel queues across 8 workers.
3. **Performance phase (parallel-8, ~2 min):** 14 slots at ctx=2048 with `PERF_ONLY=1`, only `30_bench` runs. Uses `tg256` automatically because ctx ≥ 2048.

### 5.4 Wall-clock estimate

| Phase | Sequential | Parallel-8 |
|---|---|---|
| Warm-up | 30 min | 30 min |
| Accuracy (14 slots × 6 tasks) | ~370 h | **~46 h** |
| Perf at ctx=2048 | 14 min | 2 min |
| **Total** | ~370 h | **~46–48 h ≈ 2 days** |

MMLU dominates: ~14h × 14 slots ÷ 8 workers ≈ 25 of those 46 hours. If you need to finish in ~24h, drop MMLU from this sweep and run it as a separate smaller job later. For the Kurt anchor model **keep MMLU in** — Kurt reports it and you'll need it for the Avg.

---

## 6. Live monitoring

In a second SSH session (no tmux needed):

```bash
# Slot completion count
watch -n 60 '
  cd ~/benchmark
  acc=$(ls -d results/*-meta-llama_llama-3.1-8b-instruct-*-ctx512/ 2>/dev/null | wc -l)
  perf=$(ls -d results/*-meta-llama_llama-3.1-8b-instruct-*-ctx2048/ 2>/dev/null | wc -l)
  printf "accuracy slots: %s/14   perf slots: %s/14\n" "$acc" "$perf"
'

# Live log tail
tail -f ~/benchmark/logs/sweep_llama_*.log

# Memory + CPU pressure
htop   # top of htop should show ~88 cores at 100% during accuracy phase
```

---

## 7. Recovery if a slot dies

The pipeline is idempotent. Re-run the same `sweep.sh` command:

- The fetch step skips if the FP16 GGUF SHA matches.
- The quantize step skips if `quantized/<gguf>.sha256` exists and matches.
- A new run-id timestamp directory is created **only** if the previous slot was incomplete. **Already-complete `lm_eval/<task>.json` files are skipped per task** — if MMLU died but HellaSwag completed, only MMLU re-runs.
- The bench step re-runs every time (it's the variable measurement).

If a single slot's `results.json` is missing or corrupt after the sweep, just re-run sweep.sh and only the failed slot's missing tasks re-execute.

---

## 8. Validate against Kurt's Table 2 before celebrating

```bash
cd ~/benchmark
for SCHEME in F16 Q3_K_S Q4_K_S Q5_0 Q8_0; do
  R=$(ls -td results/*-meta-llama_llama-3.1-8b-instruct-${SCHEME,,}-ctx512/results.json 2>/dev/null | head -1)
  [ -z "$R" ] && { printf "MISSING: %s\n" "$SCHEME"; continue; }
  python3 -c "
import json
d = json.load(open('$R'))
ha = d['tasks'].get('hellaswag', {}).get('acc_norm', 0)
g  = d['tasks'].get('gsm8k', {}).get('exact_match_flexible_extract', 0)
m  = d['tasks'].get('mmlu', {}).get('acc', 0)
print(f\"{d['quant']:8s} HSwag={ha:.4f}  GSM8K-FE={g:.4f}  MMLU={m:.4f}\")
"
done
```

Reference (Kurt 2026, Table 2):

| Quant | HSwag | GSM8K-FE | MMLU |
|---|---|---|---|
| F16 | 0.7251 | 0.7763 | 0.6350 |
| Q3_K_S | 0.7187 | 0.6831 | 0.5931 |
| Q4_K_S | 0.7279 | 0.7733 | 0.6206 |
| Q5_0 | 0.7263 | 0.7908 | 0.6318 |
| Q8_0 | 0.7252 | 0.7748 | 0.6343 |

**Pass criteria:** every value within 1 percentage point of Kurt's. If anything's off by >2 points, that slot is suspect — re-run only that one with `./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct <SCHEME> 512 3 <task>` before trusting the sweep.

---

## 9. Pull data back, shut down

```bash
# From your Mac
mkdir -p ~/gcp-results
gcloud compute scp --recurse \
  benchmark-c3-192:~/benchmark/results \
  benchmark-c3-192:~/benchmark/env \
  benchmark-c3-192:~/benchmark/logs \
  ~/gcp-results/ --zone=europe-west4-a

# Stop (keeps disk for retries — costs ~€0.04/GiB/month while stopped)
gcloud compute instances stop benchmark-c3-192 --zone=europe-west4-a

# Or delete (only after you've verified the data on your Mac)
gcloud compute instances delete benchmark-c3-192 --zone=europe-west4-a
```

Keep the instance stopped — not deleted — for at least a week in case you find something needs re-running during analysis.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `tmux: command not found` | Not installed | `sudo apt-get install -y tmux` |
| `tmux` exits immediately | Stale socket | `rm -rf /tmp/tmux-*; tmux new -s sweep` |
| `403 Forbidden` on Llama download | Meta gating not approved | Wait for the email; Mistral & Qwen are open in the meantime |
| `OOM` during quantization | Disk full | Confirm 500 GB SSD; full quant set ≈ 100 GiB |
| `bench.json` IQR warnings on every slot | Background processes / governor changed mid-run | Re-run §3.2; verify `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` |
| Numbers diverge from Mac smoke by >1 pt | Different llama.cpp commit or build flags | `git -C $LLAMA_CPP_DIR rev-parse HEAD` should match `env/llama_cpp.json` |
| `--limit` accidentally still set | Forgot to unset env var | `unset LM_EVAL_LIMIT` before kicking the sweep |
| `tg=128 tok/s` at ctx=2048 in bench output | Old `30_bench.sh` | `grep TG_N scripts/30_bench.sh` should show the conditional logic |
| Aggregate refuses with `env drift` | governor/llama.cpp/lm_eval changed since slot.json was written | Re-run that slot with `./scripts/run_slot.sh ...` |
| `cpupower: command not found` | Package missing | `sudo apt-get install -y linux-tools-common linux-tools-$(uname -r)` or use the `echo performance` fallback in §3.2 |
| `caching_allocator_warmup ... CUDA enabled` errors | torch trying to initialise CUDA on CPU-only box | Make sure you're using `--model gguf_local` (the project's path), not `--model hf` |

---

## 11. Quick command reference

```bash
# One slot
LM_EVAL_LIMIT=200 LM_EVAL_N_THREADS=88 \
  ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 hellaswag

# One slot — perf only at ctx=2048
PERF_ONLY=1 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 2048 3

# Full sweep (Llama, all 14 configs, all 6 tasks, parallel-8)
LM_EVAL_N_THREADS=12 \
  ./scripts/sweep.sh meta-llama/Llama-3.1-8B-Instruct \
    hellaswag,gsm8k,ifeval,mmlu,truthfulqa_mc2,wikitext 8

# Full sweep without MMLU (cross-family models — half the time)
LM_EVAL_N_THREADS=12 \
  ./scripts/sweep.sh Qwen/Qwen3-8B \
    hellaswag,gsm8k,ifeval,truthfulqa_mc2,wikitext 8

# Tmux essentials
tmux new -s sweep         # start
# Ctrl-b d                # detach
tmux attach -t sweep      # reattach
tmux ls                   # list
tmux kill-session -t sweep  # kill (only when done)
```

---

## 12. After Llama finishes

Same sweep command for the cross-family models, no MMLU (saves ~50% wall-clock per model — scope cut documented in Limitations):

```bash
# Mistral — open license, no auth needed
LM_EVAL_N_THREADS=12 \
  ./scripts/sweep.sh mistralai/Mistral-7B-Instruct-v0.3 \
    hellaswag,gsm8k,ifeval,truthfulqa_mc2,wikitext 8

# Qwen — confirm the id with the user (Qwen3-8B was the proposed substitution
# for the missing Qwen3.5-9B / Qwen30-9B; check before launching)
LM_EVAL_N_THREADS=12 \
  ./scripts/sweep.sh Qwen/Qwen3-8B \
    hellaswag,gsm8k,ifeval,truthfulqa_mc2,wikitext 8
```

Each ~24 h parallel-8. Total cross-family: ~48 h ≈ 2 days. Combined with Llama's ~48 h: **the full thesis benchmark fits in ~4 days of GCP wall-clock**, well under your one-month rental budget.
