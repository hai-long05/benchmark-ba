# Running the Full Llama-3.1-8B Sweep on GCP — Step-by-Step

This guide takes you from "I have a `c3-standard-176` instance" to "I have all 14 quants × all 6 tasks of Kurt's Llama-3.1-8B-Instruct benchmark on disk in ~24-36 hours of wall-clock."

The pipeline at HEAD already has all the changes you need: FP16 baseline path, single-process multi-task lm_eval, wikitext perplexity via `loglikelihood_rolling`, `tg2048` for the long-context perf sweep, and a `sweep.sh` driver with GNU parallel support.

---

## 1. Pre-flight (do this BEFORE you provision the instance)

The instance costs ~€10/hour. Don't pay it to idle while you debug.

### Hugging Face access (manual review, can take 24–48 h)

1. Sign in to https://huggingface.co with the account you'll use on GCP.
2. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and click **Agree and access repository**.
3. At https://huggingface.co/settings/tokens create a token with **read** scope; save it somewhere safe.
4. Verify locally on your Mac:
   ```bash
   source .venv/bin/activate
   huggingface-cli login   # paste the token
   python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('meta-llama/Llama-3.1-8B-Instruct', 'config.json'); print('OK')"
   ```
   If you see "awaiting review", wait. If it prints `OK`, you're cleared.

### Smoke-test the local pipeline

The Mac is your last sanity check before the GCP node. With Llama access cleared:

```bash
cd /Users/I589258/Documents/benchmark
source .venv/bin/activate
export LLAMA_CPP_DIR=../llama.cpp

# 30-min smoke: limit=20, hellaswag + gsm8k, FP16 + Q4_K_S
LM_EVAL_LIMIT=20 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct F16 512 3 hellaswag,gsm8k
LM_EVAL_LIMIT=20 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 hellaswag,gsm8k
```

Each takes ~25 minutes on M1 Pro. You're checking that:
- Llama auth works (the F16 fetch is ~16 GiB)
- `results.json` shows `quant: F16, bits_nominal: 16, size_reduction_pct: 0` for the F16 row
- HellaSwag acc_norm is in the right ballpark (FP16 should be ~0.72, Q4_K_S ~0.70 at limit=20 with broad CIs)

If anything fails here, debug on the Mac. The GCP box won't fix it.

---

## 2. Provision the GCP instance

### Choose `c3-standard-176`

Closest match to Kurt's hardware (96-core Xeon Platinum 8488C, AVX-512 + BF16). Uses Sapphire-Rapids generation Xeons; same CPU family.

```bash
gcloud compute instances create benchmark-c3-176 \
  --zone=europe-west4-a \
  --machine-type=c3-standard-176 \
  --image-family=ubuntu-2404-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-ssd
```

500 GB disk is enough: 14 quants × 8 GiB max + FP16 (~16 GiB) + lm_eval datasets (~3 GiB) ≈ 130 GiB peak.

Region: pick `europe-west4` if you want low latency from Berlin; `us-central1` typically has best instance availability if `c3-standard-176` is constrained in your region.

### SSH in

```bash
gcloud compute ssh benchmark-c3-176 --zone=europe-west4-a
```

---

## 3. Set up the box (~25 minutes)

```bash
# System packages
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv python3-pip parallel jq

# Repo
git clone <your-repo-url> ~/benchmark   # or scp the directory from your Mac
cd ~/benchmark

# llama.cpp — same commit you used on Mac (recorded in env/llama_cpp.json)
cd ..
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 1bfbdb134e4b983f7cbbde252d004483e31206a2   # commit you've been using
cmake -B build -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j

# Python venv
cd ~/benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# HF auth
huggingface-cli login   # paste your token

# Capture environment metadata
export LLAMA_CPP_DIR=$HOME/llama.cpp
./scripts/00_capture_env.sh
```

Verify the build picked up AVX-512 + BF16:

```bash
grep -E 'AVX512|BF16|amx' /proc/cpuinfo | head -1
$LLAMA_CPP_DIR/build/bin/llama-cli --version 2>&1 | head -3
```

You should see `avx512f`, `avx512bw`, `bf16`, `amx_bf16` in the cpuinfo output.

---

## 4. Quick GCP smoke (~10 minutes)

Do this before the full sweep. You're verifying the GCP box agrees with Mac on Q4_K_S HellaSwag at limit=200.

```bash
LM_EVAL_LIMIT=200 LM_EVAL_N_THREADS=88 \
  ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 hellaswag
```

Note `LM_EVAL_N_THREADS=88` — pins llama-cpp-python to 88 physical cores (the c3 has 88 hyperthreads × 2 = 176 vCPUs, but llama.cpp scales with physical cores, not threads). Adjust to whatever `nproc --all` divided by 2 reports.

Expected: ~5–8 minutes wall-clock. Compare your acc_norm to the Mac smoke — they should agree within ~1 percentage point. If they diverge by more, stop and investigate.

---

## 5. Run the full Llama-3.1-8B sweep

```bash
cd ~/benchmark
source .venv/bin/activate
export LLAMA_CPP_DIR=$HOME/llama.cpp

# In a tmux session so the run survives SSH drops
tmux new -s sweep
# inside tmux:

# 8 parallel slots × 11 cores each = 88 cores. Memory used: ~80 GB out of ~700 GB.
LM_EVAL_N_THREADS=11 \
  ./scripts/sweep.sh meta-llama/Llama-3.1-8B-Instruct \
    hellaswag,gsm8k,ifeval,mmlu,truthfulqa_mc2,wikitext \
    8 \
  2>&1 | tee logs/sweep_llama_$(date +%Y%m%d_%H%M).log
```

What this does:
1. **Warm-up (sequential, ~30 min):** fetch FP16 GGUF once, quantize all 13 quants. F16 reuses the FP16 GGUF — no quantize step.
2. **Accuracy phase (parallel-8, ~24–30h wall-clock):** runs all 14 (model, quant, ctx=512) slots, each running all 6 lm_eval tasks in one Python process. Slots are independent; GNU parallel queues them across 8 workers.
3. **Performance phase (parallel-8, ~30 min):** 14 slots at ctx=2048 with `PERF_ONLY=1`, only `30_bench` runs (no lm_eval), measures `pp2048` and `tg256`.

Detach tmux with `Ctrl-b d`. Reattach later with `tmux attach -t sweep`.

### Wall-clock estimate breakdown

| Phase | Items | Sequential | Parallel-8 |
|---|---|---|---|
| Warm-up (fetch + 13× quantize) | 14 | ~30 min | ~30 min |
| Accuracy: 14 slots × 6 tasks | 14 slots | ~14 × 30h = 420h | **~52h** |
| Perf at ctx=2048 | 14 slots | ~14 × 1min = 14min | ~2 min |
| **Total** | | **~440 h** | **~53 h ≈ 2.2 days** |

The accuracy estimate assumes each slot at full sample count takes ~30 hours on a single 11-core worker (extrapolating from M1 Pro's 26 min for 200 hellaswag items × ~50× scale, ÷ 6 cores' improvement over M1's 8). MMLU is the long pole — ~14h of those 30h per slot. If you skip MMLU, drop to ~16h/slot sequentially, ~28h wall-clock parallel.

### Live monitoring

```bash
# Per-slot progress
ls -la results/ | tail -20

# Watch the live log
tail -f logs/sweep_llama_*.log

# How many slots completed?
ls -d results/*-meta-llama_llama-3.1-8b-instruct-*-ctx512/ | wc -l   # target: 14
ls -d results/*-meta-llama_llama-3.1-8b-instruct-*-ctx2048/ | wc -l  # target: 14
```

### If something dies

The pipeline is idempotent. Re-running `sweep.sh` skips already-complete slots (each `lm_eval/<task>.json` is checked individually; if an MMLU run died halfway, it re-runs MMLU only). Just re-execute the same command.

---

## 6. Validate against Kurt before celebrating

Once the sweep finishes, compare the FP16 + 13 quants to Kurt's Table 2:

```bash
# Quick comparison table
for r in results/*-meta-llama_llama-3.1-8b-instruct-*-ctx512/results.json; do
  python3 -c "
import json
d = json.load(open('$r'))
ha = d['tasks'].get('hellaswag', {}).get('acc_norm', '?')
g = d['tasks'].get('gsm8k', {}).get('exact_match_flexible_extract', '?')
m = d['tasks'].get('mmlu', {}).get('acc', '?')
print(f\"{d['quant']:8s} size={d['size_mib']:7.1f}MiB  HSwag={ha}  GSM8K-FE={g}  MMLU={m}\")
"
done | sort
```

Reference values from Kurt 2026 Table 2:

| Quant | HSwag | GSM8K-FE | MMLU |
|---|---|---|---|
| F16 | 0.7251 | 0.7763 | 0.6350 |
| Q4_K_S | 0.7279 | 0.7733 | 0.6206 |
| Q5_0 | 0.7263 | 0.7908 | 0.6318 |
| Q3_K_S | 0.7187 | 0.6831 | 0.5931 |
| Q8_0 | 0.7252 | 0.7748 | 0.6343 |

If your numbers fall within ~1 percentage point of these on the listed five anchors, the replication is solid. If anything is off by more than 2 percentage points, that quant is suspect — re-run that one slot before trusting the sweep.

---

## 7. Pull data back, shut down

```bash
# From your Mac, pull the results back
gcloud compute scp --recurse \
  benchmark-c3-176:~/benchmark/results \
  benchmark-c3-176:~/benchmark/env \
  ./gcp-results/ --zone=europe-west4-a

# Stop the instance (still costs storage but not compute)
gcloud compute instances stop benchmark-c3-176 --zone=europe-west4-a

# Or delete it (once you have the data)
gcloud compute instances delete benchmark-c3-176 --zone=europe-west4-a
```

Stopped instance disk costs are roughly €0.04/GB/month — your 500 GB disk costs ~€20/month while stopped. Keep it stopped (not deleted) for a week in case you need to re-run something.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `403 Forbidden` on Llama download | Meta gating not approved yet | Wait for the email; meanwhile run with Mistral-7B-v0.3 (non-gated) |
| `OOM` during quantization | Disk full | Confirm 500 GB boot disk; F16 GGUF + 13 quants ≈ 110 GiB |
| `bench.json` IQR warnings on every slot | Background processes (other tenants? cron?) | Re-run with `nice -n -10` and confirm exclusive use of the box |
| Numbers diverge from Mac smoke by >1 pt | Different llama.cpp commit | `git -C $LLAMA_CPP_DIR rev-parse HEAD` should match `env/llama_cpp.json` |
| `--limit` accidentally still set | Forgot to unset env var | `unset LM_EVAL_LIMIT` before kicking the sweep |
| `tg=128 tok/s` at ctx=2048 | bench.sh used old `-n 128` | Confirm 30_bench.sh has the `TG_N` logic at the top |

---

## What's next after this completes

Once Llama-3.1-8B is done:

1. Run the same `sweep.sh` for `mistralai/Mistral-7B-Instruct-v0.3`. No HF gating; expect ~50h parallel-8 (slightly faster, smaller model).
2. Run for the corrected Qwen id (likely `Qwen/Qwen3-8B`). ~50h.
3. **Optional scope cut for the cross-family models:** drop MMLU from steps 1 and 2 to halve the wall-clock. Justify in the Limitations chapter — H1/H2 don't depend on cross-family MMLU parity.

Total budget: **~3 models × ~50h = 150h ≈ 6.3 days** at parallel-8 if you keep MMLU on all three; **~3.5 days** if you drop MMLU on Mistral and Qwen. Well within your one-month rental.

---

## Quick command reference

```bash
# Single slot
LM_EVAL_LIMIT=200 LM_EVAL_N_THREADS=88 \
  ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 hellaswag

# Single slot — perf only at ctx=2048
PERF_ONLY=1 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 2048 3

# Full sweep (one model, all 14 configs, all 6 tasks, parallel-8)
LM_EVAL_N_THREADS=11 \
  ./scripts/sweep.sh meta-llama/Llama-3.1-8B-Instruct \
    hellaswag,gsm8k,ifeval,mmlu,truthfulqa_mc2,wikitext 8

# Sweep without MMLU (for cross-family models)
LM_EVAL_N_THREADS=11 \
  ./scripts/sweep.sh Qwen/Qwen3-8B \
    hellaswag,gsm8k,ifeval,truthfulqa_mc2,wikitext 8
```
