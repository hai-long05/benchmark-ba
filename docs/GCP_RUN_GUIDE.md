# GCP Full Llama-3.1-8B Sweep — Step-by-Step

This guide takes you from "I just provisioned a GCP `c3-standard-192`" to "the full Kurt-equivalent benchmark for Llama-3.1-8B-Instruct (14 quants × 6 tasks at ctx=512 + 14 perf-only slots at ctx=2048) is on disk in ~50–55 hours of wall-clock."

The pipeline at HEAD has all the changes you need: FP16 baseline path, single-process multi-task lm_eval, wikitext perplexity via `loglikelihood_rolling`, `tg128` at both context lengths (matches Kurt Tab. 3 and the methodology's H3 operationalization), `PERF_ONLY=1` for ctx=2048 bench-only runs, a `sweep.sh` driver with GNU parallel support, **and a chat-template path that replicates Kurt 2026 §3.1**.

The reference machine assumed throughout: **`c3-standard-192`** (192 vCPUs / 96 physical cores, 768 GiB RAM, Intel Xeon Platinum 8481C, AVX-512 + AMX-BF16). This is the same Sapphire-Rapids generation as Kurt's 8488C — direct comparability.

---

## 0. What changed since v1 of this guide

If you read an older version of this guide, four things are different now:

1. **llama.cpp commit corrected.** v1 pinned `1bfbdb134…`, but Kurt 2026 §3.1 footnote 3 cites release tag `b7600`, which is git commit `be47fb9285779e900915bd8246eb9664110d4ba5`. Quantized weights from a different llama-quantize build can shift K-quant outputs enough to break the ±2·SE_Kurt acceptance band — re-build llama.cpp from this commit on the GCP box (§3.3) before any quantization step.
2. **Chat-template convention is on by default.** All five prompted accuracy tasks (GSM8K, HellaSwag, IFEval, MMLU, TruthfulQA-MC2) are now run with `--apply_chat_template --fewshot_as_multiturn`, sourcing each model's default template from its HF tokenizer. WikiText-2 is run separately without those flags. This replicates Kurt 2026 §3.1 ("prompts were formatted using the default Llama-3.1-8B-Instruct chat template") and is required for SF1 to land inside the ±2·SE_Kurt acceptance band.
3. **Cross-family models locked in.** Qwen variant is **Qwen3-8B** (decided), not Qwen3-4B. All three models are now in the 7–8 B class, which doubles as a model-size control for the cross-family comparison.
4. **Mistral may be gated.** Recent mistralai-org policy gates several Instruct repos. Verify access in §1.1 *before* paying for the GCP instance.

Every script change that backs these decisions is already on `main` — `scripts/lib/lm_eval_gguf_runner.py`, `scripts/40_lm_eval.sh`, `scripts/run_slot.sh`, `scripts/sweep.sh`, `configs/tasks.yaml`, `requirements.txt`. You don't need to touch them — but if you fork an older branch, rebase first.

---

## 1. Pre-flight on your laptop (do this BEFORE provisioning the instance)

The instance costs ~€10/hour. Don't pay it to idle while you debug auth.

### 1.1 Hugging Face access (Llama-3.1 is gated; Mistral may be)

1. Sign in to https://huggingface.co with the account you'll also use on GCP.
2. **Llama-3.1-8B-Instruct (definitely gated):** visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct → **Agree and access repository**. Approval is usually automatic; can take up to 48 h.
3. **Mistral-7B-Instruct-v0.3 (sometimes gated):** visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3. If you see an "Agree and access repository" button, click it. If not, you're already cleared.
4. **Qwen3-8B (open):** https://huggingface.co/Qwen/Qwen3-8B is Apache-2.0, no gating action needed.
5. At https://huggingface.co/settings/tokens create a token with **read** scope. Save it.
6. Verify locally — both gated repos plus the open one:
   ```bash
   source .venv/bin/activate
   huggingface-cli login   # paste the token
   python3 -c "
   from huggingface_hub import hf_hub_download
   for repo in ['meta-llama/Llama-3.1-8B-Instruct',
                'mistralai/Mistral-7B-Instruct-v0.3',
                'Qwen/Qwen3-8B']:
       hf_hub_download(repo, 'config.json'); print(f'OK: {repo}')
   "
   ```
   Three `OK` lines = cleared. `awaiting review` = wait. Anything else = stop and fix before going to GCP.

### 1.2 Mac smoke (final sanity check before paying for GCP)

```bash
cd ~/benchmark   # or wherever the repo lives
source .venv/bin/activate
export LLAMA_CPP_DIR=../llama.cpp

LM_EVAL_LIMIT=20 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct F16 512 3 hellaswag,gsm8k
LM_EVAL_LIMIT=20 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 hellaswag,gsm8k
```

`run_slot.sh` automatically derives `TOKENIZER_REPO` from the HF id, so no extra env-var is needed.

You're verifying that:
- Llama auth works (the F16 fetch is ~16 GiB)
- The chat-template path doesn't crash (look for `chat_template_applied: True` in the lm_eval log)
- `results.json` for the F16 row shows `"quant": "F16", "bits_nominal": 16, "size_reduction_pct": 0.0`
- HellaSwag acc_norm is in roughly the right ballpark — Q4_K_S near 0.70, F16 slightly higher (broad CIs at limit=20)
- `slot.json` contains a `chat_template` block with `tokenizer_repo: meta-llama/Llama-3.1-8B-Instruct`

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

500 GB is enough: 3× FP16 GGUF (~50 GiB total) + 3× 13 quants (~210 GiB total) + lm_eval datasets (~3 GiB) + HF tokenizer cache (~30 MiB) + logs ≈ 270 GiB peak. SSD matters — quantization is single-threaded I/O-bound.

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

# llama.cpp at the same commit Kurt 2026 used (release tag b7600 →
# commit be47fb9285779e900915bd8246eb9664110d4ba5). This is THE pin —
# K-quants are community-driven and change between commits, so a
# different SHA can shift quantized weights enough to break the
# ±2·SE_Kurt acceptance band on Llama-3.1-8B.
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout be47fb9285779e900915bd8246eb9664110d4ba5

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

### 3.4 Python venv + HF tokenizer pre-cache

```bash
cd ~/benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# IFEval needs nltk's 'punkt' corpora at runtime (one-time download, ~3 MB)
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"
```

**Pre-cache the HF tokenizers for all three models** so 8 parallel slots don't race the same `~/.cache/huggingface/` during sweep startup:

```bash
huggingface-cli login   # paste the token

python3 -c "
from transformers import AutoTokenizer
for repo in ['meta-llama/Llama-3.1-8B-Instruct',
             'mistralai/Mistral-7B-Instruct-v0.3',
             'Qwen/Qwen3-8B']:
    AutoTokenizer.from_pretrained(repo)
    print(f'cached: {repo}')
"
```

Three `cached:` lines = ready. If Mistral 401's, go back to §1.1. If Llama 403's, your token is missing or your gate isn't approved.

### 3.5 Hugging Face auth (already done in §3.4 if you followed in order)

```bash
huggingface-cli whoami  # confirm — should print your username
```

### 3.6 Capture environment metadata

```bash
export LLAMA_CPP_DIR=$HOME/llama.cpp
./scripts/00_capture_env.sh
cat env/host.json | python3 -m json.tool | head -10
```

You should see `"governor": "performance"`, `"cpu_cores": 192`, and the AVX-512 + AMX-BF16 flags. **From this moment on, every slot is hashed against this `env/`. Don't change governor or rebuild llama.cpp without re-running this and accepting that prior slots become un-mergeable.**

The chat-template convention itself is *not* part of the env-hash — it's recorded per-slot in `slot.json` (field `chat_template`) so each measurement is self-describing for the reproducibility package, but changing template kwargs between slots wouldn't trigger the env-drift guard. If you change the convention mid-sweep, the data are no longer mergeable; treat that as a hard discipline rule.

---

## 4. GCP smoke test (~15 min, do this BEFORE the full sweep)

You're verifying that (1) the GCP box agrees with Kurt's published Q4_K_S numbers within ~1 pp at limit=200, (2) the build actually uses AMX-BF16 (~5x faster than Mac expectations), and (3) the chat-template path is genuinely engaged — not silently skipped.

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

`LM_EVAL_N_THREADS=88` pins llama-cpp-python to 88 physical cores (out of 96), leaving 8 for OS + I/O + tqdm. Single-slot smoke maximises per-slot throughput. `TOKENIZER_REPO` defaults to the HF id, so no extra env-var needed. IFEval is in the smoke set this time — it's the task most sensitive to the template change and the cheapest way to confirm the convention is engaged.

### 4.2 Detach and let it run

`Ctrl-b` then `d` — you're back at the regular shell; the smoke keeps running.

### 4.3 Reattach later

```bash
tmux attach -t smoke
```

### 4.4 Pass criteria (n=200)

```bash
RESULT=$(ls -td results/*-meta-llama_llama-3.1-8b-instruct-q4_k_s-ctx512/results.json | head -1)
python3 <<PY
import json
d = json.load(open("$RESULT"))
t = d['tasks']
print(f"pp512: {d['pp']['median']:.1f} tok/s   (target: > 350 — Kurt: 92.5 on Mac, you have AMX)")
print(f"tg128: {d['tg']['median']:.2f} tok/s   (target: > 30)")
print(f"hellaswag acc_norm:    {t['hellaswag']['acc_norm']:.4f}    (target: 0.71-0.75; Kurt full: 0.7279)")
print(f"gsm8k flex-extract:    {t['gsm8k']['exact_match_flexible_extract']:.4f}    (target: 0.74-0.81; Kurt full: 0.7733)")
print(f"truthfulqa_mc2:        {t['truthfulqa_mc2']['acc']:.4f}    (target: 0.50-0.57; Kurt full: 0.5340)")
# IFEval has four sub-metrics; primary is prompt_level_loose_acc (PLL) per Kurt Tab 6.
ifeval_pll = t['ifeval'].get('prompt_level_loose_acc', t['ifeval'].get('prompt_level_strict_acc'))
print(f"ifeval PLL:            {ifeval_pll:.4f}    (target: 0.74-0.84; Kurt full Q4_K_S PLL: 0.7911)")
PY
```

If **all six** pass, you're cleared for the full sweep. **If IFEval fails low** (PLL < 0.70), the chat-template path is not engaged — go to §4.5 and verify before re-running. **If anything else misses, stop** — re-running 14 broken slots wastes ~50 hours of GCP time. Tell me the specific failure.

### 4.5 Verify the chat-template path actually fired

This is the single most likely source of silent miscalibration. Two checks:

```bash
SLOT=$(ls -td results/*-meta-llama_llama-3.1-8b-instruct-q4_k_s-ctx512 | head -1)

# 1. slot.json records the convention
jq '.chat_template' "$SLOT/slot.json"
# expect: {"tokenizer_repo": "meta-llama/Llama-3.1-8B-Instruct",
#          "chat_template_kwargs": null,
#          "applied_to": ["hellaswag","gsm8k","ifeval","mmlu","truthfulqa_mc2"],
#          "skipped_for": ["wikitext"],
#          "fewshot_as_multiturn": true}

# 2. lm_eval's templated-group log mentions the template was applied
grep -E "chat_template|apply_chat_template|tokenizer_repo" \
  "$SLOT/lm_eval/_group_fs0_tmpl.log" "$SLOT/lm_eval/_group_fs5_tmpl.log" 2>/dev/null | head -20
# expect: at least one line mentioning the applied template / tokenizer_repo,
# zero lines saying "instruct/chat variant but chat template is not applied"
```

If §4.5 check 2 surfaces the warning **"appears to be an instruct or chat variant but chat template is not applied"** — the convention silently fell back to off. Stop, debug locally, do not start the sweep.

### 4.6 Pre-fetch all 14 quants while smoke runs

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

### 5.1 Chat-template convention (auto-applied; here for completeness)

`run_slot.sh` automatically:

- sets `TOKENIZER_REPO` to the HF id of the model being measured (override with `TOKENIZER_REPO=...` if needed; not needed for Llama/Mistral/Qwen3),
- sets `CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'` when the HF id contains `qwen3` (suppresses Qwen3 reasoning blocks during loglikelihood scoring),
- writes the resolved values into `slot.json/chat_template` so each slot is self-describing for §3.6's reproducibility discipline.

You **don't need to set anything** for the standard three-model run. The two env-vars are only for overrides — e.g. if you wanted to measure Llama under Mistral's template (you don't, but the path exists).

### 5.2 Concurrency layout for `c3-standard-192`

96 physical cores. Three layouts; pick by goal:

| Layout | Concurrent slots | n_threads/slot | RAM/slot × N | Best for |
|---|---|---|---|---|
| **8 × 12** | 8 | 12 | ~80 GB | **Recommended** — overall throughput |
| 12 × 8 | 12 | 8 | ~120 GB | Maximises tail-end completion (fewer slots stragglers at the end) |
| 4 × 24 | 4 | 24 | ~40 GB | Single-slot wall-clock matters more |

768 GiB RAM means you're nowhere near memory-bound — pick by core sharing efficiency. **Use 8 × 12.**

### 5.3 Start the sweep in tmux

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

### 5.4 What `sweep.sh` does

1. **Warm-up (sequential, ~5 min if §4.6 was done):** confirms FP16 GGUF + 13 quants exist; runs whichever are missing. F16 reuses the FP16 GGUF directly — no quantize step.
2. **Accuracy phase (parallel-8, ~50–55 h):** runs all 14 (model, quant, ctx=512) slots. Each slot dispatches lm_eval in **two groups**: a templated group with `--apply_chat_template --fewshot_as_multiturn` for the five prompted tasks, and a separate raw-text group for wikitext. Within the templated group, fewshot_as_multiturn is no-op for the four 0-shot tasks and active only for GSM8K (5-shot).
3. **Performance phase (parallel-8, ~2 min):** 14 slots at ctx=2048 with `PERF_ONLY=1`, only `30_bench` runs. Uses `tg128` at both context lengths (matches Kurt Tab. 3 and the methodology's H3 operationalization).

### 5.5 Wall-clock estimate

| Phase | Sequential | Parallel-8 |
|---|---|---|
| Warm-up | 30 min | 30 min |
| Accuracy (14 slots × 6 tasks, with template) | ~400 h | **~50–55 h** |
| Perf at ctx=2048 | 14 min | 2 min |
| **Total per model** | ~400 h | **~50–55 h ≈ 2.2 days** |

The estimate moved up from "~46 h" in v1 because IFEval with the chat template generates noticeably longer completions (the model respects format constraints rather than truncating). MMLU is still the single biggest task: ~14h × 14 slots ÷ 8 workers ≈ 25 of those 50 hours. For the Kurt anchor model **keep MMLU in** — Kurt reports it and you'll need it for the Avg.

For all three models combined (Llama → Mistral → Qwen3-8B): **~6.5 days of wall-clock**. Comfortably inside your 14-day GCP rental.

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

## 8. Validate against Kurt's tables before celebrating

Tolerance is **±2·SE_Kurt** per Kurt's harness-reported standard errors (matches the methodology's acceptance band; one-pp fixed tolerance from v1 was a placeholder).

```bash
cd ~/benchmark
python3 <<'PY'
import json, glob, os, sys

# Kurt 2026 reference values (Llama-3.1-8B-Instruct on Xeon 8488C). Format: (mean, SE).
# HellaSwag: acc_norm. GSM8K: exact_match_flexible_extract. MMLU: acc.
# IFEval: prompt_level_loose_acc (PLL) per Tab. 6. TruthfulQA: MC2 acc.
KURT = {
  'F16':    {'hswag': (0.7251, 0.0045), 'gsm8k_fe': (0.7763, None),
             'mmlu':  (0.6350, 0.0038), 'ifeval_pll':(0.7708, 0.0181),
             'tqa_mc2': (0.5479, 0.0160)},
  'Q3_K_S': {'hswag': (0.7187, 0.0045), 'gsm8k_fe': (0.6831, None),
             'mmlu':  (0.5931, 0.0039), 'ifeval_pll':(0.7116, 0.0195),
             'tqa_mc2': (0.5408, 0.0158)},
  'Q4_K_S': {'hswag': (0.7279, 0.0044), 'gsm8k_fe': (0.7733, None),
             'mmlu':  (0.6206, 0.0039), 'ifeval_pll':(0.7911, 0.0175),
             'tqa_mc2': (0.5340, 0.0159)},
  'Q5_0':   {'hswag': (0.7263, 0.0044), 'gsm8k_fe': (0.7908, None),
             'mmlu':  (0.6318, 0.0038), 'ifeval_pll':(0.7856, 0.0177),
             'tqa_mc2': (0.5457, 0.0160)},
  'Q8_0':   {'hswag': (0.7252, 0.0045), 'gsm8k_fe': (0.7748, None),
             'mmlu':  (0.6343, 0.0038), 'ifeval_pll':(0.7745, 0.0180),
             'tqa_mc2': (0.5481, 0.0160)},
}

def fetch(scheme):
    paths = sorted(glob.glob(
      f'results/*-meta-llama_llama-3.1-8b-instruct-{scheme.lower()}-ctx512/results.json'),
      key=os.path.getmtime, reverse=True)
    if not paths:
        return None
    return json.load(open(paths[0]))

def cmp(label, our, ref):
    if our is None or ref is None: return f'  {label:14s}  N/A'
    mean, se = ref
    delta = our - mean
    band = 2 * (se or 0.005)  # GSM8K SE missing in some Kurt rows; use 0.005 proxy
    flag = 'OK ' if abs(delta) <= band else 'OFF'
    return f'  {label:14s}  ours={our:.4f}  kurt={mean:.4f}  Δ={delta:+.4f}  ±2SE={band:.4f}  [{flag}]'

print(f"{'='*72}\nReplication check vs. Kurt 2026 (±2·SE_Kurt per Methodology §5.2)\n{'='*72}")
for scheme in ['F16','Q3_K_S','Q4_K_S','Q5_0','Q8_0']:
    d = fetch(scheme)
    if d is None:
        print(f'{scheme}: MISSING'); continue
    t = d.get('tasks', {})
    print(f'\n{scheme}:')
    print(cmp('hellaswag',  t.get('hellaswag',{}).get('acc_norm'), KURT[scheme]['hswag']))
    print(cmp('gsm8k_fe',   t.get('gsm8k',{}).get('exact_match_flexible_extract'), KURT[scheme]['gsm8k_fe']))
    print(cmp('mmlu',       t.get('mmlu',{}).get('acc'), KURT[scheme]['mmlu']))
    ifv = t.get('ifeval',{}).get('prompt_level_loose_acc') or t.get('ifeval',{}).get('prompt_level_strict_acc')
    print(cmp('ifeval_pll', ifv, KURT[scheme]['ifeval_pll']))
    print(cmp('tqa_mc2',    t.get('truthfulqa_mc2',{}).get('acc'), KURT[scheme]['tqa_mc2']))
PY
```

**Pass criteria:** every row marked `[OK ]`. If any row is `[OFF]` by more than 1.5× the band, that slot is suspect — re-run only that one with `./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct <SCHEME> 512 3 <task>` before trusting the sweep. If IFEval-PLL is consistently `[OFF]` across multiple schemes, the chat-template path is broken — go back to §4.5 and confirm.

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
| `401 Unauthorized` on Mistral fetch | Mistral repo gated and not approved | Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and click Agree |
| `ImportError: No module named transformers` | Old `requirements.txt` (pre-template) | `pip install -r requirements.txt` again — must include transformers, tokenizers, jinja2 |
| `RuntimeError: tokenizer_repo set but transformers not installed` | Same as above | Same |
| `RuntimeError: apply_chat_template called but tokenizer_repo not set` | `TOKENIZER_REPO` env-var got unset somewhere | The `run_slot.sh` default should resolve this; if you're calling `40_lm_eval.sh` directly, pass `TOKENIZER_REPO=<hf-id>` |
| `ERROR: CHAT_TEMPLATE_KWARGS must not contain commas` | You set a multi-key JSON with commas | lm_eval splits model_args on `,`; use a single-key JSON like `'{"enable_thinking": false}'` |
| Smoke IFEval-PLL ≪ 0.70 | Chat-template path silently off | Check `slot.json/chat_template`; check `_group_fs0_tmpl.log` for "instruct/chat variant but chat template is not applied" warning |
| `OOM` during quantization | Disk full | Confirm 500 GB SSD; full quant set across 3 models ≈ 270 GiB |
| `bench.json` IQR warnings on every slot | Background processes / governor changed mid-run | Re-run §3.2; verify `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` |
| Numbers diverge from Mac smoke by >1 pt | Different llama.cpp commit or build flags | `git -C $LLAMA_CPP_DIR rev-parse HEAD` should match `env/llama_cpp.json` |
| `--limit` accidentally still set | Forgot to unset env var | `unset LM_EVAL_LIMIT` before kicking the sweep |
| `tg=256 tok/s` instead of `tg=128 tok/s` | Old `30_bench.sh` from before the B2 fix | `grep "TG_N=" scripts/30_bench.sh` should show `TG_N="${N_GEN:-128}"`, not a conditional with 256 |
| Aggregate refuses with `env drift` | governor/llama.cpp/lm_eval changed since slot.json was written | Re-run that slot with `./scripts/run_slot.sh ...` |
| `cpupower: command not found` | Package missing | `sudo apt-get install -y linux-tools-common linux-tools-$(uname -r)` or use the `echo performance` fallback in §3.2 |
| `caching_allocator_warmup ... CUDA enabled` errors | torch trying to initialise CUDA on CPU-only box | Make sure you're using `--model gguf_local` (the project's path), not `--model hf` |
| Qwen3 IFEval/MMLU near-zero | `<think>` blocks leaking into the loglikelihood path | Confirm `slot.json/chat_template/chat_template_kwargs` shows `{"enable_thinking": false}`; HF id must contain `Qwen3` (case-insensitive match in `run_slot.sh`) |

---

## 11. Quick command reference

```bash
# One slot (TOKENIZER_REPO auto-derived from HF id)
LM_EVAL_LIMIT=200 LM_EVAL_N_THREADS=88 \
  ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 512 3 hellaswag,ifeval

# One slot — perf only at ctx=2048
PERF_ONLY=1 ./scripts/run_slot.sh meta-llama/Llama-3.1-8B-Instruct Q4_K_S 2048 3

# One slot — override TOKENIZER_REPO (rarely needed; only if GGUF was quantized
# from a different base than the chat-template owner)
TOKENIZER_REPO=meta-llama/Llama-3.1-8B-Instruct \
  ./scripts/run_slot.sh some-org/some-repackaged-llama Q4_K_S 512 3 hellaswag

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

Same sweep command for the cross-family models. Drop MMLU on the cross-family runs — saves ~50% wall-clock per model and is documented as a scope cut in the thesis Limitations:

```bash
# Mistral — verify §1.1 gating cleared first
LM_EVAL_N_THREADS=12 \
  ./scripts/sweep.sh mistralai/Mistral-7B-Instruct-v0.3 \
    hellaswag,gsm8k,ifeval,truthfulqa_mc2,wikitext 8

# Qwen3-8B — open Apache-2.0, no gating. enable_thinking=false is auto-set
# by run_slot.sh on the qwen3 pattern match.
LM_EVAL_N_THREADS=12 \
  ./scripts/sweep.sh Qwen/Qwen3-8B \
    hellaswag,gsm8k,ifeval,truthfulqa_mc2,wikitext 8
```

Each ~25–30 h parallel-8 (without MMLU). Total cross-family: ~50–60 h ≈ 2.5 days. Combined with Llama's ~50–55 h: **the full thesis benchmark fits in ~6.5 days of GCP wall-clock**, well under your 14-day rental.

---

## 13. Run the paired bootstrap (after the sweep, before pulling data)

The methodology (04_methodik.tex §subsection paired bootstrap, Z. 615-677) requires per-task paired-bootstrap CIs for every (model, scheme, task) tuple's Quant-vs-FP16 difference. The data needed is in the `--log_samples` jsonl files written under each slot's `lm_eval/_group_*_raw/` directory — the bootstrap can run after the sweep is complete and is fast (~1 minute per (model, scheme) pair on the c3 box).

```bash
cd ~/benchmark
source .venv/bin/activate
mkdir -p bootstrap_results

# Llama: FP16 baseline + all 13 quants
BASE_LLAMA=$(ls -td results/*-meta-llama_llama-3.1-8b-instruct-f16-ctx512 | head -1)
python3 scripts/60_bootstrap.py \
  --baseline-slot "$BASE_LLAMA" \
  --quant-glob   'results/*-meta-llama_llama-3.1-8b-instruct-*-ctx512' \
  --tasks         hellaswag,gsm8k,ifeval,mmlu,truthfulqa_mc2 \
  --b             1000 \
  --out-dir       bootstrap_results/llama/

# Mistral: same convention
BASE_MISTRAL=$(ls -td results/*-mistralai_mistral-7b-instruct-v0.3-f16-ctx512 | head -1)
python3 scripts/60_bootstrap.py \
  --baseline-slot "$BASE_MISTRAL" \
  --quant-glob   'results/*-mistralai_mistral-7b-instruct-v0.3-*-ctx512' \
  --tasks         hellaswag,gsm8k,ifeval,truthfulqa_mc2 \
  --out-dir       bootstrap_results/mistral/

# Qwen3-8B: same
BASE_QWEN=$(ls -td results/*-qwen_qwen3-8b-f16-ctx512 | head -1)
python3 scripts/60_bootstrap.py \
  --baseline-slot "$BASE_QWEN" \
  --quant-glob   'results/*-qwen_qwen3-8b-*-ctx512' \
  --tasks         hellaswag,gsm8k,ifeval,truthfulqa_mc2 \
  --out-dir       bootstrap_results/qwen3/
```

For any (model, scheme, task) where the printed CI is marginal (e.g. `[-0.001,+0.003]`), re-run that one with `--b 10000` for a stable percentile estimate before reporting the result. Random seed is fixed (`0xBACE10AD`) so re-runs of the same data produce the same CI.

WikiText-2 is exempted (corpus-aggregated, not item-resamplable); the per-config PPL with its harness-internal SE is reported directly without bootstrap, per methodology Z. 679-683.
