#!/usr/bin/env bash
# Capture environment metadata into env/. Idempotent.
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
LM_EVAL_REQUIRED="0.4.9.2"

mkdir -p env

# host.json
{
  echo '{'
  echo "  \"captured_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"uname\": \"$(uname -a | sed 's/"/\\"/g')\","
  if command -v lscpu >/dev/null 2>&1; then
    echo "  \"cpu_model\": \"$(lscpu | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')\","
    echo "  \"cpu_cores\": $(lscpu | awk -F: '/^CPU\(s\):/ {gsub(/ /,"",$2); print $2; exit}'),"
    flags=$(lscpu | awk -F: '/Flags/ {print $2}' | tr -s ' ' | sed 's/^ //;s/"/\\"/g')
    echo "  \"flags\": \"$flags\","
    gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    echo "  \"governor\": \"$gov\","
  else
    echo "  \"cpu_model\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)\","
    echo "  \"cpu_cores\": $(sysctl -n hw.physicalcpu 2>/dev/null || echo 0),"
    echo "  \"flags\": \"$(sysctl -n machdep.cpu.features 2>/dev/null || echo unknown)\","
    echo "  \"governor\": \"n/a-darwin\","
  fi
  if command -v free >/dev/null 2>&1; then
    echo "  \"ram_total_mib\": $(free -m | awk '/Mem:/ {print $2}'),"
  else
    echo "  \"ram_total_mib\": $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 )),"
  fi
  echo "  \"platform\": \"$(uname -s)\""
  echo '}'
} > env/host.json

# llama_cpp.json
if [ -d "$LLAMA_CPP_DIR/.git" ]; then
  commit=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
else
  commit="UNKNOWN-no-git"
fi
{
  echo '{'
  echo "  \"path\": \"$LLAMA_CPP_DIR\","
  echo "  \"commit\": \"$commit\","
  echo "  \"build_dir\": \"$LLAMA_CPP_DIR/build\""
  echo '}'
} > env/llama_cpp.json

# lm_eval.json
ver=$(python -c "import lm_eval; print(lm_eval.__version__)" 2>/dev/null || echo "MISSING")
if [ "$ver" != "$LM_EVAL_REQUIRED" ]; then
  echo "ERROR: lm_eval version is '$ver', expected '$LM_EVAL_REQUIRED' (per exposé §4.4)" >&2
  exit 1
fi
{
  echo '{'
  echo "  \"version\": \"$ver\""
  echo '}'
} > env/lm_eval.json

# Compute and print env hash for caller convenience.
# shellcheck source=lib/env_hash.sh
source "$(dirname "$0")/lib/env_hash.sh"
echo "env_hash: $(env_hash env)"
echo "env captured: env/host.json env/llama_cpp.json env/lm_eval.json"
