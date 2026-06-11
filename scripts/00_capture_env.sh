#!/usr/bin/env bash
# Capture environment metadata into env/. Safe to re-run (overwrites).
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
LM_EVAL_REQUIRED="0.4.9.2"

mkdir -p env

# Collect host data once. lscpu is invoked at most once.
uname_str=$(uname -a)
platform=$(uname -s)

if command -v lscpu >/dev/null 2>&1; then
  lscpu_out=$(lscpu)
  cpu_model=$(echo "$lscpu_out" | awk '/^Model name[[:space:]]*:/ {sub(/^[^:]+:[[:space:]]*/,""); print; exit}')
  cpu_cores=$(echo "$lscpu_out" | awk '/^CPU\(s\):[[:space:]]*/ {print $NF; exit}')
  flags=$(echo "$lscpu_out" | awk '/^Flags[[:space:]]*:/ {sub(/^[^:]+:[[:space:]]*/,""); print; exit}')
  governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
else
  cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
  cpu_cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 0)
  flags=$(sysctl -n machdep.cpu.features 2>/dev/null || echo "")
  governor="n/a-darwin"
fi

# Defensive defaults — never emit invalid JSON.
[ -n "$cpu_model" ] || cpu_model="unknown"
[ -n "$cpu_cores" ] || cpu_cores=0
[ -n "$flags" ] || flags=""
[ -n "$governor" ] || governor="unknown"

if command -v free >/dev/null 2>&1; then
  ram_total_mib=$(free -m | awk '/^Mem:/ {print $2; exit}')
else
  mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  ram_total_mib=$(( mem_bytes / 1024 / 1024 ))
fi
[ -n "$ram_total_mib" ] || ram_total_mib=0

# Write host.json via Python so escaping is bulletproof.
python3 - "$cpu_model" "$cpu_cores" "$flags" "$governor" "$ram_total_mib" "$uname_str" "$platform" <<'PY' > env/host.json
import json, sys, datetime
cpu_model, cpu_cores, flags, governor, ram_total_mib, uname_str, platform = sys.argv[1:8]
obj = {
    "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "uname": uname_str,
    "cpu_model": cpu_model,
    "cpu_cores": int(cpu_cores) if cpu_cores.isdigit() else 0,
    "flags": flags,
    "governor": governor,
    "ram_total_mib": int(ram_total_mib) if ram_total_mib.isdigit() else 0,
    "platform": platform,
}
print(json.dumps(obj, indent=2))
PY

# llama_cpp.json
if [ -d "$LLAMA_CPP_DIR/.git" ]; then
  commit=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
else
  commit="UNKNOWN-no-git"
fi
python3 - "$LLAMA_CPP_DIR" "$commit" <<'PY' > env/llama_cpp.json
import json, sys
path, commit = sys.argv[1], sys.argv[2]
print(json.dumps({"path": path, "commit": commit, "build_dir": f"{path}/build"}, indent=2))
PY

# lm_eval.json — hard-fail on version mismatch
ver=$(python3 -c "import lm_eval; print(lm_eval.__version__)" 2>/dev/null || echo "MISSING")
if [ "$ver" != "$LM_EVAL_REQUIRED" ]; then
  echo "ERROR: lm_eval version is '$ver', expected '$LM_EVAL_REQUIRED' (per exposé §4.4)" >&2
  exit 1
fi
python3 - "$ver" <<'PY' > env/lm_eval.json
import json, sys
print(json.dumps({"version": sys.argv[1]}, indent=2))
PY

# Compute and print env hash for caller convenience.
# shellcheck source=lib/env_hash.sh
source "$(dirname "$0")/lib/env_hash.sh"
echo "env_hash: $(env_hash env)"
echo "env captured: env/host.json env/llama_cpp.json env/lm_eval.json"
