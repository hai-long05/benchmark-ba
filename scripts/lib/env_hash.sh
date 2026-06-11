#!/usr/bin/env bash
# Compute a stable SHA256 of the env/ directory contents.
# Sourced by 00_capture_env.sh and run_slot.sh — do not execute directly.

env_hash() {
  local env_dir="${1:-env}"
  if [ ! -d "$env_dir" ]; then
    echo "env_hash: $env_dir not found" >&2
    return 1
  fi
  # Sorted file list, then SHA256 of concatenated SHAs of each file.
  ( cd "$env_dir" && find . -type f -name '*.json' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 \
    | shasum -a 256 \
    | awk '{print $1}' )
}
