#!/usr/bin/env bash
# Run lm-evaluation-harness on one task against a quantized GGUF.
# Usage: 40_lm_eval.sh <gguf> <task_key> <out_dir>
#   <task_key> matches a key under `tasks:` in configs/tasks.yaml (e.g. hellaswag).
#
# Architecture:
#   llama-server (new OpenAI-compat format)  ->  proxy (echo emulator)  ->  lm_eval gguf backend
#
# Compatibility note:
#   lm_eval 0.4.9.2's `gguf` backend (--model gguf) was designed for older llama-server
#   builds that echoed prompt-token logprobs in /v1/completions responses.  Modern
#   llama-server (>=b4000, this project uses b9595) silently ignores echo=True and
#   only returns the single generated token's logprobs.
#
#   This script works around the incompatibility by starting a thin Python reverse
#   proxy that intercepts echo=True requests and reconstructs the legacy logprobs
#   format via N sequential /v1/completions calls per continuation (one per token).
#   llama-server's KV-cache makes each call fast after the first.
#
#   Trade-off: the proxy approach is CPU-bound by HTTP round-trips (~1 it/s on Apple
#   M1 Pro for multiple-choice tasks).  Full hellaswag (10042 examples) takes ~8 h.
#   Add LM_EVAL_LIMIT=<N> to run on a subset for smoke-testing:
#     LM_EVAL_LIMIT=100 ./scripts/40_lm_eval.sh ...  (~5 min)
set -euo pipefail

GGUF="${1:?Usage: $0 <gguf> <task_key> <out_dir>}"
TASK_KEY="${2:?task_key required}"
OUT_DIR="${3:?out_dir required}"
TASKS_YAML="${TASKS_YAML:-configs/tasks.yaml}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"   # optional: set to N to limit examples

[ -f "$GGUF" ] || { echo "ERROR: $GGUF not found" >&2; exit 1; }
[ -f "$TASKS_YAML" ] || { echo "ERROR: $TASKS_YAML not found" >&2; exit 1; }
[ -x "$SERVER_BIN" ] || { echo "ERROR: $SERVER_BIN not found or not executable" >&2; exit 1; }

# Read task config via Python (yaml).
read -r task_id num_fewshot < <(python3 - "$TASKS_YAML" "$TASK_KEY" <<'PY'
import yaml, sys
yaml_path, key = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(yaml_path))['tasks']
if key not in cfg:
    sys.exit(f"unknown task_key: {key}")
t = cfg[key]
print(t['task_id'], t['num_fewshot'])
PY
)

mkdir -p "$OUT_DIR/lm_eval"

# Idempotency: skip if results.json already exists.
if [ -f "$OUT_DIR/lm_eval/${TASK_KEY}.json" ]; then
  echo "lm_eval already done: $OUT_DIR/lm_eval/${TASK_KEY}.json"
  exit 0
fi

# Pick two free ports (server + proxy).
read -r SERVER_PORT PROXY_PORT < <(python3 - <<'PY'
import socket, random
ports = []
for _ in range(40):
    p = random.randint(18000, 28000)
    if p in ports:
        continue
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', p)) != 0:
            ports.append(p)
    if len(ports) == 2:
        break
if len(ports) < 2:
    raise RuntimeError("could not find two free ports")
print(ports[0], ports[1])
PY
)

SERVER_LOG="$OUT_DIR/lm_eval/${TASK_KEY}_server.log"
PROXY_LOG="$OUT_DIR/lm_eval/${TASK_KEY}_proxy.log"
PROXY_SCRIPT="/tmp/_lm_eval_proxy_$$.py"
SERVER_PID=""
PROXY_PID=""

cleanup() {
  [ -n "$PROXY_PID"  ] && kill "$PROXY_PID"  2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  rm -f "$PROXY_SCRIPT"
  wait 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Write the echo-emulating proxy script.
#
# The proxy intercepts /v1/completions requests with echo=True (sent by lm_eval
# for loglikelihood scoring).  For each such request it:
#   1. Tokenises the full prompt (context+continuation) via /tokenize.
#   2. For each continuation token at position i, sends one completion request
#      for the prefix (context+continuation[:i]) with max_tokens=1 and logprobs=N
#      to obtain log P(token_i | prefix).  KV-cache reuse keeps each call fast.
#   3. Returns a legacy logprobs dict with tokens / token_logprobs / text_offset
#      in the format lm_eval 0.4.9.2 expects.
# ---------------------------------------------------------------------------
cat > "$PROXY_SCRIPT" <<'PYEOF'
import http.server, socketserver, urllib.request, json, sys

PROXY_PORT  = int(sys.argv[1])
SERVER_PORT = int(sys.argv[2])
UPSTREAM    = f"http://127.0.0.1:{SERVER_PORT}"
DEFAULT_TOP = 10


def _post(path, body):
    req = urllib.request.Request(
        UPSTREAM + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def tokenize(text):
    return _post("/tokenize", {"content": text})["tokens"]


def detokenize(ids):
    return _post("/detokenize", {"tokens": ids})["content"]


def build_echo_logprobs(prompt_text, n_top):
    """
    Build the legacy logprobs dict for prompt_text by scoring each token
    given its preceding tokens.  text_offsets are character positions in
    prompt_text so lm_eval's get_result() can find where the continuation starts.
    """
    full_tokens = tokenize(prompt_text)
    tokens_out, token_logprobs_out, top_logprobs_out, text_offsets = [], [], [], []
    char_offset = 0

    for i, tok_id in enumerate(full_tokens):
        prefix_tokens = full_tokens[:i]
        prefix_text   = detokenize(prefix_tokens) if prefix_tokens else ""
        tok_str       = detokenize([tok_id])

        resp    = _post("/v1/completions", {
            "prompt":      prefix_text,
            "max_tokens":  1,
            "logprobs":    n_top,
            "temperature": 0,
        })
        content = (resp["choices"][0].get("logprobs") or {}).get("content", [])

        lp       = None
        top_dict = {}
        if content:
            # Flatten token + its top_logprobs into a single lookup dict.
            for entry in content:
                for item in [entry] + entry.get("top_logprobs", []):
                    top_dict[item["token"]] = item["logprob"]
            lp = top_dict.get(tok_str)
            if lp is None:
                lp = content[0]["logprob"]  # fallback: logprob of generated token

        tokens_out.append(tok_str)
        token_logprobs_out.append(lp if lp is not None else 0.0)
        top_logprobs_out.append(top_dict)
        text_offsets.append(char_offset)
        char_offset += len(tok_str)

    # Append a dummy "generated" token at the end.
    # lm_eval's get_result() does sum(token_logprobs[idx:-1]) which excludes this.
    tokens_out.append("")
    token_logprobs_out.append(0.0)
    top_logprobs_out.append({})
    text_offsets.append(char_offset)

    return {
        "tokens":         tokens_out,
        "token_logprobs": token_logprobs_out,
        "top_logprobs":   top_logprobs_out,
        "text_offset":    text_offsets,
    }


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def _forward_raw(self, body_bytes):
        req = urllib.request.Request(
            UPSTREAM + self.path,
            data=body_bytes or None,
            headers={k: v for k, v in self.headers.items()
                     if k.lower() not in ("host", "content-length")},
            method=self.command,
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read(), resp.headers.get("Content-Type", "application/json")

    def _send(self, raw, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle(self):
        length   = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length) if length else b""

        if "/v1/completions" in self.path and b'"echo"' in body_raw:
            try:
                body = json.loads(body_raw)
                if body.get("echo") is True:
                    prompt = body.get("prompt", "")
                    n_top  = int(body.get("logprobs", DEFAULT_TOP)) if body.get("logprobs") else DEFAULT_TOP
                    lp     = build_echo_logprobs(prompt, n_top)
                    result = {
                        "choices": [{
                            "text":          "",
                            "index":         0,
                            "logprobs":      lp,
                            "finish_reason": "length",
                        }],
                        "model":  "proxy",
                        "object": "text_completion",
                    }
                    self._send(json.dumps(result).encode())
                    return
            except Exception:
                pass  # fall through to verbatim forward on any error

        try:
            raw, ctype = self._forward_raw(body_raw)
        except Exception as exc:
            self.send_error(502, str(exc))
            return
        self._send(raw, ctype)

    do_GET  = _handle
    do_POST = _handle
    do_HEAD = _handle


class FastHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threading server; skips the slow socket.getfqdn() DNS reverse lookup."""
    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host  # skip getfqdn (can take 30+ s on some macOS hosts)
        self.server_port = port


server = FastHTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
print(f"proxy listening on {PROXY_PORT} -> upstream {SERVER_PORT}", flush=True)
server.serve_forever()
PYEOF

# ---------------------------------------------------------------------------
# Start llama-server in the background.
# ---------------------------------------------------------------------------
echo "Starting llama-server on port $SERVER_PORT ..."
"$SERVER_BIN" \
  --model "$GGUF" \
  --host 127.0.0.1 \
  --port "$SERVER_PORT" \
  --ctx-size 2048 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait up to 120 s for the server health endpoint.
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
    echo "llama-server ready (${i}s)."
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: llama-server exited. See $SERVER_LOG" >&2; exit 1; }
  sleep 1
done
curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1 \
  || { echo "ERROR: llama-server did not become ready in 120s" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Start the echo-emulating proxy.
# ---------------------------------------------------------------------------
echo "Starting echo-emulating proxy on port $PROXY_PORT ..."
python3 "$PROXY_SCRIPT" "$PROXY_PORT" "$SERVER_PORT" >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

# Wait up to 10 s for proxy to be ready.
for i in $(seq 1 10); do
  if curl -sf "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
    echo "Proxy ready (${i}s)."
    break
  fi
  kill -0 "$PROXY_PID" 2>/dev/null || { echo "ERROR: proxy exited. See $PROXY_LOG" >&2; exit 1; }
  sleep 1
done
curl -sf "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1 \
  || { echo "ERROR: proxy did not become ready in 10s" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Run lm_eval against the proxy.
# Greedy decoding (temperature=0).  Item-level logs go alongside.
# ---------------------------------------------------------------------------
LIMIT_ARGS=""
if [ -n "$LM_EVAL_LIMIT" ]; then
  LIMIT_ARGS="--limit $LM_EVAL_LIMIT"
  echo "INFO: limiting to $LM_EVAL_LIMIT examples (LM_EVAL_LIMIT is set)"
fi

# shellcheck disable=SC2086
lm_eval \
  --model gguf \
  --model_args "base_url=http://127.0.0.1:${PROXY_PORT}" \
  --tasks "$task_id" \
  --num_fewshot "$num_fewshot" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUT_DIR/lm_eval/${TASK_KEY}_raw" \
  --gen_kwargs "temperature=0,do_sample=False" \
  $LIMIT_ARGS \
  2>&1 | tee "$OUT_DIR/lm_eval/${TASK_KEY}.log"

# Locate the results JSON (written deep inside output_path) and copy it.
results_json=$(find "$OUT_DIR/lm_eval/${TASK_KEY}_raw" -name 'results*.json' -type f | head -1)
if [ -z "$results_json" ]; then
  echo "ERROR: lm-eval produced no results JSON" >&2
  exit 1
fi
cp "$results_json" "$OUT_DIR/lm_eval/${TASK_KEY}.json"
echo "lm_eval done: $OUT_DIR/lm_eval/${TASK_KEY}.json"
