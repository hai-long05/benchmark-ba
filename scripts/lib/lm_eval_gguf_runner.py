"""Run lm-eval-harness against a GGUF model via llama-cpp-python directly.

Replaces scripts/40_lm_eval.sh's llama-server + reverse-proxy hack. The proxy
emulated lm_eval's gguf-backend HTTP protocol token-by-token via /v1/completions
and used a buggy fallback (records the model's *generated* token logprob when the
target token is missing from top_logprobs) that drove HellaSwag from ~0.73 down
to ~0.40 on Llama-3.1-8B-Q4_K_S.

This adapter loads the GGUF in-process and computes exact per-token logprobs
from the full softmax — the same scoring path lm_eval's `hf` backend uses for
non-GGUF models. Multiple-choice loglikelihood tasks (HellaSwag/MMLU/TQA-MC2)
share one context across N continuations, so the adapter caches the
post-context KV state via Llama.save_state()/load_state() and only forwards
the continuation tokens per choice — ~3-4× speedup vs reset-and-eval.

Used as a custom lm_eval model via:
    lm_eval --model_args 'pretrained=<gguf>,n_ctx=2048,tokenizer_repo=<hf-id>,...' \\
            --include_path scripts/lib --model gguf_local ...

Registered with @register_model('gguf_local').

Chat-template policy (replicates Kurt 2026, §3.1):
    When `tokenizer_repo` is set, an HF AutoTokenizer is loaded in parallel to
    the GGUF (HF tokenizer for prompt formatting only — never for inference).
    apply_chat_template() then dispatches to the HF tokenizer, which is the
    canonical source of the model's default chat template. Special tokens
    introduced by the template (e.g. <|start_header_id|>) are recognised at
    encode time via `special=True`, otherwise they would shatter into BPE
    fragments and the loglikelihood would be meaningless.

    For Qwen3, `chat_template_kwargs={enable_thinking: False}` is forwarded to
    apply_chat_template so the loglikelihood path does not get prefixed with
    <think>...</think> tokens (which would render the continuation logprob
    near-zero).
"""
from __future__ import annotations

import json as _json
import math
from typing import Any

import numpy as np
from llama_cpp import Llama
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("gguf_local")
class GGUFLocal(LM):
    """In-process GGUF scorer + generator using llama-cpp-python.

    Implements the four lm_eval LM methods used by the project's task set:
      - loglikelihood: HellaSwag, MMLU, TruthfulQA-MC2 (with prefix-cache
        fast path for shared-context groups, ~3-4× speedup)
      - loglikelihood_rolling: WikiText-2 perplexity (sliding-window)
      - generate_until: GSM8K, IFEval (greedy, temperature=0)
      - apply_chat_template / chat_template / tokenizer_name: chat-template
        rendering via parallel-loaded HF AutoTokenizer
    """

    def __init__(
        self,
        pretrained: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        batch_size: int = 1,
        max_length: int | None = None,
        tokenizer_repo: str | None = None,
        chat_template_kwargs: str | None = None,
        prefix_cache: bool | str = True,
        **kwargs: Any,
    ):
        super().__init__()
        self._pretrained = pretrained
        self._n_ctx = int(n_ctx)
        # llama-cpp-python: logits_all=True records logits for every token in the
        # batch, which is exactly what loglikelihood scoring needs. Without it,
        # only the last token's logits are kept and we cannot reconstruct the
        # per-token logprobs of an arbitrary continuation.
        self._llm = Llama(
            model_path=pretrained,
            n_ctx=self._n_ctx,
            n_threads=n_threads,
            n_gpu_layers=int(n_gpu_layers),
            logits_all=True,
            verbose=bool(verbose),
        )
        self._batch_size = int(batch_size)
        self._max_length = int(max_length) if max_length else self._n_ctx
        self._eot = self._llm.token_eos()

        # Prefix-cache toggle. Default ON (3-4× speedup on tasks with shared
        # contexts: HellaSwag 1ctx→4cont, MMLU 1ctx→4cont, TruthfulQA-MC2
        # 1ctx→N_choices). Set prefix_cache=False (or 'no' / 'off' / '0') in
        # model_args to force the legacy reset-and-full-eval path; used for
        # numerical-equivalence validation smokes.
        if isinstance(prefix_cache, str):
            self._prefix_cache = prefix_cache.lower() not in ("0", "no", "off", "false", "")
        else:
            self._prefix_cache = bool(prefix_cache)

        # HF tokenizer, loaded only if tokenizer_repo is given. Used exclusively
        # for apply_chat_template() — the HF tokenizer renders chat-history dicts
        # to a template string that lm-eval then passes back through our
        # tok_encode / loglikelihood path. Inference still runs through the
        # GGUF tokenizer above; the HF tokenizer never sees inference tensors.
        self._hf_tokenizer = None
        self._tokenizer_repo = tokenizer_repo
        if tokenizer_repo:
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "tokenizer_repo set but `transformers` not installed — "
                    "add transformers to requirements.txt"
                ) from e
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_repo, trust_remote_code=False
            )

        # Optional kwargs forwarded into apply_chat_template (e.g.
        # `{"enable_thinking": false}` for Qwen3 to suppress reasoning blocks
        # during loglikelihood scoring). lm_eval passes model_args as strings,
        # so we accept either a JSON-encoded string or a dict.
        self._chat_template_kwargs: dict[str, Any] = {}
        if chat_template_kwargs:
            if isinstance(chat_template_kwargs, dict):
                self._chat_template_kwargs = dict(chat_template_kwargs)
            else:
                try:
                    self._chat_template_kwargs = _json.loads(chat_template_kwargs)
                except _json.JSONDecodeError as e:
                    raise ValueError(
                        f"chat_template_kwargs must be JSON, got: {chat_template_kwargs!r}"
                    ) from e

    # ---- lm_eval API ---------------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        return self._eot

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def tok_encode(self, string: str, add_bos: bool = False) -> list[int]:
        # `special=True` so that template markers like <|start_header_id|> are
        # recognised as single special tokens rather than shattered into BPE
        # pieces. Safe for non-templated text too: the GGUF tokenizer falls
        # back to normal BPE for substrings that don't match a special token.
        return self._llm.tokenize(string.encode("utf-8"), add_bos=add_bos, special=True)

    def tok_decode(self, tokens: list[int]) -> str:
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")

    # ---- core scoring --------------------------------------------------------

    def _logprob_from_logits(self, logits: np.ndarray, tok_id: int) -> tuple[float, bool]:
        """Compute log P(tok_id | preceding context) from a single logits row.

        Returns (logprob, is_argmax). Uses stable log-softmax: the max-shift
        trick avoids overflow when logits have large magnitudes (Llama's
        unembedding can produce values in the [-30, +40] range).
        """
        mx = float(np.max(logits))
        lse = mx + math.log(float(np.sum(np.exp(logits - mx))))
        lp = float(logits[tok_id]) - lse
        is_argmax = int(np.argmax(logits)) == int(tok_id)
        return lp, is_argmax

    def _score_one_legacy(self, ctx_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
        """Reset-and-eval-from-scratch path (no prefix cache).

        Used only when prefix_cache=False is set in model_args. Kept for
        numerical-equivalence validation against the prefix-cached path.
        """
        full = ctx_ids + cont_ids
        self._llm.reset()
        self._llm.eval(full)
        scores = self._llm.scores[: len(full)]
        cont_start = len(ctx_ids)
        sum_lp = 0.0
        all_greedy = True
        for i, tok_id in enumerate(cont_ids):
            row = cont_start + i - 1
            if row < 0 or row >= scores.shape[0]:
                return float("-inf"), False
            lp, is_arg = self._logprob_from_logits(scores[row], tok_id)
            sum_lp += lp
            if not is_arg:
                all_greedy = False
        return sum_lp, all_greedy

    def _score_continuations_with_shared_ctx(
        self,
        ctx_ids: list[int],
        continuations: list[list[int]],
    ) -> list[tuple[float, bool]]:
        """Score N continuations that all share the same context, reusing the
        KV cache directly via n_tokens-pointer reset. This is the fast path.

        Performance-critical: an earlier version used `save_state()/load_state()`
        which copies the entire scores matrix (n_ctx × vocab = 2048 × 128256 ×
        4 B = 1 GiB for Llama-3.1) PER CONTINUATION. With 4 continuations per
        HellaSwag item and 100k+ items per slot, that path produced terabytes
        of memory copies and dominated wall-clock by ~50-100×. The fix here
        relies on llama-cpp-python's eval() which calls kv_cache_seq_rm(-1,
        self.n_tokens, -1) at its start: setting n_tokens = ctx_n before
        eval(cont) truncates the KV cache to "after ctx" and appends cont
        at positions [ctx_n .. ctx_n+len(cont)-1] — no copies, no allocations.

        Layout per continuation:
          1. (once per group) reset, eval(ctx) — KV cache now holds ctx tokens
             at positions [0..ctx_n-1], scores[ctx_n-1] is the logit row
             that predicts cont[0].
          2. (per cont) set n_tokens=ctx_n (truncates KV to ctx),
             eval(cont) appends and writes scores[ctx_n..ctx_n+len(cont)-1],
             read logits at positions [ctx_n-1 .. ctx_n+len(cont)-2].

        The scores at [0..ctx_n-1] are the same Python ndarray object across
        all continuations of this group — no copy needed. Only the rows
        [ctx_n..] get overwritten by each cont's eval, but we read them
        before the next iteration overwrites them.
        """
        # 1. Build the context KV state once.
        self._llm.reset()
        self._llm.eval(ctx_ids)
        ctx_n = len(ctx_ids)
        # Snapshot the ctx-tail logits row (predicts cont[0]) before subsequent
        # eval(cont) overwrites scores[ctx_n-1]. This is the only memory copy
        # we need, and it's a single (vocab,)-shaped float32 row (~512 KB),
        # not the full 1 GB scores matrix.
        ctx_last_logits = self._llm.scores[ctx_n - 1].copy() if ctx_n > 0 else None

        out: list[tuple[float, bool]] = []
        for cont_ids in continuations:
            if not cont_ids:
                out.append((0.0, True))
                continue

            # Reset n_tokens so eval() truncates the KV cache to ctx and appends
            # cont. No state copy. eval() internally calls kv_cache_seq_rm(-1,
            # self.n_tokens, -1) which is a constant-time pointer-rewind.
            self._llm.n_tokens = ctx_n
            self._llm.eval(cont_ids)

            # Logits-row that predicted cont[i] is row (ctx_n + i - 1):
            # - For i=0: row ctx_n-1 was produced during the ctx eval. We
            #   snapshotted it as ctx_last_logits because eval(cont_A) would
            #   overwrite it with cont_A's last logit if we read it later.
            #   Wait — actually eval(cont) writes to rows [ctx_n .. ctx_n+len(cont)-1],
            #   NOT to row ctx_n-1. So scores[ctx_n-1] is preserved across
            #   the cont evals. ctx_last_logits is a defensive belt-and-braces.
            # - For i>0: row ctx_n+i-1 was produced during the cont eval.
            sum_lp = 0.0
            all_greedy = True
            for i, tok_id in enumerate(cont_ids):
                row_idx = ctx_n + i - 1
                if row_idx < 0:
                    # Pathological: empty ctx. Continuation token at position 0
                    # has no preceding context to predict from.
                    sum_lp = float("-inf")
                    all_greedy = False
                    break
                if i == 0 and ctx_last_logits is not None:
                    # Use the snapshotted ctx-tail row (defensive — should be
                    # identical to scores[ctx_n-1] since eval(cont) only writes
                    # rows >= ctx_n, but safer against future llama-cpp-python
                    # changes).
                    logits = ctx_last_logits
                else:
                    logits = self._llm.scores[row_idx]
                lp, is_arg = self._logprob_from_logits(logits, tok_id)
                sum_lp += lp
                if not is_arg:
                    all_greedy = False
            out.append((sum_lp, all_greedy))
        return out

    def _tokenize_request(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Tokenize a (context, continuation) pair with BPE-correct word boundaries.

        Critical: BPE tokenizers produce different token sequences for
        ``tokenize(ctx) + tokenize(cont)`` versus ``tokenize(ctx + cont)`` when
        the boundary falls inside a word piece (e.g. continuation `" The"`
        merges with a preceding period differently when tokenized in isolation
        vs. as part of the joined string). lm-eval's TemplateLM uses the
        "tokenize joined, then find the split via re-tokenizing context"
        convention; we mirror it so HellaSwag/MMLU/TQA-MC2 logprobs match the
        harness's reference implementation.

        Returns (ctx_ids, cont_ids) where the concatenation reproduces the
        joint tokenization of (context + continuation). Truncates ctx_ids
        from the left if the joined sequence overflows max_length.

        BOS rationale: when a chat template is applied by lm-eval, the
        templated context already starts with the model's BOS token (e.g.
        <|begin_of_text|> for Llama-3.1) which `special=True` recognises as
        a single special token. Adding another BOS via add_bos=True would
        double it and shift every loglikelihood by the (large negative)
        logprob of a duplicate BOS.
        """
        whole_ids = self._llm.tokenize(
            (context + continuation).encode("utf-8"), add_bos=False, special=True
        )
        ctx_only_ids = self._llm.tokenize(
            context.encode("utf-8"), add_bos=False, special=True
        )
        # The continuation tokens are the suffix of the joint tokenization
        # that follows the context-prefix. Use the length of the standalone
        # ctx tokenization as the split index — this is the same heuristic
        # TemplateLM._encode_pair uses in v0.4.9.2.
        ctx_len = len(ctx_only_ids)
        if ctx_len > len(whole_ids):
            # Pathological: tokenizing (ctx+cont) jointly produced FEWER
            # tokens than tokenizing ctx alone. Can happen if the boundary
            # creates a special token. Fall back to the joint sequence with
            # an empty cont; the caller will treat it as no-op (0.0).
            return whole_ids, []
        ctx_ids = whole_ids[:ctx_len]
        cont_ids = whole_ids[ctx_len:]

        full_len = len(ctx_ids) + len(cont_ids)
        if full_len > self._max_length:
            overflow = full_len - self._max_length
            ctx_ids = ctx_ids[overflow:]
        return ctx_ids, cont_ids

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        """Score every (context, continuation) pair in `requests`.

        Performance optimization: when `prefix_cache=True` (default), groups
        consecutive requests by identical context and scores the N continuations
        of a context with a single ctx-eval + N short cont-evals (plus state
        save/load between conts). For HellaSwag (1 ctx → 4 conts), MMLU
        (1 ctx → 4 conts) and TruthfulQA-MC2 (1 ctx → variable conts) this
        gives ~3.5× speedup vs full-reset-per-call.

        lm-eval emits requests in document order, so the 4 HellaSwag
        continuations of doc D arrive consecutively as 4 separate Instances
        with .args[0] == identical context string. Same for MMLU and TQA-MC2.
        We exploit that ordering directly.
        """
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:  # pragma: no cover
            tqdm = lambda x, **k: x  # noqa: E731

        out: list[tuple[float, bool]] = []
        iterator = requests if disable_tqdm else tqdm(
            requests, desc=f"loglikelihood (gguf_local{', cached' if self._prefix_cache else ', no-cache'})"
        )

        if not self._prefix_cache:
            for req in iterator:
                ctx, cont = req.args
                ctx_ids, cont_ids = self._tokenize_request(ctx, cont)
                if not cont_ids:
                    out.append((0.0, True))
                    continue
                out.append(self._score_one_legacy(ctx_ids, cont_ids))
            return out

        # Prefix-cached path. Walk requests, accumulate same-context runs,
        # flush each run with shared-ctx scoring.
        pending_ctx_str: str | None = None
        pending_ctx_ids: list[int] | None = None
        pending_conts: list[list[int]] = []

        def flush() -> None:
            nonlocal pending_ctx_str, pending_ctx_ids, pending_conts
            if pending_ctx_ids is None or not pending_conts:
                pending_ctx_str = None
                pending_ctx_ids = None
                pending_conts = []
                return
            results = self._score_continuations_with_shared_ctx(pending_ctx_ids, pending_conts)
            out.extend(results)
            pending_ctx_str = None
            pending_ctx_ids = None
            pending_conts = []

        for req in iterator:
            ctx, cont = req.args
            # Joint tokenization is required for BPE word-boundary correctness
            # (see _tokenize_request docstring). We can't tokenize cont alone
            # and concatenate — that produces different token IDs at the
            # boundary and shifts logprobs measurably. So we re-tokenize the
            # joined string per request, but skip the costly ctx-eval when
            # the ctx string matches the pending one.
            ctx_ids, cont_ids = self._tokenize_request(ctx, cont)
            if not cont_ids:
                # Empty cont (rare, see pathological branch in _tokenize_request).
                # Flush any pending and emit a no-op score.
                flush()
                out.append((0.0, True))
                continue
            if ctx != pending_ctx_str:
                flush()
                pending_ctx_str = ctx
                pending_ctx_ids = ctx_ids
            elif len(ctx_ids) != len(pending_ctx_ids or []):
                # Same ctx string but different ctx token count after joint
                # tokenization — can happen if the cont starts with a char that
                # shifts the boundary tokenization. Flush and start fresh so
                # the cached ctx_ids matches what cont_ids was carved from.
                flush()
                pending_ctx_str = ctx
                pending_ctx_ids = ctx_ids
            # If a single (ctx + cont) overflows max_length, _tokenize_request
            # has already truncated ctx from the left; we still need to ensure
            # the cached ctx_ids is consistent with the newly-truncated one.
            # Easiest correct path: drop into legacy for the truncated case.
            if pending_ctx_ids is not None and len(pending_ctx_ids) + len(cont_ids) > self._max_length:
                # Should not happen given _tokenize_request truncated, but defensive.
                flush()
                out.append(self._score_one_legacy(ctx_ids, cont_ids))
                continue
            pending_conts.append(cont_ids)

        flush()
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        """Compute total log-likelihood of a long string with no separate context.

        Used by perplexity-style tasks (wikitext). lm-eval expects, per request, the
        sum of log-probabilities of the full token sequence under the model. Strategy:
        slide a context-size window with stride n_ctx//2; for each window, score every
        token whose position is past the previous window's last *kept* index. This is
        the standard "halve the context, keep the second half's logprobs" trick used
        by HuggingFace's perplexity recipe.
        """
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:  # pragma: no cover
            tqdm = lambda x, **k: x  # noqa: E731

        out: list[float] = []
        # Cap window at max_length-1 so we always have a row of logits to score against.
        win = max(2, self._max_length - 1)
        stride = max(1, win // 2)

        iterator = requests if disable_tqdm else tqdm(requests, desc="loglikelihood_rolling (gguf_local)")
        for req in iterator:
            string = req.args[0]
            tokens = self._llm.tokenize(string.encode("utf-8"), add_bos=True, special=True)
            n = len(tokens)
            if n < 2:
                out.append(0.0)
                continue

            total_lp = 0.0
            scored_until = 0  # index after which we still need to score tokens
            start = 0
            while start < n - 1:
                end = min(start + win, n)
                chunk = tokens[start:end]
                self._llm.reset()
                self._llm.eval(chunk)
                scores = self._llm.scores[: len(chunk)]
                # Within this window, token at local position i was predicted by
                # logits at local row i-1. We must score token absolute positions
                # max(start+1, scored_until) up through end-1, but skip the first
                # one if start == 0 (no logit predicts the very first token).
                first_abs = max(start + 1, scored_until)
                for abs_pos in range(first_abs, end):
                    local_i = abs_pos - start
                    row = local_i - 1
                    if row < 0 or row >= scores.shape[0]:
                        continue
                    logits = scores[row]
                    mx = float(np.max(logits))
                    lse = mx + math.log(float(np.sum(np.exp(logits - mx))))
                    total_lp += float(logits[tokens[abs_pos]]) - lse
                scored_until = end
                if end >= n:
                    break
                start += stride
            out.append(total_lp)
        return out

    # ---- generation (GSM8K, IFEval) -----------------------------------------

    def _generate_one(
        self,
        context: str,
        until: list[str],
        max_new_tokens: int,
    ) -> str:
        """Greedy-generate from `context` until any of `until` appears in the suffix
        or `max_new_tokens` is reached.

        Tokenises the prompt ourselves with `add_bos=False, special=True` so the
        chat-template's leading BOS (e.g. <|begin_of_text|> for Llama-3.1) is
        recognised as a single special token instead of being doubled by
        llama-cpp-python's create_completion (which auto-prepends BOS by
        default and emits the "Detected duplicate leading <|begin_of_text|>"
        warning, degrading IFEval / GSM8K generation quality).

        Generation runs greedy (temperature=0) per the project's spec §4.4.
        """
        prompt_tokens = self._llm.tokenize(
            context.encode("utf-8"), add_bos=False, special=True
        )
        out = self._llm.create_completion(
            prompt=prompt_tokens,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repeat_penalty=1.0,
            stop=list(until) if until else [],
            echo=False,
            stream=False,
        )
        text = out["choices"][0]["text"]
        # Defensive: truncate at the earliest occurrence of any until-string.
        for u in until:
            if u and u in text:
                text = text.split(u, 1)[0]
        return text

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        """Generate completions for each request.

        Each lm_eval Instance.args is (context, gen_kwargs_dict). gen_kwargs
        carries `until` (list of stop strings), `do_sample`, `temperature`,
        `max_gen_toks`, etc. We honour `until` and `max_gen_toks` and force
        greedy decoding (temperature=0) per the project's spec §4.4.
        """
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:  # pragma: no cover
            tqdm = lambda x, **k: x  # noqa: E731

        out: list[str] = []
        iterator = requests if disable_tqdm else tqdm(requests, desc="generate_until (gguf_local)")
        for req in iterator:
            args = req.args
            context = args[0]
            gen_kwargs = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
            until = gen_kwargs.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            text = self._generate_one(context, list(until), max_new)
            out.append(text)
        return out

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:  # type: ignore[override]
        """Render a chat-history list to a string under the model's default template.

        Loaded from the HF tokenizer of `tokenizer_repo` (set via model_args). This
        replicates Kurt 2026 §3.1: "prompts were formatted using the default
        Llama-3.1-8B-Instruct chat template". For Mistral-Instruct and Qwen3 the
        same convention applies — each model's own default template, sourced from
        its HF repo.

        Forwards `chat_template_kwargs` (also from model_args) into
        tokenizer.apply_chat_template — used for Qwen3's `enable_thinking=False`
        to suppress reasoning blocks during loglikelihood scoring.
        """
        if self._hf_tokenizer is None:
            raise RuntimeError(
                "apply_chat_template called but tokenizer_repo not set in model_args. "
                "Pass tokenizer_repo=<hf-id> when running with --apply_chat_template."
            )
        return self._hf_tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self._chat_template_kwargs,
        )

    @property
    def tokenizer_name(self) -> str:
        # lm-eval's chat-template logic queries this to log which tokenizer
        # produced the prompts. Fall back to the GGUF path so it never crashes.
        return self._tokenizer_repo or self._pretrained

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        """Return the chat-template string lm-eval should record / apply.

        Signature matches LM base class in lm_eval/api/model.py: a METHOD (not
        a property) accepting an optional bool/str argument. lm-eval calls it
        as `lm.chat_template(apply_chat_template_arg)` from
        evaluator.simple_evaluate to log which template was applied.

        - If apply_chat_template was disabled at the CLI, lm-eval passes
          False and we return None (consistent with "no template").
        - If enabled (True or a name), we return the HF tokenizer's template
          string when a tokenizer_repo was configured; otherwise None.

        We don't differentiate by `chat_template` value (True vs a named
        template) — we always use the HF tokenizer's default template,
        which is what Kurt 2026 §3.1 specifies.
        """
        if not chat_template:
            return None
        if self._hf_tokenizer is None:
            return None
        return getattr(self._hf_tokenizer, "chat_template", None)
