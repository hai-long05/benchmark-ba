"""Run lm-eval-harness against a GGUF model via llama-cpp-python directly.

Replaces scripts/40_lm_eval.sh's llama-server + reverse-proxy hack. The proxy
emulated lm_eval's gguf-backend HTTP protocol token-by-token via /v1/completions
and used a buggy fallback (records the model's *generated* token logprob when the
target token is missing from top_logprobs) that drove HellaSwag from ~0.73 down
to ~0.40 on Llama-3.1-8B-Q4_K_S.

This adapter loads the GGUF in-process and computes exact per-token logprobs
from the full softmax — the same scoring path lm_eval's `hf` backend uses for
non-GGUF models. Result: correct numbers, ~10x faster, no HTTP, no proxy.

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

import datetime as _dt
import json as _json
import math
from typing import Any

import numpy as np
from llama_cpp import Llama
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("gguf_local")
class GGUFLocal(LM):
    """In-process GGUF loglikelihood scorer using llama-cpp-python.

    Implements the three lm_eval LM methods used by 0-shot multiple-choice tasks
    (hellaswag, mmlu, truthfulqa_mc2): loglikelihood, loglikelihood_rolling,
    generate_until. The first is what HellaSwag uses. The other two are stubs
    sufficient for the smoke-test scope; extend if you need GSM8K / IFEval.
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

        # HF tokenizer, loaded only if tokenizer_repo is given. Used exclusively
        # for apply_chat_template() and for special-token-aware encoding when a
        # template-formatted prompt comes back into tok_encode/_score_continuation.
        # Inference still runs through the GGUF tokenizer above.
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

    def _score_continuation(self, context: str, continuation: str) -> tuple[float, bool]:
        """Score continuation tokens given context. Returns (sum_logprob, is_greedy).

        is_greedy: whether each continuation token was the argmax at its position.
                   Used by hellaswag's `acc` metric (acc_norm uses sum_logprob).
        """
        # Tokenise. `special=True` so chat-template markers (when --apply_chat_template
        # is on) survive tokenisation as single special tokens. `add_bos=False`
        # for both context and continuation: when a chat template is applied
        # by lm-eval, the template string already starts with the model's BOS
        # token (e.g. <|begin_of_text|> for Llama-3.1) which is encoded as a
        # special token via special=True. Adding another BOS via add_bos=True
        # produces the "Added a BOS token... the prompt also starts with a BOS
        # token" warning from llama.cpp and shifts every loglikelihood score by
        # the (large negative) logprob of a duplicate BOS. For non-templated
        # tasks (wikitext), loglikelihood_rolling adds BOS itself when
        # appropriate. The harness's dataset-level convention also assumes the
        # template / preface already provides any necessary BOS — see Kurt
        # 2026 §3.1 for the same convention.
        ctx_ids = self._llm.tokenize(context.encode("utf-8"), add_bos=False, special=True)
        cont_ids = self._llm.tokenize(continuation.encode("utf-8"), add_bos=False, special=True)

        if not cont_ids:
            return 0.0, True

        # Truncate the context from the LEFT if the joined sequence would overflow.
        # We MUST keep the full continuation — we score every token of it.
        full = ctx_ids + cont_ids
        if len(full) > self._max_length:
            overflow = len(full) - self._max_length
            ctx_ids = ctx_ids[overflow:]
            full = ctx_ids + cont_ids

        # Reset cache state and evaluate the full sequence.
        self._llm.reset()
        self._llm.eval(full)

        # llama-cpp-python stores logits for tokens [0, n_tokens) in self._scores
        # when logits_all=True. Each row scores the *next* token, so the logit
        # for token at position i was produced from prefix [0..i-1], i.e. by
        # scoring row i-1.
        scores = self._llm.scores[: len(full)]  # (T, vocab)

        # Continuation tokens occupy positions [len(ctx_ids), len(full)). The
        # logit row that predicted continuation token at position p is row p-1.
        cont_start = len(ctx_ids)
        sum_lp = 0.0
        all_greedy = True
        for i, tok_id in enumerate(cont_ids):
            row = cont_start + i - 1
            if row < 0 or row >= scores.shape[0]:
                # Should not happen for non-empty contexts; defensive.
                return float("-inf"), False
            logits = scores[row]
            # Stable log-softmax.
            mx = float(np.max(logits))
            lse = mx + math.log(float(np.sum(np.exp(logits - mx))))
            lp = float(logits[tok_id]) - lse
            sum_lp += lp
            if int(np.argmax(logits)) != int(tok_id):
                all_greedy = False
        return sum_lp, all_greedy

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        # `requests` is a list of lm_eval Instance objects whose .args is (context, continuation).
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:  # pragma: no cover
            tqdm = lambda x, **k: x  # noqa: E731

        out: list[tuple[float, bool]] = []
        iterator = requests if disable_tqdm else tqdm(requests, desc="loglikelihood (gguf_local)")
        for req in iterator:
            ctx, cont = req.args
            out.append(self._score_continuation(ctx, cont))
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

        Uses llama-cpp-python's create_completion in non-streaming mode with
        temperature=0 (greedy) so output is deterministic. The `stop` parameter
        of llama-cpp-python is honoured server-side, but we also defensively
        truncate the result on each `until` token after the fact in case the
        model emitted a stop string mid-token (rare but possible with BPE).
        """
        # llama-cpp-python's `stop` accepts a list of strings; pass the same list.
        # Empty stops list is fine.
        out = self._llm.create_completion(
            prompt=context,
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
