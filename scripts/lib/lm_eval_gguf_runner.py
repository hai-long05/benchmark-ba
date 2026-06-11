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
    lm_eval --model_args 'pretrained=<gguf>,n_ctx=2048,...' \\
            --include_path scripts/lib --model gguf_local ...

Registered with @register_model('gguf_local').
"""
from __future__ import annotations

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
        return self._llm.tokenize(string.encode("utf-8"), add_bos=add_bos, special=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")

    # ---- core scoring --------------------------------------------------------

    def _score_continuation(self, context: str, continuation: str) -> tuple[float, bool]:
        """Score continuation tokens given context. Returns (sum_logprob, is_greedy).

        is_greedy: whether each continuation token was the argmax at its position.
                   Used by hellaswag's `acc` metric (acc_norm uses sum_logprob).
        """
        # Tokenise. BOS only on the very first prompt position; the harness's
        # default convention is no extra BOS for either context or continuation
        # because the dataset already includes any necessary preface.
        ctx_ids = self._llm.tokenize(context.encode("utf-8"), add_bos=True, special=False)
        cont_ids = self._llm.tokenize(continuation.encode("utf-8"), add_bos=False, special=False)

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
        # Used by perplexity-style tasks (wikitext). Scores the full string under
        # an empty context. Not exercised by the smoke set; implement when needed.
        raise NotImplementedError(
            "gguf_local: loglikelihood_rolling not implemented yet (needed for wikitext PPL)"
        )

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

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:  # type: ignore[override]
        # HellaSwag and other loglikelihood tasks do not invoke this; raw text scoring is correct.
        # Implement only if a future task needs chat-templated prompts.
        raise NotImplementedError(
            "gguf_local: apply_chat_template not implemented; only raw-text loglikelihood tasks are supported"
        )
