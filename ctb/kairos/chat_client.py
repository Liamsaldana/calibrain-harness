"""Adapter that drives an OpenAI-compatible chat LLM (DeepSeek) through the System One contract.

The harness asks typed, labellable questions and validates typed answers (ctb.kairos.parse). A
general chat model does not speak /v1/systemone, so this client:

- sends the state and the questions as JSON, with a system prompt that fixes the answer format
  per question type (noul / choice / score), in JSON mode at temperature 0;
- derives only what is arithmetic from the model's own probabilities: `choice` is the argmax,
  `score` is the probability-weighted level, `confidence` is the top probability;
- rescales probabilities that sum to within RENORM_TOL of 1 (LLMs round), and records every
  rescaled question in metadata["renormalized"] so the evaluation can see it. Anything further off
  is passed through unchanged and the strict parser records it as a schema error.

A failed request or an unparseable reply is an ERROR (KairosError), never a negative judgment.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any

import aiohttp

from ..models import ModelSpec
from .client import KairosError, KairosResult, _error_from

RENORM_TOL = 0.10
MAX_OUTPUT_TOKENS = 4000

SYSTEM_PROMPT = """You are a calibrated probabilistic forecaster for crypto and stock perpetual futures.
You receive a JSON object with a market `state` and a set of `questions`. Every question asks about an
observable outcome that will be checked against real prices afterwards, so your probabilities are scored
(Brier score). Be calibrated, not bold: when `state.historical_base_rates` is present, start from those
frequencies and move away from them only as far as specific evidence in the state justifies.

Reply with ONE json object and nothing else, shaped exactly as:
{"answers": {"<question id>": <answer>, ...}}
with one answer for every question id, using these shapes by question type:

- type "noul" (yes/no): {"type": "noul", "noul": P}
  where P is the probability (0 to 1) that the "true" criterion happens.
- type "choice": {"type": "choice", "probabilities": {"<option>": p, ...}}
  with exactly the option keys listed in that question's criteria; probabilities sum to 1.
- type "score": {"type": "score", "probabilities": [p0, p1, ...]}
  one probability per level listed in that question's criteria, in order; they sum to 1.

Use plain decimal numbers. No comments, no prose, no extra keys."""


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _rescale(values: list[float]) -> list[float] | None:
    """Rescale to sum 1 when the model was close; None when it was not (left for the parser)."""
    if not values or not all(_finite(v) and v >= 0 for v in values):
        return None
    total = sum(values)
    if total <= 0 or abs(total - 1) > RENORM_TOL:
        return None
    return [v / total for v in values]


def normalize_answers(raw: dict[str, Any], questions: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map the model's reply onto System One answers. Returns (answers, renormalized question ids)."""
    answers: dict[str, Any] = {}
    renormalized: list[str] = []
    for qid, q in questions.items():
        a = raw.get(qid)
        if not isinstance(a, dict):
            continue  # the parser reports the missing answer
        qtype = q.get("type")
        out: dict[str, Any] = {"type": qtype}
        if qtype == "noul":
            out["noul"] = a.get("noul")
        elif qtype == "choice":
            opts = list(q.get("criteria") or {})
            probs = a.get("probabilities")
            if isinstance(probs, dict) and opts and all(o in probs for o in opts) and set(probs) <= set(opts):
                scaled = _rescale([probs[o] for o in opts])
                if scaled is not None:
                    if abs(sum(probs[o] for o in opts) - 1) > 1e-9:
                        renormalized.append(qid)
                    probs = dict(zip(opts, scaled))
                    best = max(opts, key=lambda o: probs[o])  # ties -> first option, as in the contract
                    out["choice"] = best
                    out["confidence"] = probs[best]
            out["probabilities"] = probs
        elif qtype == "score":
            n = len(q.get("criteria") or [])
            probs = a.get("probabilities")
            if isinstance(probs, dict):
                try:
                    probs = [probs[k] for k in sorted(probs, key=lambda k: int(k))]
                except (ValueError, TypeError):
                    probs = None
            if isinstance(probs, list) and len(probs) == n:
                scaled = _rescale(probs)
                if scaled is not None:
                    if abs(sum(probs) - 1) > 1e-9:
                        renormalized.append(qid)
                    probs = scaled
                    out["score"] = sum(i * p for i, p in enumerate(probs))
                    out["confidence"] = max(probs)
            out["probabilities"] = probs
            if "score" not in out and _finite(a.get("score")):
                out["score"] = a["score"]
        else:
            out.update(a)
        answers[qid] = out
    for qid in raw:
        if qid not in questions:
            answers[qid] = raw[qid]  # surfaced by the parser as an unexpected id
    return answers, renormalized


class ChatClassifierClient:
    """Same interface as SystemOneClient (spec, provider, model, configured, ask)."""

    def __init__(self, spec: ModelSpec, session: aiohttp.ClientSession, *, timeout_s: float = 120, max_retries: int = 2,
                 base_url: str | None = None, api_key_getter=None, model: str | None = None):
        self.spec = spec
        self.provider = spec.provider
        self.url = (base_url or spec.base_url).rstrip("/") + "/chat/completions"
        self._key = api_key_getter or (lambda: spec.api_key)
        self.session = session
        self.model = model or spec.model
        self.timeout = aiohttp.ClientTimeout(total=timeout_s, connect=10)
        self.max_retries = max_retries

    @property
    def configured(self) -> bool:
        return bool(self._key())

    def build_body(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        user = json.dumps({"state": state, "questions": questions}, separators=(",", ":"), allow_nan=False)
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }

    async def ask(self, state: Any, questions: dict[str, Any], *, task_profile: str | None = None) -> KairosResult:
        key = self._key()
        spec = self.spec
        if not key:
            raise KairosError("not_configured", f"{spec.key_env} is not set")
        if task_profile:
            raise KairosError("invalid_request", f"{spec.label} does not support task_profile={task_profile!r}")
        if not questions or (spec.max_questions and len(questions) > spec.max_questions):
            raise KairosError("invalid_request", f"{spec.label} accepts 1-{spec.max_questions or 'n'} questions per request, got {len(questions)}")
        payload = json.dumps(self.build_body(state, questions), separators=(",", ":"), allow_nan=False)
        if len(payload.encode()) > spec.max_payload_bytes:
            raise KairosError("payload_too_large", f"request exceeds the {spec.max_payload_bytes // 1024} KiB limit set for {spec.label}")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        attempt = 0
        t0 = time.perf_counter()
        while True:
            attempt += 1
            try:
                async with self.session.post(self.url, data=payload, headers=headers, timeout=self.timeout) as r:
                    text = await r.text()
                    try:
                        data = json.loads(text) if text else {}
                    except json.JSONDecodeError:
                        data = text
                    if r.status != 200:
                        err = _error_from(r.status, data, spec)
                        if err.retryable and attempt <= self.max_retries:
                            retry_after = r.headers.get("Retry-After")
                            delay = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 2.0 * attempt
                            await asyncio.sleep(min(delay, 30))
                            continue
                        raise err
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt <= self.max_retries:
                    await asyncio.sleep(2.0 * attempt)
                    continue
                raise KairosError("transport", repr(exc)) from exc
            break

        latency = (time.perf_counter() - t0) * 1000
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise KairosError("malformed_response", f"response has no message content: {str(data)[:300]}") from None
        if choice.get("finish_reason") == "length":
            raise KairosError("malformed_response", "reply was cut off at max_tokens")
        try:
            obj = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            raise KairosError("malformed_response", f"reply is not JSON: {str(content)[:300]}") from None
        raw = obj.get("answers") if isinstance(obj, dict) else None
        if not isinstance(raw, dict):
            raise KairosError("malformed_response", f"reply has no answers object: {str(content)[:300]}")
        answers, renormalized = normalize_answers(raw, questions)

        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                 "cache_hit_tokens": u.get("prompt_cache_hit_tokens"), "cache_miss_tokens": u.get("prompt_cache_miss_tokens")}
        meta = {"model": data.get("model") or self.model, "request_id": data.get("id"), "protocol": "chat",
                "system_fingerprint": data.get("system_fingerprint"), "renormalized": renormalized}
        return KairosResult(answers=answers, usage=usage, metadata=meta, latency_ms=latency, attempts=attempt,
                            request_bytes=len(payload.encode()), raw=data)
