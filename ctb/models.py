"""Classification models the harness can drive. Two protocols:

- "systemone": POST {base_url}/v1/systemone with {"model", "state", "questions"} and typed answers
  back (Kairos, Jev).
- "chat": an OpenAI-compatible chat-completions LLM (DeepSeek) prompted to return the same typed
  answers as JSON; ctb.kairos.chat_client adapts it to the System One contract.

Adding a model = adding a ModelSpec here (and its key in .env). Everything downstream (questions,
validation, calibration, policy, evaluation) is model-agnostic; decisions, calibration weights and
runs are kept separate per model so one model's results never leak into another's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    provider: str                 # stable id stored with every decision ("kairos", "jev")
    label: str                    # shown in the terminal
    vendor: str
    key_env: str                  # environment variable holding the API key
    base_url_env: str | None
    default_base_url: str
    model_env: str | None         # environment variable that may override the model id
    default_model: str
    max_questions: int | None     # hard limit per request, if the API has one
    max_payload_bytes: int
    retry_statuses: frozenset[int] = field(default_factory=frozenset)
    supports_news_return: bool = False  # Kairos-only task_profile for headline forecasts
    key_url: str = ""
    key_hint: str = ""
    protocol: str = "systemone"   # "systemone" | "chat"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.key_env, "").strip()

    @property
    def base_url(self) -> str:
        raw = os.environ.get(self.base_url_env, "") if self.base_url_env else ""
        return (raw.strip() or self.default_base_url).rstrip("/")

    @property
    def model(self) -> str:
        raw = os.environ.get(self.model_env, "") if self.model_env else ""
        return raw.strip() or self.default_model


MODELS: dict[str, ModelSpec] = {
    "kairos": ModelSpec(
        provider="kairos", label="Kairos", vendor="DMind via Minara Cloud",
        key_env="KAIROS_API_KEY", base_url_env="KAIROS_BASE_URL", default_base_url="https://minara.cloud",
        model_env=None, default_model="DMIND/Kairos",
        max_questions=8, max_payload_bytes=1024 * 1024,
        retry_statuses=frozenset({429, 502, 503, 504}), supports_news_return=True,
        key_url="https://minara.cloud",
        key_hint="The key must belong to a Kairos-enabled group with access to DMIND/Kairos.",
    ),
    "jev": ModelSpec(
        provider="jev", label="Jev", vendor="TypeSafe",
        key_env="TYPESAFE_API_KEY", base_url_env="TYPESAFE_BASE_URL", default_base_url="https://api.typesafe.ai",
        # pinned version by default: the jev-latest alias moves with releases, which would mix model
        # versions inside one evaluation. Override with CTB_JEV_MODEL (e.g. jev-latest).
        model_env="CTB_JEV_MODEL", default_model="jev-1.14.0",
        max_questions=None, max_payload_bytes=32 * 1024 * 1024,
        retry_statuses=frozenset({429, 529, 502, 503, 504}), supports_news_return=False,
        key_url="https://console.typesafe.ai/keys",
        key_hint="Create a key in the TypeSafe console. Requests are billed per input token.",
    ),
    "deepseek": ModelSpec(
        provider="deepseek", label="DeepSeek", vendor="DeepSeek",
        key_env="DEEPSEEK_API_KEY", base_url_env="DEEPSEEK_BASE_URL", default_base_url="https://api.deepseek.com",
        # a general LLM adapted to the question contract, not a trained classifier: judge it on the
        # EVAL/VALID screens like any other model before trusting its probabilities
        model_env="CTB_DEEPSEEK_MODEL", default_model="deepseek-chat",
        max_questions=None, max_payload_bytes=512 * 1024,
        retry_statuses=frozenset({429, 500, 502, 503, 504}), supports_news_return=False,
        key_url="https://platform.deepseek.com/api_keys",
        key_hint="Create a key on the DeepSeek platform. Requests are billed per token.",
        protocol="chat",
    ),
}


def get(provider: str) -> ModelSpec:
    if provider not in MODELS:
        raise ValueError(f"unknown model provider {provider!r}; choose one of {', '.join(MODELS)}")
    return MODELS[provider]
