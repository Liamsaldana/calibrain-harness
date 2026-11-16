"""DeepSeek adapter: a chat LLM's JSON reply is mapped onto the System One answer contract and then
passes the same strict parser as Kairos and Jev. No network: the HTTP session is faked."""

import asyncio
import json

import pytest

from ctb.kairos.chat_client import ChatClassifierClient, normalize_answers
from ctb.kairos.client import KairosError, SystemOneClient, make_client
from ctb.kairos.parse import validate
from ctb.kairos.questions import build_questions
from ctb.models import get


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status, self._body, self.headers = status, body, headers or {}

    async def text(self):
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.sent.append({"url": url, "body": json.loads(data), "headers": headers})
        return _Resp(*self.responses.pop(0))


def _reply(answers, finish="stop"):
    return {"id": "req-1", "model": "deepseek-chat",
            "choices": [{"message": {"role": "assistant", "content": json.dumps({"answers": answers})}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300}}


def _good_answers():
    return {
        "long_bracket": {"type": "choice", "probabilities": {"target_first": 0.2, "stop_first": 0.25, "neither": 0.55}},
        "short_bracket": {"type": "choice", "probabilities": {"target_first": 0.24, "stop_first": 0.21, "neither": 0.5}},  # sums to 0.95
        "return_distribution": {"type": "choice", "probabilities": {f"bin_{i}": p for i, p in enumerate([.05, .1, .2, .3, .2, .1, .05])}},
        "direction": {"type": "noul", "noul": 0.52},
        "big_move": {"type": "noul", "noul": 0.4},
        "mean_reversion": {"type": "noul", "noul": 0.35},
        "setup_quality": {"type": "score", "probabilities": [0.3, 0.4, 0.2, 0.1, 0.0]},
    }


def _client(session, key="sk-test"):
    return ChatClassifierClient(get("deepseek"), session, api_key_getter=lambda: key, max_retries=1)


def test_spec_and_factory_pick_the_chat_adapter():
    spec = get("deepseek")
    assert spec.protocol == "chat" and spec.key_env == "DEEPSEEK_API_KEY" and spec.model == "deepseek-chat"
    assert isinstance(make_client(spec, _Session([])), ChatClassifierClient)
    assert isinstance(make_client(get("kairos"), _Session([])), SystemOneClient)


def test_reply_passes_the_strict_parser():
    qs = build_questions(60, 90, 120)
    session = _Session([(200, _reply(_good_answers()))])
    res = asyncio.run(_client(session).ask({"last_price": 100}, qs))
    parsed, issues = validate(res.answers, qs)
    assert issues == []
    assert set(parsed) == set(qs)
    # the near-miss sum was rescaled and recorded, the argmax and score were derived
    assert res.metadata["renormalized"] == ["short_bracket"]
    assert abs(sum(parsed["short_bracket"]["probs"].values()) - 1) < 1e-9
    assert parsed["long_bracket"]["choice"] == "neither"
    assert abs(parsed["setup_quality"]["score"] - (0.4 + 0.4 + 0.3)) < 1e-9
    assert res.usage["input_tokens"] == 1200


def test_request_is_json_mode_temperature_zero_with_bearer_key():
    qs = build_questions(60, 90, 120)
    session = _Session([(200, _reply(_good_answers()))])
    asyncio.run(_client(session).ask({"last_price": 100}, qs))
    sent = session.sent[0]
    assert sent["url"] == "https://api.deepseek.com/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer sk-test"
    body = sent["body"]
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0
    assert "json" in body["messages"][0]["content"].lower()  # JSON mode requires the word in the prompt
    user = json.loads(body["messages"][1]["content"])
    assert set(user["questions"]) == set(qs)


def test_far_off_probabilities_are_not_repaired():
    qs = {"long_bracket": build_questions(60, 90, 120)["long_bracket"]}
    raw = {"long_bracket": {"type": "choice", "probabilities": {"target_first": 0.5, "stop_first": 0.5, "neither": 0.5}}}
    answers, renorm = normalize_answers(raw, qs)
    assert renorm == []
    _, issues = validate(answers, qs)
    assert any("sum" in i for i in issues)


def test_errors_are_errors_never_answers():
    qs = build_questions(60, 90, 120)
    with pytest.raises(KairosError) as e:
        asyncio.run(_client(_Session([(200, {"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]})])).ask({}, qs))
    assert e.value.kind == "malformed_response"
    with pytest.raises(KairosError) as e:
        asyncio.run(_client(_Session([(200, _reply(_good_answers(), finish="length"))])).ask({}, qs))
    assert e.value.kind == "malformed_response"
    with pytest.raises(KairosError) as e:
        asyncio.run(_client(_Session([(401, {"error": {"message": "Authentication Fails"}})])).ask({}, qs))
    assert e.value.kind == "auth"
    with pytest.raises(KairosError) as e:
        asyncio.run(_client(_Session([]), key="").ask({}, qs))
    assert e.value.kind == "not_configured"


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    qs = build_questions(60, 90, 120)
    session = _Session([(429, {"error": {"message": "rate limit"}}), (200, _reply(_good_answers()))])
    res = asyncio.run(_client(session).ask({}, qs))
    assert res.attempts == 2


def test_news_profile_is_refused():
    with pytest.raises(KairosError):
        asyncio.run(_client(_Session([])).ask({}, {"x": {"type": "noul", "instructions": "?"}}, task_profile="news_return"))
