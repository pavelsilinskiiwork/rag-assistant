import json
import logging
import time
from fastapi.testclient import TestClient

import agent
import observability
from llm_provider import LLMResult
from api import app
from fakes import FakeProvider, text_result, tool_result


CHUNK = {"text": "Meal allowance is $75.", "source": "policy.pdf", "page": 3, "distance": 0.5}


def test_estimate_cost():
    assert observability.estimate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) == 12.0
    assert observability.estimate_cost("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    assert observability.estimate_cost("unknown-model", 1_000_000, 0) == 2.0  # default price


def test_haiku_long_prompt_uses_higher_tier():
    short = observability.estimate_cost("claude-haiku-5-5", 100_000, 1_000_000)
    long = observability.estimate_cost("claude-haiku-5-5", 100_001, 1_000_000)
    assert round(short, 6) == round(0.1 * 0.1 + 0.5, 6)  # 100K tokens at $0.10 + 1M out at $0.50
    assert round(long, 4) == round(100_001 * 0.5 / 1e6 + 2.5, 4)  # whole call at $0.50 in / $2.50 out


def test_record_llm_call_accumulates_inside_a_request():
    with observability.request_stats() as stats:
        for _ in range(2):
            observability.record_llm_call(text_result("x"), time.perf_counter())
        usage = observability.current_usage()
    assert stats.llm_calls == 2
    assert usage["input_tokens"] == 2000 and usage["output_tokens"] == 400
    assert usage["cost_usd"] == round(2 * (1000 * 2 + 200 * 10) / 1_000_000, 6)


def test_missing_usage_does_not_break_and_outside_request_is_ignored():
    observability.record_llm_call(LLMResult(text=""), time.perf_counter())
    assert observability.current_usage() is None
    with observability.request_stats():
        observability.record_llm_call(LLMResult(text=""), time.perf_counter())
        assert observability.current_usage()["input_tokens"] == 0


def test_llm_call_log_line_is_valid_json(caplog):
    with caplog.at_level(logging.INFO, logger="metrics"):
        with observability.request_stats():
            observability.record_llm_call(text_result("x"), time.perf_counter())
    record = json.loads(caplog.records[-1].getMessage())
    assert record["event"] == "llm_call"
    assert {"request_id", "latency_ms", "input_tokens", "output_tokens", "cost_usd"} <= record.keys()


def test_agent_usage_adds_up_over_steps(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda q, n_results=3: [CHUNK])
    monkeypatch.setattr(
        "rag_assistant.provider",
        FakeProvider(
            [
                tool_result("search_documents", {"query": "meal"}, input_tokens=500, output_tokens=50),
                text_result("$75", input_tokens=900, output_tokens=30),
            ]
        ),
    )
    with observability.request_stats():
        agent.run_agent("What is the meal allowance?")
        usage = observability.current_usage()
    assert usage["llm_calls"] == 2
    assert usage["input_tokens"] == 1400 and usage["output_tokens"] == 80


def test_ask_endpoint_returns_usage_and_headers(monkeypatch):
    monkeypatch.setattr("rag_assistant.retrieve", lambda q, n_results=3: [CHUNK])
    monkeypatch.setattr("rag_assistant.provider", FakeProvider([text_result("$75 [policy.pdf, p. 3]")]))
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the meal allowance?"})
    body = response.json()
    assert body["usage"]["llm_calls"] == 1
    assert body["usage"]["input_tokens"] == 1000
    assert body["usage"]["cost_usd"] > 0
    assert body["usage"]["provider"] == "fake" and body["usage"]["model"] == "claude-sonnet-5-5"
    assert len(response.headers["x-request-id"]) == 12
    assert int(response.headers["x-latency-ms"]) >= 0


def test_out_of_scope_question_costs_nothing(monkeypatch):
    far = dict(CHUNK, distance=1.9)
    monkeypatch.setattr("rag_assistant.retrieve", lambda q, n_results=3: [far])
    with TestClient(app) as client:
        body = client.post("/ask", json={"question": "Who won the football game?"}).json()
    assert body["llm_called"] is False
    assert body["usage"]["llm_calls"] == 0 and body["usage"]["cost_usd"] == 0


def test_request_log_line_has_totals(monkeypatch, caplog):
    monkeypatch.setattr("rag_assistant.retrieve", lambda q, n_results=3: [CHUNK])
    monkeypatch.setattr("rag_assistant.provider", FakeProvider([text_result("$75")]))
    with caplog.at_level(logging.INFO, logger="metrics"):
        with TestClient(app) as client:
            client.post("/ask", json={"question": "What is the meal allowance?"})
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "metrics"]
    request_line = next(line for line in lines if line["event"] == "request")
    assert request_line["path"] == "/ask" and request_line["status"] == 200
    assert request_line["llm_calls"] == 1 and request_line["input_tokens"] == 1000
