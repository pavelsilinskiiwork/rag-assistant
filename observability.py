"""Structured logging and per-request usage tracking (latency, tokens, cost).

Every LLM call writes one JSON log line; the API middleware writes one more line
per HTTP request with the totals. All numbers are also returned to the client in
the `usage` field, so a demo UI can show what each question cost.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request HTTP noise

log = logging.getLogger("metrics")

# USD per 1M tokens as (input, output), from the Anthropic price list (Oct 2026).
# An ESTIMATE: update when prices change; prompt-caching rates are not modelled.
# "long" is the higher tier some models charge for prompts above LONG_PROMPT_TOKENS.
LONG_PROMPT_TOKENS = 100_000
PRICES = {
    "claude-fable-5-1": {"short": (10.00, 50.00)},
    "claude-opus-5-5": {"short": (4.00, 20.00)},
    "claude-sonnet-5-5": {"short": (2.00, 10.00)},
    "claude-haiku-5-5": {"short": (0.10, 0.50), "long": (0.50, 2.50)},
    "claude-sonnet-4-6": {"short": (3.00, 15.00)},  # older model, not in the current list
}
DEFAULT_PRICE = (2.00, 10.00)  # unknown model: assume Sonnet 5.5 pricing


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    tiers = PRICES.get(model)
    if tiers is None:
        price_in, price_out = DEFAULT_PRICE
    elif input_tokens > LONG_PROMPT_TOKENS and "long" in tiers:
        price_in, price_out = tiers["long"]
    else:
        price_in, price_out = tiers["short"]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class RequestStats:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_latency_ms: float = 0.0

    def usage(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_latency_ms": round(self.llm_latency_ms),
        }

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000


_current: ContextVar[RequestStats | None] = ContextVar("request_stats", default=None)


@contextmanager
def request_stats():
    """Open a stats scope for one request; LLM calls made inside are added to it."""
    stats = RequestStats()
    token = _current.set(stats)
    try:
        yield stats
    finally:
        _current.reset(token)


def current_usage() -> dict | None:
    stats = _current.get()
    return stats.usage() if stats else None


def record_llm_call(response, model: str, started: float) -> None:
    """Log one LLM call and add it to the current request. Never raises."""
    latency_ms = (time.perf_counter() - started) * 1000
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cost = estimate_cost(model, input_tokens, output_tokens)

    stats = _current.get()
    if stats:
        stats.llm_calls += 1
        stats.input_tokens += input_tokens
        stats.output_tokens += output_tokens
        stats.cost_usd += cost
        stats.llm_latency_ms += latency_ms

    log.info(
        json.dumps(
            {
                "event": "llm_call",
                "request_id": stats.request_id if stats else None,
                "model": model,
                "latency_ms": round(latency_ms),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            }
        )
    )


def log_request(stats: RequestStats, method: str, path: str, status: int) -> None:
    log.info(
        json.dumps(
            {
                "event": "request",
                "request_id": stats.request_id,
                "method": method,
                "path": path,
                "status": status,
                "latency_ms": round(stats.elapsed_ms()),
                **stats.usage(),
            }
        )
    )
