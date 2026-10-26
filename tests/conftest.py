"""Make every test independent of the shell it runs in.

Setting DEMO_MODE / RATE_LIMIT_PER_HOUR in the terminal (to try the public demo by hand)
used to leak into the tests: uploads returned 403 and the shared limiter ran out of quota.
This fixture resets all demo-related settings before each test; tests that need demo
mode switch it on themselves.
"""

import pytest

import api
from ratelimit import RateLimiter

DEMO_ENV = (
    "DEMO_MODE", "RATE_LIMIT_PER_HOUR", "DAILY_LIMIT", "TRUST_PROXY", "FALLBACK_PROVIDER",
    # AWS credentials set in the terminal for a manual Bedrock run must not change /config
    "AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE", "BEDROCK_MODEL_ID",
)


@pytest.fixture(autouse=True)
def clean_demo_environment(monkeypatch):
    for name in DEMO_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api, "DEMO_MODE", False)
    monkeypatch.setattr(api, "limiter", RateLimiter(enabled=False))
