from fastapi.testclient import TestClient

import api
import rag_assistant
from api import app
from fakes import FakeCollection, FakeProvider, make_pdf, text_result
from llm_provider import ProviderError
from ratelimit import RateLimiter, env_flag

OK = {"answer": "ok", "sources": [], "llm_called": False}


# ---------- rate limiter ----------


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_limiter_per_ip_window():
    clock = Clock()
    limiter = RateLimiter(per_ip_per_hour=2, daily_total=100, clock=clock)
    assert limiter.check("1.1.1.1") is None
    assert limiter.check("1.1.1.1") is None
    reason, retry_after = limiter.check("1.1.1.1")
    assert reason == "ip" and 0 < retry_after <= 3601
    assert limiter.check("2.2.2.2") is None  # another visitor is unaffected
    clock.now += 3601
    assert limiter.check("1.1.1.1") is None  # the window has passed


def test_limiter_daily_cap_resets_next_day():
    clock = Clock()
    limiter = RateLimiter(per_ip_per_hour=50, daily_total=3, clock=clock)
    for i in range(3):
        assert limiter.check(f"10.0.0.{i}") is None
    assert limiter.check("10.0.0.9")[0] == "daily"
    clock.now += 86400
    assert limiter.check("10.0.0.9") is None


def test_env_flag_is_forgiving(monkeypatch):
    for value in ["1", "1 ", " true", "TRUE", "yes", "on"]:
        monkeypatch.setenv("FLAG", value)
        assert env_flag("FLAG") is True, value
    for value in ["0", "", "no", "false", "2"]:
        monkeypatch.setenv("FLAG", value)
        assert env_flag("FLAG") is False, value
    monkeypatch.delenv("FLAG")
    assert env_flag("FLAG") is False


def test_limiter_from_env_accepts_windows_trailing_spaces(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1 ")  # what `set DEMO_MODE=1 ` produces in cmd
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "3 ")
    limiter = RateLimiter.from_env()
    assert limiter.enabled and limiter.per_ip == 3


def test_disabled_limiter_never_blocks():
    limiter = RateLimiter(per_ip_per_hour=0, daily_total=0, enabled=False)
    assert all(limiter.check("x") is None for _ in range(5))


def test_api_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(api, "limiter", RateLimiter(per_ip_per_hour=2, daily_total=100))
    monkeypatch.setattr("api.ask", lambda question, **kw: dict(OK))
    with TestClient(app) as client:
        codes = [client.post("/ask", json={"question": "How many days?"}).status_code for _ in range(3)]
        blocked = client.post("/ask", json={"question": "How many days?"})
    assert codes == [200, 200, 429]
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0


def test_client_ip_uses_last_forwarded_entry_only_when_trusted(monkeypatch):
    seen = []
    monkeypatch.setattr(api, "limiter", RateLimiter(per_ip_per_hour=1, daily_total=100))
    monkeypatch.setattr("api.ask", lambda question, **kw: dict(OK))
    original = api.client_ip
    monkeypatch.setattr(api, "client_ip", lambda request: seen.append(original(request)) or seen[-1])
    monkeypatch.setenv("TRUST_PROXY", "1")
    with TestClient(app) as client:
        client.post("/ask", json={"question": "How many days?"}, headers={"X-Forwarded-For": "6.6.6.6, 9.9.9.9"})
        monkeypatch.delenv("TRUST_PROXY")
        client.post("/ask", json={"question": "How many days?"}, headers={"X-Forwarded-For": "6.6.6.6, 9.9.9.9"})
    assert seen == ["9.9.9.9", "testclient"]


# ---------- demo mode, config, UI ----------


def test_upload_disabled_in_demo_mode(monkeypatch):
    monkeypatch.setattr(api, "DEMO_MODE", True)
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("a.pdf", make_pdf(["x"]), "application/pdf")})
    assert response.status_code == 403


def test_config_lists_only_providers_with_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(api, "DEMO_MODE", True)
    with TestClient(app) as client:
        body = client.get("/config").json()
    assert body["providers"] == ["groq"] and body["demo_mode"] is True
    assert "key" not in str(body).lower()


def test_index_page_is_served():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200 and "RAG Assistant" in response.text
    assert "innerHTML" not in response.text  # untrusted text must only go through textContent


# ---------- provider choice and fallback ----------


def test_request_can_choose_provider(monkeypatch):
    chosen = FakeProvider([text_result("x")])
    chosen.name = "groq"
    monkeypatch.setattr("api.resolve_provider", lambda name: chosen if name == "groq" else None)
    seen = {}

    def fake_ask(question, provider=None):
        seen["provider"] = provider
        return dict(OK)

    monkeypatch.setattr("api.ask", fake_ask)
    with TestClient(app) as client:
        client.post("/ask", json={"question": "How many days?", "provider": "groq"})
        assert seen["provider"] is chosen
        assert client.post("/ask", json={"question": "How many days?", "provider": "grok"}).status_code == 422


def test_rate_limited_provider_falls_back(monkeypatch):
    primary, backup = FakeProvider([text_result("p")]), FakeProvider([text_result("b")])
    primary.name, backup.name = "groq", "anthropic"
    monkeypatch.setenv("FALLBACK_PROVIDER", "anthropic")
    monkeypatch.setattr("api.resolve_provider", lambda name: {"groq": primary, "anthropic": backup}.get(name))
    used = []

    def fake_ask(question, provider=None):
        used.append(provider)
        if provider is primary:
            raise ProviderError("limit", rate_limited=True)
        return dict(OK)

    monkeypatch.setattr("api.ask", fake_ask)
    with TestClient(app) as client:
        body = client.post("/ask", json={"question": "How many days?", "provider": "groq"}).json()
    assert used == [primary, backup] and body["fallback_used"] is True


def test_no_fallback_when_it_would_be_the_same_provider(monkeypatch):
    only = FakeProvider([text_result("p")])
    only.name = "groq"
    monkeypatch.setenv("FALLBACK_PROVIDER", "groq")
    monkeypatch.setattr("api.resolve_provider", lambda name: only if name else None)

    def fake_ask(question, provider=None):
        raise ProviderError("limit", rate_limited=True)

    monkeypatch.setattr("api.ask", fake_ask)
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "How many days?", "provider": "groq"})
    assert response.status_code == 429


# ---------- PDFs placed in docs/ are indexed at startup ----------


def test_index_documents_loads_pdfs_from_docs_folder(monkeypatch, tmp_path):
    (tmp_path / "policy.pdf").write_bytes(make_pdf(["Home office stipend is $500.", "Meals are $75 per day."]))
    (tmp_path / "notes.txt").write_text("Vacation is 20 days.", encoding="utf-8")
    fake = FakeCollection()
    monkeypatch.setattr(rag_assistant, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(rag_assistant, "collection", fake)

    rag_assistant.index_documents()

    pdf = [u for u in fake.upserts if u["metadatas"][0]["source"] == "policy.pdf"]
    assert len(pdf) == 1 and [m["page"] for m in pdf[0]["metadatas"]] == [1, 2]
    assert any(u["metadatas"][0]["source"] == "notes.txt" for u in fake.upserts)
