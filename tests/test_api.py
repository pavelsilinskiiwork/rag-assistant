from fastapi.testclient import TestClient

from api import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_question_too_short_returns_422():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_unrelated_question_does_not_call_llm():
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "Who won the football game yesterday?"}
        )
    body = response.json()
    assert response.status_code == 200
    assert body["llm_called"] is False
    assert body["sources"] == []


def test_relevant_question_returns_sources(monkeypatch):
    # Replace the real Claude client with a fake one: no network, no cost
    class Block:
        text = "20 days [handbook.txt]"

    class Response:
        content = [Block()]

    class FakeMessages:
        def create(self, **kwargs):
            return Response()

    class FakeLLM:
        messages = FakeMessages()

    monkeypatch.setattr("rag_assistant.llm", FakeLLM())

    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "How many vacation days do I get?"}
        )
    body = response.json()
    assert body["llm_called"] is True
    assert body["answer"] == "20 days [handbook.txt]"
    assert body["sources"][0]["source"] == "handbook.txt"