from fastapi.testclient import TestClient

import api
import rag_assistant
from api import app
from fakes import FakeCollection, FakeProvider, make_pdf, text_result


# ---------- helpers ----------


# ---------- /health and /ask ----------


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
    monkeypatch.setattr("rag_assistant.provider", FakeProvider([text_result("20 days [handbook.txt]")]))

    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "How many vacation days do I get?"}
        )
    body = response.json()
    assert body["llm_called"] is True
    assert body["answer"] == "20 days [handbook.txt]"
    assert body["sources"][0]["source"] == "handbook.txt"


# ---------- PDF ingestion ----------


def test_upload_pdf_indexes_pages_with_metadata(monkeypatch):
    fake = FakeCollection()
    monkeypatch.setattr("rag_assistant.collection", fake)
    pdf = make_pdf(
        [
            "Parental leave lasts twelve weeks and is fully paid.",
            "Travel expenses must be booked through the company portal.",
        ]
    )

    with TestClient(app) as client:
        response = client.post(
            "/upload", files={"file": ("policy.pdf", pdf, "application/pdf")}
        )

    assert response.status_code == 200
    assert response.json() == {"filename": "policy.pdf", "pages_indexed": 2, "chunks": 2}

    # the txt documents are indexed at startup too, so look only at the PDF
    pdf_upserts = [u for u in fake.upserts if u["metadatas"][0]["source"] == "policy.pdf"]
    assert len(pdf_upserts) == 1
    metadatas = pdf_upserts[0]["metadatas"]
    assert [m["page"] for m in metadatas] == [1, 2]
    assert "Parental leave" in pdf_upserts[0]["documents"][0]
    # the old chunks of the same file are removed before re-indexing
    assert {"source": "policy.pdf"} in fake.deleted


def test_upload_rejects_non_pdf_extension():
    with TestClient(app) as client:
        response = client.post(
            "/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
    assert response.status_code == 415


def test_upload_rejects_fake_pdf():
    with TestClient(app) as client:
        response = client.post(
            "/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")}
        )
    assert response.status_code == 415


def test_upload_rejects_too_large_file(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    pdf = make_pdf(["x" * 500])
    with TestClient(app) as client:
        response = client.post(
            "/upload", files={"file": ("big.pdf", pdf, "application/pdf")}
        )
    assert response.status_code == 413


def test_upload_pdf_without_text_returns_422(monkeypatch):
    monkeypatch.setattr("rag_assistant.collection", FakeCollection())
    with TestClient(app) as client:
        response = client.post(
            "/upload", files={"file": ("blank.pdf", make_pdf([""]), "application/pdf")}
        )
    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]


def test_context_header_contains_file_and_page(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.retrieve",
        lambda question, n_results=3: [
            {"text": "Twelve weeks.", "source": "report.pdf", "page": 3, "distance": 0.5}
        ],
    )
    fake = FakeProvider([text_result("ok")])
    monkeypatch.setattr("rag_assistant.provider", fake)

    result = rag_assistant.ask("How long is parental leave?")

    assert "[report.pdf, p. 3]" in fake.calls[0]["messages"][0]["content"]
    assert result["sources"][0]["page"] == 3


def test_chunk_text_splits_long_pdf_paragraph():
    text = "Sentence number one is here. " * 40  # one 1100+ character paragraph
    chunks = rag_assistant.chunk_text(text, max_chars=250, split_long=True)
    assert len(chunks) > 3
    assert all(len(c) <= 250 for c in chunks)
