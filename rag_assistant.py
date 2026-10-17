import io
import os
import re
from pathlib import Path

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

MODEL = "claude-sonnet-4-6"
THRESHOLD = 1.4  # max distance to treat a chunk as relevant (tune it)
DOCS_DIR = Path("docs")
NOT_FOUND = "I could not find this in the documents."
MAX_PDF_PAGES = 200

llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Persistent storage: the index survives between runs
chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.get_or_create_collection(name="docs")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph into pieces of at most max_chars.

    Splits on sentence boundaries first; a single sentence that is still too
    long is wrapped on word boundaries.
    """
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        words = sentence.split()
        # wrap a very long sentence word by word
        parts, line = [], ""
        for word in words:
            if line and len(line) + 1 + len(word) > max_chars:
                parts.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            parts.append(line)
        for part in parts:
            if current and len(current) + 1 + len(part) > max_chars:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = 250, split_long: bool = False) -> list[str]:
    """Split text into chunks by paragraphs, merging small ones up to max_chars.

    With split_long=True, a paragraph longer than max_chars is split too
    (needed for PDFs, where a whole page can be one paragraph).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if split_long:
        expanded = []
        for p in paragraphs:
            expanded.extend(_split_long(p, max_chars) if len(p) > max_chars else [p])
        paragraphs = expanded
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}".strip()
    if current:
        chunks.append(current)
    return chunks


def index_documents() -> None:
    """Read every .txt file in docs/, split into chunks, store in ChromaDB."""
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        collection.upsert(
            ids=[f"{path.name}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": path.name} for _ in chunks],
        )
        print(f"Indexed {path.name}: {len(chunks)} chunks")


def _normalize_pdf_text(text: str) -> str:
    """PDF extraction breaks lines at the page width; join them back into paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip())


def extract_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that contain extractable text."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("The PDF is encrypted.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"The PDF has more than {MAX_PDF_PAGES} pages.")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def index_pdf(data: bytes, filename: str) -> dict:
    """Index an uploaded PDF page by page, keeping source and page in metadata."""
    pages = extract_pdf_pages(data)
    ids, documents, metadatas = [], [], []
    for number, text in pages:
        chunks = chunk_text(_normalize_pdf_text(text), split_long=True)
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}-p{number}-{i}")
            documents.append(chunk)
            metadatas.append({"source": filename, "page": number})
    if not ids:
        raise ValueError(
            "No extractable text found (is it a scanned PDF? OCR is not supported)."
        )
    # Re-uploading the same file replaces its old chunks instead of mixing them
    collection.delete(where={"source": filename})
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return {"filename": filename, "pages_indexed": len(pages), "chunks": len(ids)}


def retrieve(question: str, n_results: int = 3) -> list[dict]:
    """Find the chunks closest in meaning to the question."""
    result = collection.query(query_texts=[question], n_results=n_results)
    return [
        {
            "text": doc,
            "source": meta["source"],
            "page": meta.get("page"),
            "distance": dist,
        }
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def citation(chunk: dict) -> str:
    """Label used both in the LLM context and in citations: 'file.pdf, p. 3'."""
    if chunk.get("page"):
        return f"{chunk['source']}, p. {chunk['page']}"
    return chunk["source"]


SYSTEM_PROMPT = (
    "You are a company assistant. Answer the question using ONLY the provided "
    "context. Cite sources in brackets exactly as they appear in the context "
    "headers, e.g. [handbook.txt] or [report.pdf, p. 3]. "
    f"If the context does not contain the answer, reply exactly: '{NOT_FOUND}' "
    "Do not use outside knowledge."
)


def ask(question: str) -> dict:
    """Return the answer and the chunks used. Reused by the CLI and the API."""
    chunks = [c for c in retrieve(question) if c["distance"] <= THRESHOLD]
    if not chunks:
        return {"answer": NOT_FOUND, "sources": [], "llm_called": False}

    context = "\n\n".join(f"[{citation(c)}]\n{c['text']}" for c in chunks)

    response = llm.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return {
        "answer": response.content[0].text,
        "sources": [
            {
                "source": c["source"],
                "page": c["page"],
                "distance": round(c["distance"], 3),
                "text": c["text"],
            }
            for c in chunks
        ],
        "llm_called": True,
    }


def answer(question: str) -> None:
    """CLI helper: call ask() and print the result."""
    result = ask(question)
    print("\nAnswer:", result["answer"])
    if not result["llm_called"]:
        print(f"(no chunk within threshold {THRESHOLD}, LLM was not called)")
        return
    print("\nRetrieved chunks:")
    for s in result["sources"]:
        label = citation(s)
        print(f"  distance={s['distance']:.3f} | {label} | {s['text'][:60]!r}...")


if __name__ == "__main__":
    index_documents()
    while True:
        q = input("\nQuestion (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if q:
            answer(q)