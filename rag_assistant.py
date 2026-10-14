import os
from pathlib import Path

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"
THRESHOLD = 1.4  # max distance to treat a chunk as relevant (tune it)
DOCS_DIR = Path("docs")
NOT_FOUND = "I could not find this in the documents."

llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Persistent storage: the index survives between runs
chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.get_or_create_collection(name="docs")


def chunk_text(text: str, max_chars: int = 250) -> list[str]:
    """Split text into chunks by paragraphs, merging small ones up to max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
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


def retrieve(question: str, n_results: int = 3) -> list[dict]:
    """Find the chunks closest in meaning to the question."""
    result = collection.query(query_texts=[question], n_results=n_results)
    return [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


SYSTEM_PROMPT = (
    "You are a company assistant. Answer the question using ONLY the provided "
    "context. Mention the source file name in brackets, e.g. [handbook.txt]. "
    f"If the context does not contain the answer, reply exactly: '{NOT_FOUND}' "
    "Do not use outside knowledge."
)


def ask(question: str) -> dict:
    """Return the answer and the chunks used. Reused by the CLI and the API."""
    chunks = [c for c in retrieve(question) if c["distance"] <= THRESHOLD]
    if not chunks:
        return {"answer": NOT_FOUND, "sources": [], "llm_called": False}

    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)

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
        print(f"  distance={s['distance']:.3f} | {s['source']} | {s['text'][:60]!r}...")


if __name__ == "__main__":
    index_documents()
    while True:
        q = input("\nQuestion (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if q:
            answer(q)