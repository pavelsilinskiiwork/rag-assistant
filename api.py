import os
from contextlib import asynccontextmanager

from anthropic import APIError
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_assistant import ask, index_documents, index_pdf

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_documents()  # (re)index docs/ on startup
    yield


app = FastAPI(
    title="RAG Assistant API",
    description="Ask questions about local documents. Answers are grounded in retrieved passages.",
    version="0.2.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=500, examples=["How many vacation days do I get?"]
    )


class Source(BaseModel):
    source: str
    page: int | None = None
    distance: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    llm_called: bool


class UploadResponse(BaseModel):
    filename: str
    pages_indexed: int
    chunks: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    try:
        return ask(request.question)
    except APIError as e:
        # The LLM provider failed: report a gateway error instead of a 500 crash
        raise HTTPException(
            status_code=502, detail=f"LLM provider error: {e.__class__.__name__}"
        )


@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile):
    """Index a PDF so that /ask can answer from it and cite file and page."""
    # Keep only the file name: the client-supplied path is never used on disk
    filename = os.path.basename((file.filename or "").replace("\\", "/"))
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are supported.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="The file is not a valid PDF.")

    try:
        return index_pdf(data, filename)
    except Exception as e:  # unreadable, encrypted, too long, or no text layer
        raise HTTPException(status_code=422, detail=str(e) or "Could not read the PDF.")