import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from agent import run_agent
from llm_provider import ProviderError
from observability import current_usage, log_request, request_stats
from rag_assistant import ask, index_documents, index_pdf

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_documents()  # (re)index docs/ on startup
    yield


app = FastAPI(
    title="RAG Assistant API",
    description="Ask questions about local documents. Answers are grounded in retrieved passages.",
    version="0.4.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """One structured log line per request, plus X-Request-Id and X-Latency-Ms headers."""
    with request_stats() as stats:
        response = await call_next(request)
        response.headers["X-Request-Id"] = stats.request_id
        response.headers["X-Latency-Ms"] = str(round(stats.elapsed_ms()))
        if request.url.path != "/health":
            log_request(stats, request.method, request.url.path, response.status_code)
    return response


class Usage(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float  # estimate
    llm_latency_ms: int
    provider: str | None = None
    model: str | None = None


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
    usage: Usage | None = None


class AgentStep(BaseModel):
    tool: str
    input: dict
    output: str


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    stopped: str  # "final_answer" or "max_steps"
    usage: Usage | None = None


class UploadResponse(BaseModel):
    filename: str
    pages_indexed: int
    chunks: int


def provider_http_error(e: ProviderError) -> HTTPException:
    """Rate limit -> 429 with a friendly message; any other provider failure -> 502."""
    if e.rate_limited:
        return HTTPException(
            status_code=429,
            detail="The demo has reached its request limit. Please try again later.",
        )
    return HTTPException(status_code=502, detail=f"LLM provider error: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    try:
        result = ask(request.question)
        result["usage"] = current_usage()
        return result
    except ProviderError as e:
        raise provider_http_error(e)


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(request: AskRequest):
    """Tool-calling agent: searches documents and calculates, with a visible trace of steps."""
    try:
        result = run_agent(request.question)
        result["usage"] = current_usage()
        return result
    except ProviderError as e:
        raise provider_http_error(e)


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
