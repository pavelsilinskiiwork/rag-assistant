import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import run_agent
import rag_assistant
from llm_provider import ProviderError, get_provider
from observability import current_usage, log_request, request_stats
from ratelimit import RateLimiter, env_flag
from rag_assistant import ask, index_documents, index_pdf

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
DEMO_MODE = env_flag("DEMO_MODE")  # public demo: uploads off, rate limits on
limiter = RateLimiter.from_env()
_providers: dict = {}


def resolve_provider(name: str | None):
    """Provider object for a name from the request or env; instances are cached."""
    if not name:
        return None
    if name not in _providers:
        _providers[name] = get_provider(name)
    return _providers[name]


def available_providers() -> list[str]:
    keys = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    return [name for name, env in keys.items() if os.getenv(env)]


def client_ip(request: Request) -> str:
    """Behind a reverse proxy (TRUST_PROXY=1) the last X-Forwarded-For entry is the
    address the proxy itself saw, which a client cannot forge."""
    forwarded = request.headers.get("x-forwarded-for")
    if env_flag("TRUST_PROXY") and forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def enforce_limits(request: Request) -> None:
    blocked = limiter.check(client_ip(request))
    if blocked:
        reason, retry_after = blocked
        detail = (
            "The demo has reached its daily limit. Please come back tomorrow."
            if reason == "daily"
            else "Too many requests from your address. Please try again later."
        )
        raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(retry_after)})


def run_with_fallback(fn, question: str, provider_name: str | None) -> dict:
    """Run ask/run_agent. If the provider is rate limited, retry once on FALLBACK_PROVIDER."""
    primary = resolve_provider(provider_name)
    try:
        result = fn(question, provider=primary)
        result["fallback_used"] = False
    except ProviderError as e:
        fallback = resolve_provider(os.getenv("FALLBACK_PROVIDER"))
        current = primary or rag_assistant.provider
        if not (e.rate_limited and fallback is not None and fallback.name != current.name):
            raise
        result = fn(question, provider=fallback)
        result["fallback_used"] = True
    result["usage"] = current_usage()
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEMO_MODE:
        print(f"DEMO_MODE is ON: {limiter.per_ip} requests/hour per visitor, {limiter.daily_total}/day total, uploads disabled")
    else:
        print("DEMO_MODE is OFF: no rate limits, uploads enabled")
    index_documents()  # (re)index docs/ on startup
    yield


app = FastAPI(
    title="RAG Assistant API",
    description="Ask questions about local documents. Answers are grounded in retrieved passages.",
    version="0.5.0",
    lifespan=lifespan,
)


def cors_origins() -> list[str]:
    """CORS_ORIGINS="https://user.github.io,https://other.example" (empty = same-origin only)."""
    return [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]


if cors_origins():  # lets the static GitHub Pages UI call this API from another origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["Retry-After", "X-Request-Id"],
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
    provider: Literal["anthropic", "groq"] | None = None  # default: LLM_PROVIDER


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
    fallback_used: bool = False


class AgentStep(BaseModel):
    tool: str
    input: dict
    output: str


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    stopped: str  # "final_answer" or "max_steps"
    usage: Usage | None = None
    fallback_used: bool = False


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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/config")
def config():
    """What the UI needs to know: demo mode and which providers have keys. No secrets."""
    return {
        "demo_mode": DEMO_MODE,
        "default_provider": rag_assistant.provider.name,
        "providers": available_providers(),
        "fallback_provider": os.getenv("FALLBACK_PROVIDER"),
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, request: Request):
    enforce_limits(request)
    try:
        return run_with_fallback(ask, body.question, body.provider)
    except ProviderError as e:
        raise provider_http_error(e)


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(body: AskRequest, request: Request):
    """Tool-calling agent: searches documents and calculates, with a visible trace of steps."""
    enforce_limits(request)
    try:
        return run_with_fallback(run_agent, body.question, body.provider)
    except ProviderError as e:
        raise provider_http_error(e)


@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile):
    """Index a PDF so that /ask can answer from it and cite file and page."""
    if DEMO_MODE:
        raise HTTPException(status_code=403, detail="Uploads are disabled in the public demo.")
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
