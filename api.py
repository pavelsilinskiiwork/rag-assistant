from contextlib import asynccontextmanager

from anthropic import APIError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_assistant import ask, index_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_documents()  # (re)index docs/ on startup
    yield


app = FastAPI(
    title="RAG Assistant API",
    description="Ask questions about local documents. Answers are grounded in retrieved passages.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=500, examples=["How many vacation days do I get?"]
    )


class Source(BaseModel):
    source: str
    distance: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    llm_called: bool


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