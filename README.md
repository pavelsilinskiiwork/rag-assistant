# RAG Assistant

🔗 **Live demo:** https://pavelsilinskiiwork.github.io/rag-assistant/ (first answer may take up to a minute while the free server wakes up)

A question-answering assistant over your own documents (`.txt` and `.pdf`), available as a web chat, a command-line tool and a REST API (FastAPI, Docker).
It retrieves the most relevant passages with semantic search and asks an LLM
to answer **only** from them, citing the source file and page. If the documents do not
contain the answer, it says so instead of making something up. Two providers are supported
(Anthropic Claude and Groq-hosted open models) behind one interface.

[![tests](https://github.com/pavelsilinskiiwork/rag-assistant/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/pavelsilinskiiwork/rag-assistant/actions/workflows/tests.yml)

## How it works

```
docs/*.txt, docs/*.pdf -> split into chunks -> embeddings -> ChromaDB (local, persistent)
                                                                  |
question -> embedding -> top-3 nearest chunks -> distance threshold + relative margin
                                                                  |
                                    LLM (answers only from retrieved context)
```

- **Chunking:** paragraph-aware, up to 250 characters per chunk; PDFs are read page by page so citations look like `[report.pdf, p. 3]`
- **Embeddings:** ChromaDB's built-in local model (all-MiniLM-L6-v2), no extra API key
- **Vector store:** ChromaDB, persisted to `chroma_db/`
- **Generation:** Anthropic Claude or Groq (default `openai/gpt-oss-120b`), chosen by `LLM_PROVIDER` or per request
- **Three layers of protection against hallucinations:**
  1. Distance threshold: if no chunk is close enough, the LLM is not called at all
  2. Relative margin: chunks much farther than the best match are dropped (see Evaluation)
  3. System prompt: the model must answer only from context or reply that it could not find the answer

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your API key(s) into .env
                            # Windows (cmd): copy .env.example .env
```

Put your `.txt` or `.pdf` documents into the `docs/` folder (samples `handbook.txt` and `company_policy.pdf` are included).

Configuration (environment variables, see `.env.example`):

| Variable                             | Meaning                                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| `LLM_PROVIDER`                       | `anthropic` or `groq` (default provider)                                           |
| `ANTHROPIC_API_KEY`, `GROQ_API_KEY`  | Provider keys; only the ones you use are needed                                    |
| `ANTHROPIC_MODEL`, `GROQ_MODEL`      | Override the default model                                                         |
| `RELEVANCE_MARGIN`                   | Relative cut-off around the best chunk (default `0.35`, `none` disables)           |
| `DEMO_MODE`                          | `1` for a public demo: uploads disabled, rate limits on                            |
| `RATE_LIMIT_PER_HOUR`, `DAILY_LIMIT` | Per-IP hourly and global daily request caps (demo mode)                            |
| `FALLBACK_PROVIDER`                  | Provider to retry with when the first one is rate limited                          |
| `TRUST_PROXY`                        | `1` behind a reverse proxy: the client IP is taken from `X-Forwarded-For`          |
| `CORS_ORIGINS`                       | Comma-separated origins allowed to call the API from a browser (for the static UI) |

## Usage

Web UI: run the API (below) and open http://127.0.0.1:8000/. It has Ask and Agent modes, a provider selector, clickable sample questions, citations, expandable sources, the agent's tool steps and a per-answer usage line (model, tokens, cost, latency).

Command line:

```bash
python rag_assistant.py
```

```
Question: How many vacation days do I get?
Answer: Full-time employees receive 20 paid vacation days per year. [handbook.txt]
```

## REST API

```bash
python -m uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

![Swagger UI](assets/swagger.png)

| Method | Path      | Description                                                                |
| ------ | --------- | -------------------------------------------------------------------------- |
| GET    | `/`       | Web chat UI                                                                |
| GET    | `/health` | Liveness check                                                             |
| GET    | `/config` | Demo mode and available providers (no secrets)                             |
| POST   | `/ask`    | Ask a question about the indexed documents                                 |
| POST   | `/upload` | Index a PDF (max 5 MB); answers then cite file and page (disabled in demo) |
| POST   | `/agent`  | Tool-calling agent (document search, calculator) with a visible step trace |

Request (`provider` is optional):

```json
{ "question": "How many vacation days do I get?", "provider": "groq" }
```

Response:

```json
{
  "answer": "Full-time employees receive 20 paid vacation days per year. [handbook.txt]",
  "sources": [{ "source": "handbook.txt", "distance": 0.572, "text": "..." }],
  "llm_called": true,
  "usage": {
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "llm_calls": 1,
    "input_tokens": 272,
    "output_tokens": 100,
    "cost_usd": 0.0
  }
}
```

Behavior:

- Input is validated (3-500 characters); invalid requests return `422`
- If no chunk is close enough, `llm_called` is `false` and the LLM is never called
- If the provider is rate limited the API retries with `FALLBACK_PROVIDER` when configured, otherwise returns `429`; other provider failures return `502` instead of an unhandled `500`
- In demo mode, requests beyond the per-IP or daily limit return `429` with a `Retry-After` header

## Evaluation

`eval_retrieval.py` checks retrieval quality without calling the LLM (free and deterministic).
Each test case says whether the answer should be found in the documents.

| Question                                     | Best distance | Answer in docs? | Result           |
| -------------------------------------------- | ------------- | --------------- | ---------------- |
| How many vacation days do I get?             | 0.572         | yes             | OK               |
| Can I carry over unused vacation days?       | 0.651         | yes             | OK               |
| Can I work from home on Fridays?             | 0.985         | yes             | OK               |
| Do I need approval to buy a $150 keyboard?   | 1.383         | yes             | OK               |
| What should I do if I lose my laptop?        | 1.196         | yes             | OK               |
| What is the CEO's salary?                    | 1.757         | no              | OK               |
| What is the company's parental leave policy? | 1.253         | no              | FAIL (see below) |
| Who won the football game yesterday?         | 1.835         | no              | OK               |

**Findings**

- Smaller chunks improved retrieval: the vacation question went from distance 0.812 (500-char chunks) to 0.572 (250-char chunks).
- A single distance threshold cannot separate every case: "parental leave" is semantically close to "vacation" and scores better than the valid "keyboard" question.
  The threshold (1.4) therefore only filters clearly unrelated questions; the system prompt handles the borderline ones.

### Relevance filter: absolute threshold vs. relative margin

An absolute threshold alone also lets loosely related chunks through: for
"How many vacation days do I get?" the travel paragraph from the PDF (distance 1.345) was
returned as a second source next to the correct one (0.572). The model ignored it, but it
cost tokens and made the cited sources look noisy.

A second layer keeps only chunks within 0.35 of the closest one (`RELEVANCE_MARGIN`).
Evaluated on 4 questions over the sample documents (`eval_margin.py`):

| Filter                      | Answer chunk kept | No unrelated chunk | Avg chunks kept |
| --------------------------- | ----------------- | ------------------ | --------------- |
| Absolute threshold only     | 4/4               | 2/4                | 1.8             |
| + relative margin (0.2-0.5) | 4/4               | 4/4                | 1.0             |

Small corpus and small eval set: the margin is a heuristic to re-check as documents grow.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

64 tests run offline with fake providers and cost nothing: health check, validation (422), the no-LLM path for out-of-scope questions, PDF upload and page citations, the agent loop and its injection defenses, provider adapters, cost and usage accounting, rate limiting, provider fallback, CORS and the relevance filter. CI runs them on every push.

## Run with Docker

```bash
cp .env.example .env   # add your API key(s)
docker compose up --build
```

The API is available at http://localhost:8000/docs. The vector store and the embedding model cache are kept in named volumes, documents in `docs/` are mounted read-only, and API keys are passed at runtime and never baked into the image.

## Agent and prompt-injection defenses

`POST /agent` runs a tool-calling agent (native tool use, on Anthropic and Groq) with two
read-only tools: `search_documents` and `calculate`. Every step is returned in
the response, so you can see which tool was called with which arguments.

Retrieved documents are untrusted input and can carry hidden instructions.
Defenses, in layers:

1. **Least privilege:** only two read-only tools; no files, network or shell.
2. **Tool allowlist:** a tool name the model invents is refused, never run.
3. **Untrusted data:** passages are wrapped in `<document>` tags, tag breakout is
   neutralised, and suspicious phrases are flagged and logged.
4. **System prompt:** documents are data; instructions inside them are ignored.
5. **Hard limits:** max 5 steps, capped tool output, safe AST calculator (no `eval`).

Verification: unit tests (mocked LLM) plus `eval_injection.py`, a live check
against the real model with 5 attack styles (direct override, fake system notice,
tag breakout, exfiltration, polite injection). The agent still answered the original
question and warned about the document.
The first run exposed a gap in the phrase filter (two attacks were not flagged);
it was fixed and covered by a test.

| Provider / model             | Attacks resisted |
| ---------------------------- | ---------------- |
| Anthropic / Claude Haiku 5.5 | 5/5              |
| Groq / openai/gpt-oss-120b   | 5/5              |

This is a small sample on two models, not a guarantee. Phrase filtering is only a
detection signal, and the real protection is least privilege.

## Observability

Every LLM call and every HTTP request writes one structured JSON log line with
provider, model, latency, input/output tokens and an estimated cost in USD. The same numbers are
returned to the client in the `usage` field of `/ask` and `/agent`, and each
response carries `X-Request-Id` and `X-Latency-Ms` headers.

Questions outside the documents are stopped by the distance threshold before
any LLM call, so they cost nothing (`llm_calls: 0`).

## Public demo and deployment

The live demo is split in two:

- **UI:** `static/index.html` on GitHub Pages, published by `.github/workflows/pages.yml`.
- **API:** this repository's Docker image on Render (free plan), started with `DEMO_MODE=1`.

Demo safeguards, because a public endpoint with an API key can burn money:

- uploads are disabled (`403`), documents are fictional samples
- per-IP hourly limit and a global daily cap, `429` with `Retry-After`
- the default provider is Groq's free plan; Anthropic Claude Haiku is the optional fallback with a monthly spending limit set in the provider console
- CORS only allows the Pages origin (`CORS_ORIGINS`)

The free Render instance sleeps after 15 minutes without traffic, so the first request after a pause can take about a minute (the UI says so). The embedding model is baked into the Docker image to keep cold starts short.

## Limitations and next steps

- No authentication; the public demo relies on rate limits, disabled uploads and spending caps
- Rate limits are in memory, so they reset when the server restarts and are per instance
- Tiny sample corpus, so distance values are noisy and the evaluation sets are small
- Documents are re-indexed on every startup; there is no incremental update or deletion
- DOCX ingestion is not implemented
- Planned: AWS Bedrock provider, reranking, answer-level evaluation (Ragas), pgvector backend
