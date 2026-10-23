"""Shared test doubles: a scripted provider, so no test touches the network."""

from llm_provider import LLMResult, Provider, ToolCall


class FakeProvider(Provider):
    """Plays back prepared results (the last one repeats) and records every request."""

    name = "fake"
    model = "claude-sonnet-5-5"

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def generate(self, system, messages, tools=None, max_tokens=500):
        self.calls.append(
            {"system": system, "messages": list(messages), "tools": tools, "max_tokens": max_tokens}
        )
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def assistant_message(self, result):
        return {"role": "assistant", "content": result.text, "tools": [c.name for c in result.tool_calls]}

    def tool_result_messages(self, outputs):
        blocks = [{"type": "tool_result", "tool_use_id": c.id, "content": o} for c, o in outputs]
        return [{"role": "user", "content": blocks}]


def text_result(text, input_tokens=1000, output_tokens=200):
    return LLMResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens,
                     model="claude-sonnet-5-5", provider="fake")


def tool_result(name, args, call_id="t1", input_tokens=1000, output_tokens=200):
    return LLMResult(text="", tool_calls=[ToolCall(call_id, name, args)], input_tokens=input_tokens,
                     output_tokens=output_tokens, model="claude-sonnet-5-5", provider="fake")


def make_pdf(pages: list[str]) -> bytes:
    """Build a tiny valid PDF in memory (one text line per page), no extra libraries."""

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = {1: "<< /Type /Catalog /Pages 2 0 R >>"}
    kids = []
    for i, text in enumerate(pages):
        page_id, content_id = 4 + 2 * i, 5 + 2 * i
        kids.append(f"{page_id} 0 R")
        stream = f"BT /F1 12 Tf 72 720 Td ({esc(text)}) Tj ET"
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects[content_id] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    objects[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = b"%PDF-1.4\n"
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n{objects[obj_id]}\nendobj\n".encode("latin-1")
    xref_pos = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for obj_id in range(1, size):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    return out


class FakeCollection:
    """Stands in for ChromaDB so upload tests stay offline and leave no data behind."""

    def __init__(self):
        self.upserts = []
        self.deleted = []

    def upsert(self, ids, documents, metadatas):
        self.upserts.append(
            {"ids": ids, "documents": documents, "metadatas": metadatas}
        )

    def delete(self, where):
        self.deleted.append(where)
