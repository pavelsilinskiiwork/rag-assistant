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
