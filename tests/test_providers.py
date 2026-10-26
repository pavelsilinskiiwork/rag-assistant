import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

import llm_provider
from api import app
from llm_provider import AnthropicProvider, GroqProvider, LLMResult, ProviderError, ToolCall, get_provider

TOOLS = [
    {
        "name": "calculate",
        "description": "Arithmetic",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
    }
]


def groq_with(handler):
    return GroqProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))


def groq_reply(message, usage=None):
    return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage or {}})


@pytest.fixture(autouse=True)
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")


# ---------- Groq ----------


def test_groq_text_answer_and_request_shape():
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return groq_reply({"content": "hello"}, {"prompt_tokens": 12, "completion_tokens": 3})

    result = groq_with(handler).generate("SYS", [{"role": "user", "content": "hi"}], max_tokens=77)

    assert result.text == "hello" and result.tool_calls == []
    assert (result.input_tokens, result.output_tokens) == (12, 3)
    assert result.provider == "groq" and result.model == "openai/gpt-oss-120b"
    assert seen["auth"] == "Bearer test-groq-key"
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["payload"]["max_completion_tokens"] == 77
    assert "tools" not in seen["payload"]


def test_groq_text_is_normalised():
    text = "Remote work is allowed\u202f3\u202fdays per week\u202f【poisoned.txt, p. 2】"
    result = groq_with(lambda r: groq_reply({"content": text})).generate("S", [])
    assert result.text == "Remote work is allowed 3 days per week [poisoned.txt, p. 2]"


def test_groq_converts_tools_and_parses_tool_calls():
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        call = {"id": "c1", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}}
        return groq_reply({"content": None, "tool_calls": [call]})

    result = groq_with(handler).generate("S", [{"role": "user", "content": "q"}], tools=TOOLS)

    assert seen["payload"]["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]
    assert seen["payload"]["tool_choice"] == "auto"
    assert result.tool_calls == [ToolCall("c1", "calculate", {"expression": "1+1"})]


def test_groq_bad_tool_arguments_become_empty_input():
    def handler(request):
        call = {"id": "c1", "type": "function", "function": {"name": "calculate", "arguments": "{not json"}}
        return groq_reply({"content": "", "tool_calls": [call]})

    result = groq_with(handler).generate("S", [], tools=TOOLS)
    assert result.tool_calls[0].input == {}


def test_groq_message_formats_for_tool_loop():
    provider = groq_with(lambda r: groq_reply({"content": ""}))
    call = {"id": "c1", "type": "function", "function": {"name": "calculate", "arguments": "{}"}}
    result = LLMResult(text="", raw={"content": None, "tool_calls": [call]})
    assert provider.assistant_message(result)["tool_calls"] == [call]
    outputs = [(ToolCall("c1", "calculate", {}), "2")]
    assert provider.tool_result_messages(outputs) == [{"role": "tool", "tool_call_id": "c1", "content": "2"}]


def test_groq_rate_limit_and_errors():
    with pytest.raises(ProviderError) as limited:
        groq_with(lambda r: httpx.Response(429)).generate("S", [])
    assert limited.value.rate_limited is True
    with pytest.raises(ProviderError) as failed:
        groq_with(lambda r: httpx.Response(500)).generate("S", [])
    assert failed.value.rate_limited is False


def test_groq_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        groq_with(lambda r: groq_reply({"content": "x"})).generate("S", [])


# ---------- Anthropic ----------


class FakeAnthropicClient:
    def __init__(self, response=None, error=None):
        self.messages = self
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def test_anthropic_maps_blocks_and_builds_messages():
    blocks = [
        SimpleNamespace(type="text", text="thinking"),
        SimpleNamespace(type="tool_use", id="t1", name="calculate", input={"expression": "2*3"}),
    ]
    client = FakeAnthropicClient(SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=40, output_tokens=7)))
    provider = AnthropicProvider(model="claude-haiku-5-5", client=client)

    result = provider.generate("SYS", [{"role": "user", "content": "q"}], tools=TOOLS, max_tokens=99)

    assert client.kwargs["system"] == "SYS" and client.kwargs["tools"] == TOOLS and client.kwargs["max_tokens"] == 99
    assert result.text == "thinking" and result.tool_calls == [ToolCall("t1", "calculate", {"expression": "2*3"})]
    assert (result.input_tokens, result.output_tokens, result.model) == (40, 7, "claude-haiku-5-5")
    assistant = provider.assistant_message(result)
    assert assistant["content"][1] == {"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expression": "2*3"}}
    user = provider.tool_result_messages([(result.tool_calls[0], "6")])
    assert user == [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "6"}]}]


def test_anthropic_rate_limit_is_translated():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    error = anthropic.RateLimitError("slow down", response=response, body=None)
    provider = AnthropicProvider(model="m", client=FakeAnthropicClient(error=error))
    with pytest.raises(ProviderError) as caught:
        provider.generate("S", [])
    assert caught.value.rate_limited is True


# ---------- factory and API ----------


def test_get_provider_selection(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(get_provider("groq"), GroqProvider)
    assert isinstance(get_provider("anthropic"), AnthropicProvider)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert get_provider().name == "groq"
    with pytest.raises(ValueError):
        get_provider("grok")


def test_api_maps_provider_errors(monkeypatch):
    def limited(question, **kw):
        raise ProviderError("quota", rate_limited=True)

    def broken(question, **kw):
        raise ProviderError("boom")

    with TestClient(app) as client:
        monkeypatch.setattr("api.ask", limited)
        response = client.post("/ask", json={"question": "How many days?"})
        assert response.status_code == 429 and "limit" in response.json()["detail"]
        monkeypatch.setattr("api.ask", broken)
        assert client.post("/ask", json={"question": "How many days?"}).status_code == 502


# ---------- AWS Bedrock (Converse API) ----------


class FakeBedrock:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def bedrock_reply(blocks, usage=None):
    return {"output": {"message": {"role": "assistant", "content": blocks}},
            "usage": usage or {"inputTokens": 0, "outputTokens": 0}}


def test_bedrock_text_answer_and_request_shape():
    fake = FakeBedrock(bedrock_reply([{"text": "hello\u202f[a.txt]"}], {"inputTokens": 20, "outputTokens": 5}))
    result = llm_provider.BedrockProvider(client=fake).generate(
        "SYS", [{"role": "user", "content": "hi"}], max_tokens=64)

    sent = fake.calls[0]
    assert sent["system"] == [{"text": "SYS"}]
    assert sent["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert sent["inferenceConfig"] == {"maxTokens": 64} and "toolConfig" not in sent
    assert result.text == "hello [a.txt]"
    assert (result.input_tokens, result.output_tokens) == (20, 5)
    assert result.provider == "bedrock" and result.model == "us.amazon.nova-lite-v1:0"


def test_bedrock_tools_and_tool_loop_messages():
    fake = FakeBedrock(bedrock_reply([
        {"text": "checking"},
        {"toolUse": {"toolUseId": "t1", "name": "calculate", "input": {"expression": "75*4"}}},
    ]))
    provider = llm_provider.BedrockProvider(client=fake)
    result = provider.generate("S", [{"role": "user", "content": "q"}], tools=TOOLS)

    spec = fake.calls[0]["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "calculate" and spec["inputSchema"] == {"json": TOOLS[0]["input_schema"]}
    assert result.tool_calls == [ToolCall("t1", "calculate", {"expression": "75*4"})]

    assert provider.assistant_message(result) == {"role": "assistant", "content": result.raw}
    assert provider.tool_result_messages([(result.tool_calls[0], "300")]) == [
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "300"}]}}]}
    ]


def test_bedrock_throttling_is_rate_limited_and_other_errors_are_not():
    class FakeClientError(Exception):  # same shape as botocore's ClientError, no AWS libraries needed
        def __init__(self, code):
            super().__init__(code)
            self.response = {"Error": {"Code": code, "Message": "x"}}

    with pytest.raises(ProviderError) as e:
        llm_provider.BedrockProvider(client=FakeBedrock(error=FakeClientError("ThrottlingException"))).generate("S", [])
    assert e.value.rate_limited
    with pytest.raises(ProviderError) as e:
        llm_provider.BedrockProvider(client=FakeBedrock(error=FakeClientError("AccessDeniedException"))).generate("S", [])
    assert not e.value.rate_limited and "AccessDeniedException" in str(e.value)


def test_bedrock_model_from_env_and_selection(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "my.model-v1:0")
    assert get_provider("bedrock").model == "my.model-v1:0"
    assert get_provider("bedrock").name == "bedrock"
