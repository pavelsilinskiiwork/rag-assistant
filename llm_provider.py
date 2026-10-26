"""One interface, several LLM providers (Anthropic, Groq, AWS Bedrock).

The rest of the app talks only to `Provider.generate()` and never sees a vendor
SDK. Each provider owns its message format: Anthropic keeps tool results inside a
user message, OpenAI-style APIs (Groq) use separate `tool` messages. The agent
keeps the history as opaque messages and asks the provider to build them.

Choose a provider with the LLM_PROVIDER environment variable ("anthropic", "groq" or "bedrock").
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

import httpx
from anthropic import Anthropic, APIError

log = logging.getLogger("provider")


class ProviderError(Exception):
    """The LLM provider failed. `rate_limited` is True for HTTP 429 (quota or rate limit)."""

    def __init__(self, message: str, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    stop_reason: str = ""  # why generation ended, when the provider says (e.g. content_filtered)
    raw: object = None  # provider-specific payload used to rebuild the assistant message


class Provider:
    name = "base"
    model = ""

    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 500) -> LLMResult:
        raise NotImplementedError

    def assistant_message(self, result: LLMResult) -> dict:
        raise NotImplementedError

    def tool_result_messages(self, outputs: list[tuple[ToolCall, str]]) -> list[dict]:
        raise NotImplementedError


# ---------- Anthropic ----------


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self._needs_key = client is None
        self.client = client or Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, system, messages, tools=None, max_tokens=500) -> LLMResult:
        if self._needs_key and not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("anthropic: ANTHROPIC_API_KEY is not set")
        kwargs = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages}
        if tools:
            kwargs["tools"] = tools  # our neutral tool format is Anthropic's own
        try:
            response = self.client.messages.create(**kwargs)
        except APIError as e:
            status = getattr(e, "status_code", None)
            raise ProviderError(f"anthropic: {e.__class__.__name__}", rate_limited=status == 429) from e
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            tool_calls=calls,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            model=self.model,
            provider=self.name,
            raw=response.content,
        )

    def assistant_message(self, result: LLMResult) -> dict:
        blocks = []
        for b in result.raw:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return {"role": "assistant", "content": blocks}

    def tool_result_messages(self, outputs) -> list[dict]:
        blocks = [{"type": "tool_result", "tool_use_id": call.id, "content": out} for call, out in outputs]
        return [{"role": "user", "content": blocks}]


# ---------- Groq (OpenAI-compatible API) ----------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def clean_text(text: str) -> str:
    """gpt-oss writes citations as 【file】 and uses narrow no-break spaces; normalise
    them so the UI and citation parsing see plain [file] and normal spaces."""
    return (
        text.replace("\u202f", " ").replace("\u00a0", " ").replace("【", "[").replace("】", "]")
    )


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, model: str | None = None, client: httpx.Client | None = None):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.client = client or httpx.Client(timeout=30)

    def generate(self, system, messages, tools=None, max_tokens=500) -> LLMResult:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderError("groq: GROQ_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"
        try:
            response = self.client.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {api_key}"})
        except httpx.HTTPError as e:
            raise ProviderError(f"groq: {e.__class__.__name__}") from e
        if response.status_code == 429:
            raise ProviderError("groq: rate limit reached", rate_limited=True)
        if response.status_code >= 400:
            raise ProviderError(f"groq: HTTP {response.status_code}")

        data = response.json()
        message = data["choices"][0]["message"]
        calls = []
        for tc in message.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}  # the tool layer will reject the empty input
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args if isinstance(args, dict) else {}))
        usage = data.get("usage") or {}
        return LLMResult(
            text=clean_text(message.get("content") or ""),
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            model=self.model,
            provider=self.name,
            raw=message,
        )

    def assistant_message(self, result: LLMResult) -> dict:
        message = {"role": "assistant", "content": result.raw.get("content") or ""}
        if result.raw.get("tool_calls"):
            message["tool_calls"] = result.raw["tool_calls"]
        return message

    def tool_result_messages(self, outputs) -> list[dict]:
        return [{"role": "tool", "tool_call_id": call.id, "content": out} for call, out in outputs]


# ---------- AWS Bedrock (Converse API) ----------
#
# The Converse API gives every Bedrock model the same request/response shape, including tool
# use, so one adapter covers Nova, Claude, Llama, ... Switch models with BEDROCK_MODEL_ID.
# Auth: standard AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, a profile, an IAM
# role) or a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK. Region: AWS_REGION (default us-east-1).

DEFAULT_BEDROCK_MODEL = "us.amazon.nova-lite-v1:0"


class BedrockProvider(Provider):
    name = "bedrock"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)
        self._client = client

    @property
    def client(self):
        if self._client is None:  # created lazily: importing the app must not need AWS set up
            import boto3

            self._client = boto3.client(
                "bedrock-runtime", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
            )
        return self._client

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        """Plain-string content (what the RAG path sends) becomes Converse text blocks."""
        return [
            {"role": m["role"], "content": [{"text": m["content"]}] if isinstance(m["content"], str) else m["content"]}
            for m in messages
        ]

    def generate(self, system, messages, tools=None, max_tokens=500) -> LLMResult:
        kwargs = {
            "modelId": self.model,
            "system": [{"text": system}],
            "messages": self._convert(messages),
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if tools:
            kwargs["toolConfig"] = {
                "tools": [
                    {"toolSpec": {"name": t["name"], "description": t["description"],
                                  "inputSchema": {"json": t["input_schema"]}}}
                    for t in tools
                ]
            }
        try:
            response = self.client.converse(**kwargs)
        except Exception as e:  # botocore errors; imported lazily so tests can use fake clients
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
            throttled = code in ("ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException")
            raise ProviderError(f"bedrock: {code or e.__class__.__name__}", rate_limited=throttled) from e

        blocks = response["output"]["message"]["content"]
        text = "".join(b["text"] for b in blocks if "text" in b)
        # Nova writes its reasoning as <thinking>...</thinking>; users should only see the answer
        text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).strip()
        calls = [
            ToolCall(b["toolUse"]["toolUseId"], b["toolUse"]["name"], b["toolUse"].get("input") or {})
            for b in blocks
            if "toolUse" in b
        ]
        usage = response.get("usage") or {}
        stop_reason = response.get("stopReason", "")
        if stop_reason in ("content_filtered", "guardrail_intervened"):
            # the platform's own filter replaced the answer; make that visible in the logs
            log.info(json.dumps({"event": "provider_blocked", "provider": self.name, "model": self.model,
                                 "stop_reason": stop_reason}))
        return LLMResult(
            text=clean_text(text),
            tool_calls=calls,
            input_tokens=int(usage.get("inputTokens", 0) or 0),
            output_tokens=int(usage.get("outputTokens", 0) or 0),
            model=self.model,
            provider=self.name,
            stop_reason=stop_reason,
            raw=blocks,
        )

    def assistant_message(self, result: LLMResult) -> dict:
        return {"role": "assistant", "content": result.raw}

    def tool_result_messages(self, outputs) -> list[dict]:
        blocks = [
            {"toolResult": {"toolUseId": call.id, "content": [{"text": out}]}} for call, out in outputs
        ]
        return [{"role": "user", "content": blocks}]


def get_provider(name: str | None = None) -> Provider:
    name = (name or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "groq":
        return GroqProvider()
    if name == "bedrock":
        return BedrockProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Use 'anthropic', 'groq' or 'bedrock'.")
