"""OpenAI Chat Completions adapter. Also works with OpenAI-compatible servers via base_url."""

import json
from typing import Any

from medaibot.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolDef


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(item)
        else:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def to_openai_tools(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on installed extras
                raise LLMError("install the openai extra: uv sync --extra openai") from exc
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=2)
        self._client = client
        self.model = model
        self.name = f"openai:{model}"

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef],
        require_tool: bool = True,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": to_openai_messages(system, messages)}
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "required" if require_tool else "auto"
            kwargs["parallel_tool_calls"] = False
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # the SDK raises several error types; all mean the call failed
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        if not response.choices:
            raise LLMError("model returned no choices")
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            if getattr(tc, "type", "function") != "function":
                continue
            raw = tc.function.arguments or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {"_invalid_json": raw}
            if not isinstance(arguments, dict):
                arguments = {"_invalid_json": raw}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        usage: dict[str, int] = {}
        if response.usage is not None:
            usage = {
                "input_tokens": int(response.usage.prompt_tokens or 0),
                "output_tokens": int(response.usage.completion_tokens or 0),
            }
        return LLMResponse(text=message.content or "", tool_calls=calls, model=response.model, usage=usage)
