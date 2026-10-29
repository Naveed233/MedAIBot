"""Provider-neutral chat and tool-calling types. Adapters convert to and from these."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class LLMError(Exception):
    """The model call failed (network, auth, rate limit, bad response)."""


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolDef],
        require_tool: bool = True,
    ) -> LLMResponse: ...
