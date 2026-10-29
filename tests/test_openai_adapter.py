import json
from types import SimpleNamespace

import pytest

from medaibot.llm.base import LLMError, Message, ToolCall, ToolDef
from medaibot.llm.openai_provider import OpenAIProvider, to_openai_messages

TOOL = ToolDef("search_label", "search", {"type": "object", "properties": {}})


def _response(tool_calls=None, content=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage, model="gpt-test")


def _tool_call(name, arguments, id_="call_1"):
    return SimpleNamespace(id=id_, type="function", function=SimpleNamespace(name=name, arguments=arguments))


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_messages_convert_to_chat_completions_format():
    msgs = [
        Message(role="user", content="dose?"),
        Message(role="assistant", tool_calls=[ToolCall("call_1", "search_label", {"query": "dose"})]),
        Message(role="tool", content='{"results": []}', tool_call_id="call_1"),
    ]
    out = to_openai_messages("sys", msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"] == {"name": "search_label", "arguments": '{"query": "dose"}'}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"results": []}'}


def test_tool_calls_are_parsed_and_tool_use_is_required():
    client = FakeClient(_response([_tool_call("search_label", json.dumps({"query": "dose"}))]))
    resp = OpenAIProvider("gpt-test", client=client).complete("sys", [Message("user", "q")], [TOOL])
    assert resp.tool_calls == [ToolCall("call_1", "search_label", {"query": "dose"})]
    assert resp.usage == {"input_tokens": 11, "output_tokens": 7}
    assert client.kwargs["tool_choice"] == "required"
    assert client.kwargs["parallel_tool_calls"] is False


def test_invalid_json_arguments_are_kept_for_the_validator():
    client = FakeClient(_response([_tool_call("search_label", "{not json")]))
    resp = OpenAIProvider("gpt-test", client=client).complete("sys", [Message("user", "q")], [TOOL])
    assert resp.tool_calls[0].arguments == {"_invalid_json": "{not json"}


def test_sdk_errors_become_llm_errors():
    client = FakeClient(RuntimeError("rate limited"))
    with pytest.raises(LLMError, match="rate limited"):
        OpenAIProvider("gpt-test", client=client).complete("sys", [Message("user", "q")], [TOOL])
