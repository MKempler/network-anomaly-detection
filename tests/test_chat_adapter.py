import json

from src.chat_adapter import ChatAsResponses, chat_tools, to_messages


def test_responses_conversation_becomes_chat_messages():
    items = [
        {"role": "user", "content": "Top deviating metrics: ..."},
        {"type": "reasoning", "id": "r1", "summary": []},                     # Spark's: encrypted, no text
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking cart."}]},
        {"type": "function_call", "call_id": "fc1", "name": "get_logs", "arguments": '{"service":"cart"}'},
        {"type": "function_call", "call_id": "fc2", "name": "get_trace_errors", "arguments": '{"service":"cart"}'},
        {"type": "function_call_output", "call_id": "fc1", "output": '{"call_id":"c1"}'},
        {"type": "function_call_output", "call_id": "fc2", "output": '{"call_id":"c2"}'},
        {"type": "function_call", "call_id": "fc3", "name": "submit_ranking", "arguments": '{"causes":[]}'},
    ]
    msgs = to_messages("You are an on-call engineer.", items)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert msgs[2]["reasoning_content"] == "Checking cart." and "content" not in msgs[2]
    assert [t["id"] for t in msgs[2]["tool_calls"]] == ["fc1", "fc2"]
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"service": "cart"}   # objects, not strings
    assert msgs[3] == {"role": "tool", "tool_call_id": "fc1", "name": "get_logs", "content": '{"call_id":"c1"}'}
    assert msgs[-1]["tool_calls"][0]["function"]["name"] == "submit_ranking"
    assert "r1" not in json.dumps(msgs)


def test_tools_convert_to_chat_shape():
    [t] = chat_tools([{"type": "function", "name": "get_logs", "description": "d", "parameters": {"type": "object"}}])
    assert t == {"type": "function", "function": {"name": "get_logs", "description": "d", "parameters": {"type": "object"}}}


class _FakeChat:
    """Records the request and answers like vLLM serving Glimmer."""
    def __init__(self, reply):
        self.reply, self.sent, self.chat = reply, None, self
        self.completions = self

    def create(self, **kw):
        self.sent = kw
        reply = self.reply

        class R:
            def model_dump(self, mode="json"):
                return reply
        return R()


def test_adapter_round_trip_keeps_glimmer_reasoning_in_the_conversation():
    fake = _FakeChat({"id": "c1", "model": "muse-glimmer", "usage": {"prompt_tokens": 50, "completion_tokens": 9},
                      "choices": [{"message": {"role": "assistant", "content": None, "reasoning_content": "cart first",
                                               "tool_calls": [{"id": "t1", "type": "function", "function":
                                                               {"name": "get_logs", "arguments": '{"service":"cart"}'}}]}}]})
    tools = [{"type": "function", "name": "get_logs", "description": "d", "parameters": {"type": "object"}}]
    out = ChatAsResponses(fake).responses.create(model="muse-glimmer", instructions="sys", tools=tools, store=False,
                                                 input=[{"role": "user", "content": "go"}]).model_dump()
    assert fake.sent["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    assert fake.sent["tools"][0]["function"]["name"] == "get_logs" and "store" not in fake.sent
    assert [i["type"] for i in out["output"]] == ["reasoning", "function_call"]
    assert out["usage"]["input_tokens"] == 50
    # the next turn resends those items; the reasoning comes back as reasoning_content
    nxt = to_messages("sys", [{"role": "user", "content": "go"}, *out["output"],
                              {"type": "function_call_output", "call_id": "t1", "output": "{}"}])
    assert nxt[2]["reasoning_content"] == "cart first" and nxt[3]["name"] == "get_logs"
    # over the API, arguments go back as JSON strings (vLLM rejects objects there)
    out2 = ChatAsResponses(fake).responses.create(model="muse-glimmer", instructions="sys", tools=tools,
                                                  input=[{"role": "user", "content": "go"}, *out["output"],
                                                         {"type": "function_call_output", "call_id": "t1", "output": "{}"}])
    assert out2["output"]
    assert fake.sent["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"service": "cart"}'



def test_glimmer_clients_send_modal_proxy_auth(monkeypatch):
    from src.llm import client
    for name, value in {"GLIMMER_API_KEY": "k", "GLIMMER_BASE_URL": "https://example.invalid/v1",
                        "MODAL_PROXY_KEY": "wk-test", "MODAL_PROXY_SECRET": "ws-test"}.items():
        monkeypatch.setenv(name, value)
    c = client("glimmer-tuned")
    assert isinstance(c, ChatAsResponses)
    assert c.chat.default_headers["Modal-Key"] == "wk-test" and c.chat.default_headers["Modal-Secret"] == "ws-test"
