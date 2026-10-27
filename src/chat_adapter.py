"""Run the Responses-API agent on a model served over Chat Completions.

Muse Glimmer is served by vLLM, which parses Glimmer's native tool-call format
into standard Chat Completions `tool_calls`. The agent loop and the record-and-
replay cache speak the Responses API. `ChatAsResponses` wraps a Chat
Completions client so `client.responses.create(...)` works on it, which leaves
the agent, verifier and cache unchanged.

`to_messages` and `chat_tools` are also how hosted-agent investigations become
fine-tuning data (scripts/finetune/build_sft.py), so the model is trained and
evaluated on exactly the same rendering of a conversation.

Glimmer's chat template needs tool-call arguments as objects. Over the API they
travel as JSON strings, as the OpenAI spec has them, and vLLM parses them before
templating; so `to_messages` gives objects for training and strings for serving.
Any reasoning text the model wrote goes in the assistant message's
`reasoning_content`; so do the short notes written alongside tool calls, which
the template would otherwise drop. Spark's own reasoning is encrypted by its
API, carries no text, and is skipped.
"""

import json
import uuid


def chat_tools(tools: list[dict]) -> list[dict]:
    """Responses API tool definitions -> Chat Completions shape."""
    return [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
            for t in tools]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [])


def to_messages(instructions: str, items: list[dict], json_arguments: bool = False) -> list[dict]:
    """A Responses API conversation (instructions + input/output items) -> chat messages.

    Tool-call arguments are objects, or JSON strings with `json_arguments` (for an API request).
    """
    messages = [{"role": "system", "content": instructions}]
    names: dict[str, str] = {}
    pending: dict | None = None

    def flush():
        nonlocal pending
        if pending is not None:
            notes, reasoning = pending.pop("content"), pending.pop("reasoning")
            if pending["tool_calls"]:
                reasoning = "\n\n".join(r for r in (reasoning, notes) if r)
            else:
                del pending["tool_calls"]
                pending["content"] = notes
            if reasoning:
                pending["reasoning_content"] = reasoning
            messages.append(pending)
            pending = None

    def assistant() -> dict:
        nonlocal pending
        if pending is None:
            pending = {"role": "assistant", "content": "", "reasoning": "", "tool_calls": []}
        return pending

    for item in items:
        kind = item.get("type")
        if kind == "message" and item.get("role") == "assistant":
            assistant()["content"] += "".join(p.get("text", "") for p in item.get("content", [])
                                              if p.get("type") == "output_text")
        elif kind == "reasoning":
            assistant()["reasoning"] += "".join(p.get("text", "") for p in item.get("content") or [])
        elif kind == "function_call":
            names[item["call_id"]] = item["name"]
            try:
                args = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            args = args if isinstance(args, dict) else {}
            assistant()["tool_calls"].append({"id": item["call_id"], "type": "function",
                                              "function": {"name": item["name"],
                                                           "arguments": json.dumps(args) if json_arguments else args}})
        elif kind == "function_call_output":
            flush()
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "name": names.get(item["call_id"], ""),
                             "content": item["output"]})
        elif item.get("role") == "user":
            flush()
            messages.append({"role": "user", "content": _text(item["content"])})
    flush()
    return messages


def to_response(completion: dict) -> dict:
    """A Chat Completions response -> the Responses API output items the agent reads."""
    message = completion["choices"][0]["message"]
    output = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        output.append({"type": "reasoning", "id": f"rs_{uuid.uuid4().hex[:12]}", "summary": [],
                       "content": [{"type": "reasoning_text", "text": reasoning}]})
    if message.get("content"):
        output.append({"type": "message", "role": "assistant", "id": f"msg_{uuid.uuid4().hex[:12]}",
                       "content": [{"type": "output_text", "text": message["content"]}]})
    for call in message.get("tool_calls") or []:
        output.append({"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                       "arguments": call["function"]["arguments"]})
    usage = completion.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {"id": completion.get("id"), "model": completion.get("model"), "output": output,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                      "output_tokens_details": {"reasoning_tokens": details.get("reasoning_tokens") or 0}}}


class _Dumped(dict):
    def model_dump(self, mode: str = "json") -> dict:
        return dict(self)


class ChatAsResponses:
    """`.responses.create(**responses_request)` on top of an OpenAI-compatible Chat Completions client."""

    def __init__(self, chat_client):
        self.chat, self.responses = chat_client, self

    def create(self, *, model: str, instructions: str, input: list[dict] | str, tools: list[dict] | None = None,
               text: dict | None = None, **_):
        # store/include are Responses API options with no Chat Completions equivalent
        items = [{"role": "user", "content": input}] if isinstance(input, str) else input
        fmt = (text or {}).get("format") or {}
        if fmt.get("type") == "json_schema":
            # Structured output. vLLM's response_format isn't enforced for Glimmer (its reasoning channel comes
            # first), but Glimmer calls tools reliably, so the schema is offered as the only tool and the call's
            # arguments come back as the response text, which is what a Responses API caller reads.
            name = fmt.get("name", "answer")
            tools = [{"type": "function", "name": name, "description": "Submit your answer.",
                      "parameters": fmt["schema"]}]
            instructions = f"{instructions}\n\nGive your answer by calling the {name} tool."
        completion = self.chat.chat.completions.create(model=model,
                                                       messages=to_messages(instructions, items, json_arguments=True),
                                                       tools=chat_tools(tools or []) or None)
        response = to_response(completion.model_dump(mode="json"))
        if fmt.get("type") == "json_schema":
            call = next((i for i in response["output"] if i["type"] == "function_call" and i["name"] == name), None)
            if call is not None:
                response["output"] = [i for i in response["output"] if i["type"] == "reasoning"] + [
                    {"type": "message", "role": "assistant", "id": call["call_id"],
                     "content": [{"type": "output_text", "text": call["arguments"]}]}]
        return _Dumped(response)
