"""Regenerates pyllym_golden.json from pyllym itself (the Go parity tests
compare against it):

    cd ../malayopython_reticulatus_llm && .venv/bin/python -I \
        ../gollym/protocols/chatcompletions/testdata/golden.py > \
        ../gollym/protocols/chatcompletions/testdata/pyllym_golden.json

Attachment files are written to a temp dir with fixed names and bytes; the Go
test recreates them identically.
"""

import json
import sys
import tempfile
from pathlib import Path

import pyllym
from pyllym import Content, Message, RawContent, Tool
from pyllym.chunk import Chunk  # noqa: F401
from pyllym.connection import Response
from pyllym.protocols.chat_completions import ChatCompletions
from pyllym.stream_accumulator import StreamAccumulator
from pyllym.thinking import Thinking
from pyllym.tool_call import ToolCall

FILES = {
    "notes.txt": b"hello file\n",
    "doc.pdf": b"%PDF-1.4 test",
    "pic.png": b"\x89PNG\r\n\x1a\nfakepng",
}

cfg = pyllym.config()
for k in ("openai", "deepseek"):
    setattr(cfg, f"{k}_api_key", "sk-test")


class Weather(Tool):
    description = "Get weather"

    def execute(self, *, city, days=1):
        return city


Weather.param("city", desc="City name")
Weather.param("days", type="integer", required=False)


class Ping(Tool):
    def execute(self):
        return "pong"


class Tagged(Tool):
    description = "Tagged"
    provider_params = {"function": {"strict": False}, "x_extra": 1}

    def execute(self, *, q):
        return q


def chat(model, provider=None):
    return pyllym.create_chat(model=model, provider=provider, assume_model_exists=bool(provider))


out = {"render": {}, "parse": {}, "stream": {}}

# --- render -------------------------------------------------------------------
c = chat("gpt-4o").with_instructions("be terse").with_temperature(0.7)
c.with_tools(Weather, Ping, Tagged, choice="weather", calls="one")
c.with_params(max_tokens=100, metadata={"a": {"b": 1}})
c.add_user_message("hi")
out["render"]["tools_params"] = c.render()

c = chat("o3").with_temperature(0.2).with_thinking(effort="high")
c.with_schema({"name": "My Schema!", "strict": False, "description": "d",
               "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}})
c.add_user_message("hi")
out["render"]["schema_thinking"] = c.render()

c = chat("gpt-4o").with_schema({"type": "object", "properties": {"x": {"type": "integer"}}, "strict": False})
c.add_user_message("hi")
out["render"]["bare_schema"] = c.render()

c = chat("gpt-4o-search-preview", "openai").with_temperature(0.5)
c.add_user_message("hi")
out["render"]["search_temperature"] = c.render()

with tempfile.TemporaryDirectory() as d:
    paths = {}
    for name, data in FILES.items():
        p = Path(d) / name
        p.write_bytes(data)
        paths[name] = str(p)
    c = chat("deepseek-chat", "deepseek").with_instructions("sys")
    c.add_message(role="user", content="weather?")
    c.add_message(role="assistant", content=None,
                  tool_calls={"call_1": ToolCall(id="call_1", name="weather", arguments={"city": "Paris", "note": "é<>&"})},
                  thinking=Thinking.build(text="let me check", signature="sig1"))
    c.add_message(role="tool", content="Sunny", tool_call_id="call_1")
    c.add_message(role="assistant", content=None, thinking=Thinking.build(text="done thinking"))
    c.add_user_message("look", with_=[paths["notes.txt"], paths["doc.pdf"], "https://example.com/cat.png", paths["pic.png"]])
    c.add_message(role="user", content=RawContent({"k": [1, 2]}))
    c.add_message(role="user", content=RawContent([{"type": "text", "text": "raw"}]))
    out["render"]["conversation"] = c.render()

provider = chat("gpt-4o").provider
proto = ChatCompletions(provider, pyllym.models.find("gpt-4o"))
out["render"]["stream"] = proto.render([Message(role="user", content="hi")], tools={}, temperature=None, stream=True)


# --- parse --------------------------------------------------------------------
def msg_dict(m):
    d = m.to_dict()
    if m.tool_calls:
        d["tool_calls"] = {k: v.to_dict() for k, v in m.tool_calls.items()}
    d.pop("citations", None)
    d["citations"] = [x.to_dict() for x in m.citations]
    return d


responses = {
    "think_usage_annotations": {
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "<think>hmm</think>  Answer here",
                                 "annotations": [{"type": "url_citation", "url_citation": {
                                     "url": "https://a", "title": "A", "start_index": 0, "end_index": 6}}]},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 130,
                  "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
                  "completion_tokens_details": {"reasoning_tokens": 5}},
    },
    "blocks_deepseek_search": {
        "model": "deepseek-chat",
        "search_results": [{"url": "https://s", "title": "S", "snippet": "snip"}, "junk"],
        "choices": [{"message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": [{"type": "text", "text": "a"}, {"type": "x", "text": "no"}]},
            {"type": "thinking", "thinking": "b"},
            {"type": "text", "text": "c"}, {"type": "text", "text": "d"}],
            "reasoning_signature": "rs"}, "finish_reason": "length"}],
        "usage": {"prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 3, "prompt_tokens": 10,
                  "completion_tokens": 2, "total_tokens": 15},
    },
    "no_usage_reasoning": {
        "choices": [{"message": {"role": "assistant", "content": "x", "reasoning_content": "why"}}],
    },
    "tool_calls": {
        "model": "gemini-2.5-flash",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"},
             "extra_content": {"google": {"thought_signature": "ts"}}},
            {"id": "b", "type": "function", "function": {"name": "g", "arguments": ""}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    },
    "root_citations": {
        "model": "sonar",
        "citations": ["https://one", 2, "https://two"],
        "choices": [{"message": {"role": "assistant", "content": "c"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    },
}
for name, body in responses.items():
    m = proto.parse_completion_response(Response(status=200, body=body))
    out["parse"][name] = {"response": body, "message": msg_dict(m)}

# --- stream -------------------------------------------------------------------
streams = {
    "think_tags_and_usage": [
        {"model": "gpt-4o", "choices": [{"delta": {"role": "assistant", "content": "Hi <th"}}]},
        {"choices": [{"delta": {"content": "ink>inner</thi"}}]},
        {"choices": [{"delta": {"content": "nk> there"}}]},
        {"choices": [{"delta": {"reasoning_content": "R1", "reasoning_signature": "s1"}}]},
        {"choices": [{"delta": {"reasoning": "R2", "reasoning_signature": "s2"}}]},
        {"choices": [{"delta": {"annotations": [{"url_citation": {"url": "https://u", "start_index": 0, "end_index": 2}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13,
                                  "prompt_tokens_details": {"cached_tokens": 2}}},
    ],
    "tool_call_fragments": [
        {"model": "gpt-4o", "choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "weather", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "{\"city\":"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "\"Oslo\"}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, "completion_tokens": 7}},
    ],
}
for name, chunks in streams.items():
    acc = StreamAccumulator()
    for data in chunks:
        acc.add(proto.build_chunk(data))
    out["stream"][name] = {"chunks": chunks, "message": msg_dict(acc.to_message(None))}

# --- other endpoints ------------------------------------------------------------
models_body = {"object": "list", "data": [
    {"id": "gpt-x", "object": "model", "created": 1700000000, "owned_by": "openai"},
    {"id": "other", "object": "model", "owned_by": "sys"}]}
ds = chat("deepseek-chat", "deepseek").provider
out["features"] = {
    "models_body": models_body,
    "list_models_openai": [m.to_dict() for m in proto.parse_list_models_response(
        Response(status=200, body=models_body), "openai", provider.capabilities)],
    "list_models_deepseek": [m.to_dict() for m in ChatCompletions(ds, None).parse_list_models_response(
        Response(status=200, body=models_body), "deepseek", ds.capabilities)],
    "embedding_payload": proto.render_embedding_payload(["a", "b"], model="text-embedding-3-small", dimensions=256),
    "speech_payload": proto.render_speech_payload("hello", model="gpt-4o-mini-tts", voice=None, format="wav",
                                                  params={"extra": 1}, instructions="calm", speed=1.25),
}
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "clip.mp3"
    p.write_bytes(b"ID3audio")
    payload = proto.render_transcription_payload(str(p), model="whisper-1", language="fi", temperature=0.0,
                                                 prompt="names", response_format="verbose_json")
    out["features"]["transcription_data"] = {k: str(v) for k, v in payload["data"].items()}
    out["features"]["transcription_file"] = payload["files"]["file"][0]

json.dump(out, sys.stdout, indent=1, sort_keys=True, ensure_ascii=False)
