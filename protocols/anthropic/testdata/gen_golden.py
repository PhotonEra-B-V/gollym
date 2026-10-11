"""Regenerates pyllym_golden.json from pyllym (the parity reference).

    cd protocols/anthropic/testdata
    <pyllym>/.venv/bin/python -I gen_golden.py . > pyllym_golden.json
"""
import json, sys, types, os
from pathlib import Path
import pyllym
from pyllym.protocols.anthropic import Anthropic
from pyllym.message import Message
from pyllym.content import Content, RawContent
from pyllym.search_results import SearchResults
from pyllym.thinking import Thinking, ThinkingConfig
from pyllym.tool_call import ToolCall
from pyllym.tool import Tool
from pyllym.stream_accumulator import StreamAccumulator

files = Path(sys.argv[1])
files.mkdir(exist_ok=True)
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")
(files/"cat.png").write_bytes(PNG)
(files/"report.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
(files/"notes.txt").write_text("Oulu is cold. äö")

dummy = types.SimpleNamespace(config=None, connection=None)
def proto(mid):
    return Anthropic(dummy, pyllym.models.find(mid, "anthropic"))

class Weather(Tool):
    description = "Get weather"
    _params_schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

class Ping(Tool):
    description = None
    _params_schema = {}
    provider_params = {"cache_control": {"type": "ephemeral"}}

out = {}
schema = {"name": "answer", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "strict": True}, "strict": True}

def att(*names):
    return [str(files/n) for n in names]

msgs = [
    Message(role="system", content="You are terse."),
    Message(role="system", content=RawContent([{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}])),
    Message(role="user", content="hi"),
    Message(role="assistant", content="prev", thinking=Thinking(text="mull", signature="sig1")),
    Message(role="user", content=Content("look", att("cat.png", "report.pdf", "notes.txt"))),
    Message(role="user", content=Content("remote", ["https://example.com/cat.png", "https://example.com/doc.pdf"])),
    Message(role="assistant", content="x"),
]
msgs[-1].content = {"k": [1, "ä"]}
p = proto("claude-sonnet-4-6")
out["render_full"] = p.render_payload(msgs, tools={"weather": Weather(), "ping": Ping()}, temperature=0.2, model=p.model,
    stream=True, schema=schema, thinking=ThinkingConfig(effort="high"), citations=True,
    tool_prefs={"choice": "required", "calls": "one"})
out["render_nocite"] = p.render_payload([Message(role="user", content=Content("look", att("notes.txt", "report.pdf")))], model=p.model)

hist = [
    Message(role="user", content="Weather?"),
    Message(role="assistant", content="checking", tool_calls={"tu1": ToolCall(id="tu1", name="weather", arguments={"city": "Paris"})}, thinking=Thinking(text="t1", signature="s1")),
    Message(role="tool", content="Sunny", tool_call_id="tu1"),
    Message(role="assistant", content="", tool_calls={"tu2": ToolCall(id="tu2", name="search", arguments={})}, thinking=Thinking(signature="redacted")),
    Message(role="tool", content=SearchResults({"title": "Q4", "url": "https://x.test/q4", "text": "Revenue up"}, {"title": "Notes", "text": "No url"}), tool_call_id="tu2"),
    Message(role="assistant", content=None, tool_calls={"tu3": ToolCall(id="tu3", name="noop", arguments={})}),
    Message(role="tool", content="", tool_call_id="tu3"),
    Message(role="assistant", content=RawContent([{"type": "tool_use", "id": "tu4", "name": "raw", "input": {}}]), tool_calls={"tu4": ToolCall(id="tu4", name="raw", arguments={})}),
    Message(role="tool", content=RawContent([{"type": "tool_result", "tool_use_id": "tu4", "content": "raw"}]), tool_call_id="tu4"),
    Message(role="tool", content="x", tool_call_id="tu5"),
]
hist[-1].content = {"ok": True}
p37 = proto("claude-3-7-sonnet-20250219")
out["render_history"] = p37.render_payload(hist, model=p37.model, thinking=ThinkingConfig(budget=2048),
    tool_prefs={"choice": "weather"}, tools={"weather": Weather()})
out["render_history_nothinking"] = p37.render_payload(hist, model=p37.model)

choices = {}
for c, k in [(None, "one"), ("auto", None), ("none", "many"), ("required", "many"), ("weather", "one")]:
    choices[f"{c}|{k}"] = p._build_tool_choice({"choice": c, "calls": k})
out["tool_choices"] = choices

errs = {}
for mid, cfg in [("claude-3-haiku-20240307", ThinkingConfig(budget=1024)), ("claude-3-7-sonnet-20250219", ThinkingConfig(effort="high")), ("claude-sonnet-4-6", ThinkingConfig(budget=0))]:
    try:
        q = proto(mid)
        q.render_payload([Message(role="user", content="x")], model=q.model, thinking=cfg)
        errs[mid] = None
    except ValueError as e:
        errs[mid] = str(e)
out["thinking_errors"] = errs
q = proto("claude-sonnet-4-6")
out["render_effort_none"] = q.render_payload([Message(role="user", content="x")], model=q.model, thinking=ThinkingConfig(effort="none"))
q = proto("claude-3-haiku-20240307")
out["render_haiku"] = q.render_payload([Message(role="user", content="x")], model=q.model)

def summary(m):
    d = {
        "content": m.content,
        "thinking": None if not m.thinking else {"text": m.thinking.text, "signature": m.thinking.signature},
        "tool_calls": None if not m.tool_calls else {k: {"id": v.id, "name": v.name, "arguments": v.arguments} for k, v in m.tool_calls.items()},
        "tokens": m.tokens.to_dict() if m.tokens else None,
        "finish_reason": m.finish_reason,
        "model_id": m.model_id,
        "citations": [{k: v for k, v in c.__dict__.items() if v is not None} if hasattr(c, "__dict__") else {k: getattr(c, k) for k in c.__slots__ if getattr(c, k) is not None} for c in m.citations],
    }
    return d

resp_body = {
    "id": "msg_1", "model": "claude-sonnet-4-6", "stop_reason": "tool_use",
    "content": [
        {"type": "thinking", "thinking": "Let me ", "signature": "sigA"},
        {"type": "thinking", "thinking": "think."},
        {"type": "redacted_thinking", "data": "red"},
        {"type": "text", "text": "Hyvää päivää. ", "citations": [
            {"type": "char_location", "cited_text": "Oulu is cold.", "document_index": 0, "document_title": "notes.txt", "start_char_index": 0, "end_char_index": 13}]},
        {"type": "text", "text": "Revenue rose.", "citations": [
            {"type": "search_result_location", "cited_text": "Revenue up", "source": "https://x.test/q4", "title": "Q4", "search_result_index": 1, "start_block_index": 0, "end_block_index": 1},
            {"type": "page_location", "cited_text": "p", "document_index": 2, "document_title": "report.pdf", "start_page_number": 1, "end_page_number": 3},
            {"type": "web_search_result_location", "cited_text": "w", "url": "https://web.test", "title": "Web"}]},
        {"type": "tool_use", "id": "tu1", "name": "weather", "input": {"city": "Paris"}},
    ],
    "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 30,
              "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 5},
              "output_tokens_details": {"thinking_tokens": 20}},
}
resp = types.SimpleNamespace(body=resp_body)
out["parse_body"] = resp_body
out["parse_summary"] = summary(p.parse_completion_response(resp))
resp2 = {"model": "claude-sonnet-4-6", "stop_reason": "end_turn", "content": [], "usage": {"input_tokens": 3, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
out["parse_body2"] = resp2
out["parse_summary2"] = summary(p.parse_completion_response(types.SimpleNamespace(body=resp2)))

events = [
    {"type": "message_start", "message": {"id": "m", "model": "claude-sonnet-4-6", "usage": {"input_tokens": 25, "output_tokens": 1, "cache_read_input_tokens": 7, "cache_creation_input_tokens": 3}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Hmm "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "ok"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sigS"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "ping"},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "citations_delta", "citation": {"type": "char_location", "cited_text": "c", "document_index": 1, "document_title": "D"}}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "tu9", "name": "weather", "input": {}}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ""}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": "}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"Oulu\"}"}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 42}},
    {"type": "message_stop"},
]
out["stream_events"] = events
acc = StreamAccumulator()
for e in events:
    acc.add(p.build_chunk(e))
out["stream_summary"] = summary(acc.to_message(None))

class Caps:
    def critical_capabilities_for(self, mid): return ["streaming"]
lm = {"data": [{"type": "model", "id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6", "created_at": "2026-02-17T00:00:00Z"}, {"type": "model", "id": "claude-x", "created_at": "2026-01-01T00:00:00Z"}], "has_more": False}
out["list_models_body"] = lm
out["list_models"] = [{"id": i.id, "name": i.name, "provider": i.provider, "created_at": i.created_at.isoformat() if i.created_at else None, "capabilities": list(i.capabilities)} for i in p.parse_list_models_response(types.SimpleNamespace(body=lm), "anthropic", Caps())]

# pyllym puts the raw bytes of a text document into the payload (not JSON
# serialisable upstream); gollym sends the decoded text.
for m in out["render_full"]["messages"]:
    for b in m["content"] if isinstance(m["content"], list) else []:
        if b.get("type") == "document" and b["source"]["type"] == "text":
            b["source"]["data"] = b["source"]["data"].decode("utf-8")

print(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True, default=str))
