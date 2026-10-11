"""Regenerates pyllym_golden.json from pyllym itself (core parity tests):

    cd ../malayopython_reticulatus_llm && .venv/bin/python -I \
        ../gollym/core/testdata/golden.py > ../gollym/core/testdata/pyllym_golden.json
"""

import json
import mimetypes
import sys

# gollym embeds Python's built-in table (no /etc/mime.types and friends).
mimetypes.knownfiles = []
mimetypes.inited = False
mimetypes.init()

import pyllym
from pyllym import Tool, utils
from pyllym.attachment import Attachment
from pyllym.chat import Chat
from pyllym.chunk import Chunk
from pyllym.connection import parse_error_body, retry_after_seconds
from pyllym.cost import Cost
from pyllym.errors import error_for_status
from pyllym.model.info import Info
from pyllym.protocols.chat_completions import ChatCompletions
from pyllym.stream_accumulator import StreamAccumulator
from pyllym.tokens import Tokens

out = {}

out["parse_error_body"] = [
    [b, parse_error_body(b)]
    for b in [
        "plain",
        '{"error": "e"}',
        {"error": {"message": "m"}},
        {"message": "x"},
        {"detail": "d"},
        {"error": {"code": 1}},
        [{"error": {"message": "a"}}, {"error": "b"}, "x", {"error": {"message": ""}}],
        {"error": {"message": 5}, "message": "fallback"},
    ]
]

pyllym.config().openai_api_key = "sk-test"
proto = ChatCompletions(pyllym.create_chat(model="gpt-4o").provider, None)
stream_errors = []
for data in [
    '{"error": {"type": "overloaded_error", "message": "Overloaded"}}',
    '{"error": {"code": "rate_limit_exceeded", "message": "slow"}}',
    '{"error": {"type": "insufficient_quota"}}',
    '{"error": {"type": "server_error", "message": "boom"}}',
    '{"error": "plain string"}',
    '{"message": "top"}',
    "not json",
    '{"error": {"code": 1302, "message": "1302 too fast"}}',
]:
    try:
        proto._raise_stream_error(data)
    except Exception as e:  # noqa: BLE001
        stream_errors.append([data, type(e).__name__, str(e)])
out["stream_errors"] = stream_errors

names = ["a.txt", "a.PDF", "a.json", ".json", "x.tar.gz", "a.md", "a.csv", "a.docx", "a.mp3",
         "a.wav", "a.webp", "a.heic", "noext", "a.yaml", "a.py", "a.svg", "a.m4a", "a.html",
         "a.xlsx", "a.key", "a.rtf", "a.js", "a.ts", "a.mp4", "a.mov", "a.jpeg", "a.ogg",
         "a.flac", "a.toml", "a.xml", "a.go", "a.sh", "dir.d/file", "a.doc", "a.Jpg"]
atts = []
for n in names:
    a = Attachment(b"", filename=n)
    atts.append([n, a.mime_type, a.type, a.format, a.extension])
out["attachments"] = atts

out["url_filenames"] = [
    [u, Attachment(u).filename]
    for u in ["https://x.com/a/b.png", "https://x.com/", "https://x.com/dir/", "https://x.com", "https://x.com/f.pdf?x=1"]
]

out["default_names"] = [[m, Info.default(m, "openai").name] for m in ["gpt-4o-mini-x", "my-Model", "über-model", ""]]

out["underscore"] = [[s, utils.underscore(s)] for s in ["HTTPProxyTool", "OpenAI", "getHTTPResponseCode", "ABC", "a1B2", "Already_snake"]]

tool_names = []
for cls_name in ["WeatherLookupTool", "HTTPTool", "Tool2", "Über_Tool", "Sum$Tool", "tool"]:
    cls = type(cls_name, (Tool,), {"execute": lambda self: None})
    tool_names.append([cls_name, cls().name])
out["tool_names"] = tool_names

priced = Info({"id": "p", "provider": "openai", "pricing": {
    "text_tokens": {"standard": {"input_per_million": 2.0, "output_per_million": 8.0,
                                 "cache_read_input_per_million": 0.5, "cache_write_input_per_million": 2.5,
                                 "reasoning_output_per_million": 10.0}},
    "images": {"standard": {"input_per_million": 5.0, "output_per_million": 40.0}}}})
same_reasoning = Info({"id": "q", "provider": "openai", "pricing": {
    "text_tokens": {"standard": {"input_per_million": 1.0, "output_per_million": 4.0, "reasoning_output_per_million": 4.0}}}})
unpriced = Info({"id": "u", "provider": "openai"})
costs = {
    "full": Cost(tokens=Tokens.build(input=1000, output=2000, cached=300, cache_creation=40, thinking=500), model=priced),
    "same_reasoning": Cost(tokens=Tokens.build(input=1000, output=2000, thinking=500), model=same_reasoning),
    "zero_tokens": Cost(tokens=Tokens.build(input=0, output=0), model=unpriced),
    "unpriced": Cost(tokens=Tokens.build(input=10), model=unpriced),
    "no_tokens": Cost(tokens=None, model=priced),
    "cache_only_zero": Cost(tokens=Tokens.build(cache_creation=0), model=unpriced),
    "image": Cost(tokens=Tokens.build(input=100, output=1000), model=priced, category="images",
                  input_details={"text_tokens": 40, "image_tokens": 60}),
    "image_no_details": Cost(tokens=Tokens.build(input=100, output=1000), model=priced, category="images"),
}
out["costs"] = {k: c.to_dict() for k, c in costs.items()}
agg = Cost.aggregate([costs["full"], costs["unpriced"], None, costs["no_tokens"]])
out["aggregate_missing"] = {"dict": agg.to_dict(), "input_missing": agg.is_missing("input")}
agg = Cost.aggregate([costs["full"], costs["full"]])
out["aggregate_full"] = agg.to_dict()

think_cases = [
    ["<think>a</think>b"],
    ["<thi", "nk>x", "</th", "ink>y"],
    ["a<", "b"],
    ["<think>unfinished"],
    ["x</think>y"],
    ["<think></think><think>z</think>w"],
    ["pre<think", ">mid"],
]
thinks = []
for chunks in think_cases:
    acc = StreamAccumulator()
    for c in chunks:
        acc.add(Chunk(role="assistant", content=c))
    m = acc.to_message(None)
    thinks.append([chunks, m.content, m.thinking.text if m.thinking else None])
out["think_tags"] = thinks

out["retry_after"] = [
    [h, retry_after_seconds(h)]
    for h in [
        {"retry-after": "0"},
        {"retry-after": "-5"},
        {"retry-after-ms": "abc", "retry-after": "3"},
        {"x-ratelimit-reset-tokens": "1m30s"},
        {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1.5s"},
        {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "20ms", "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "2s"},
        {"x-ratelimit-reset-requests": "5s junk"},
    ]
]

out["error_for_status"] = [
    [s, m, error_for_status(s, m).__name__]
    for s, m in [(400, None), (401, None), (402, None), (403, None), (404, None), (413, "too many tokens"),
                 (413, None), (422, None), (429, None), (429, "You exceeded your current quota"),
                 (429, "maximum context length"), (500, None), (501, None), (502, None), (503, None),
                 (504, None), (529, None), (418, None), (302, None)]
]

c = Chat.__new__(Chat)
out["schemas"] = [
    [s, c._normalize_schema_payload(s)]
    for s in [
        {"type": "object", "properties": {}},
        {"name": "x y", "schema": {"type": "object", "strict": False}},
        {"name": "", "schema": {"type": "object"}, "strict": None, "description": None},
        {"name": "ok-name_1", "schema": {"type": "object"}, "description": "desc"},
        {"type": "object", "strict": True},
        {"name": "!!!", "schema": {}},
    ]
]

json.dump(out, sys.stdout, indent=1, sort_keys=True, ensure_ascii=False)
