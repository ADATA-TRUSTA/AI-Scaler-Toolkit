"""
Identify a model's vLLM tool-call parser by round-tripping one tool call.

Run with the isolated vllm_server venv's interpreter, not the backend's:

    <vllm_server>/.venv/bin/python vllm_tool_parser_probe.py <model ref>

Emits ``TRUSTA_TOOL_PARSER=<name>`` on stdout, or nothing when no parser fits.
The marker matters: importing the parsers makes vLLM log INFO lines to stdout,
so the caller has to pick the answer out rather than read all of it.

Deliberately
imports only vllm/transformers, never ``service``: those packages live in the
isolated environment and the backend's are not importable here.

Why round-trip instead of a marker table: the model's chat template renders the
exact text the model is expected to emit for a tool call, and vLLM's own parsers
say whether they can read it back. That derives the answer from the two
authorities involved, so it needs no hand-maintained mapping and cannot drift
when vLLM renames a parser or changes its syntax.
"""

import sys

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "trusta_probe_fn",
            "description": "probe",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
_CALL = [
    {
        "id": "probe1",
        "type": "function",
        "function": {"name": "trusta_probe_fn", "arguments": {"city": "Taipei"}},
    }
]
_MESSAGES = [
    {"role": "user", "content": "probe"},
    {"role": "assistant", "content": None, "tool_calls": _CALL},
]


def main(model_ref: str) -> int:
    """Print the parser that round-trips this model's tool-call syntax."""
    from transformers import AutoTokenizer

    # vllm lives only in the isolated vllm_server venv this script runs under, so
    # it is unresolvable from the backend environment that type-checks this file.
    from vllm.tool_parsers import ToolParserManager  # pyright: ignore[reportMissingImports]

    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)

    # Render what the model is expected to emit. A template with no tool support
    # raises here, which is the correct answer: this model has no tool syntax.
    rendered = tokenizer.apply_chat_template(_MESSAGES, tools=_TOOLS, tokenize=False)
    if "trusta_probe_fn" not in rendered:
        # Template accepted tool_calls but dropped them; nothing to detect.
        return 1

    # Feed the parsers the assistant turn only, the way serving would.
    start = rendered.find("trusta_probe_fn", rendered.find("probe") + 1)
    segment = rendered[max(0, start - 400) :]

    names = sorted(set(ToolParserManager.tool_parsers) | set(ToolParserManager.lazy_parsers))
    matches: list[str] = []
    for name in names:
        try:
            parser = ToolParserManager.get_tool_parser(name)(tokenizer)
            info = parser.extract_tool_calls(segment, request=None)  # type: ignore[arg-type]
        except Exception:
            continue
        calls = getattr(info, "tool_calls", None) or []
        if getattr(info, "tools_called", False) and calls:
            if calls[0].function.name == "trusta_probe_fn":
                matches.append(name)

    if not matches:
        return 1
    if len(matches) > 1:
        # Ambiguous: refuse rather than pick arbitrarily, and say what tied.
        sys.stderr.write("ambiguous: " + ",".join(matches) + "\n")
        return 2
    sys.stdout.write(f"\nTRUSTA_TOOL_PARSER={matches[0]}\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: vllm_tool_parser_probe.py <model ref>\n")
        raise SystemExit(64)
    try:
        raise SystemExit(main(sys.argv[1]))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot detect"
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        raise SystemExit(1) from None
