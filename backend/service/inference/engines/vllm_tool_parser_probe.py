"""
Identify a model's vLLM tool-call parser by round-tripping one tool call.

Run with the interpreter vLLM runs from, not necessarily the caller's:

    <VLLM_SERVER_PROJECT_DIR>/.venv/bin/python vllm_tool_parser_probe.py <model ref>

Emits ``TRUSTA_TOOL_PARSER=<name>`` on stdout, or nothing when no parser fits.
The marker matters: importing the parsers makes vLLM log INFO lines to stdout,
so the caller has to pick the answer out rather than read all of it.

Deliberately
imports only vllm/transformers, never ``service``: those packages live in the
vLLM environment and the backend's are not importable here.

Why round-trip instead of a marker table: the model's chat template renders the
exact text the model is expected to emit for a tool call, and vLLM's own parsers
say whether they can read it back. That derives the answer from the two
authorities involved, so it needs no hand-maintained mapping and cannot drift
when vLLM renames a parser or changes its syntax.
"""

import json
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
# content is "" rather than the more literally-correct None: some chat templates
# (Qwen3-VL-4B-Instruct's included) only guard the *string* branch of "does this
# message have text content" and unconditionally iterate the non-string branch
# assuming it's always a list of parts, so a None content raises
# "TypeError: 'NoneType' object is not iterable" while rendering this history --
# a template bug unrelated to whether the model can actually make tool calls, but
# one that would otherwise make every probe against that template fail closed.
# "" satisfies `content is string` in every template seen so far and renders as
# no text, which is what a real content: null turn is trying to express anyway.
_MESSAGES = [
    {"role": "user", "content": "probe"},
    {"role": "assistant", "content": "", "tool_calls": _CALL},
]


def main(model_ref: str) -> int:
    """Print the parser that round-trips this model's tool-call syntax."""
    from transformers import AutoTokenizer

    # vllm lives in the venv this script runs under, not necessarily the caller's, so
    # it is unresolvable from the backend environment that type-checks this file.
    from vllm.tool_parsers import ToolParserManager  # pyright: ignore[reportMissingImports]

    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)

    # Render what the model is expected to emit. A template with no tool support
    # raises here, which is the correct answer: this model has no tool syntax.
    rendered = tokenizer.apply_chat_template(_MESSAGES, tools=_TOOLS, tokenize=False)
    if "trusta_probe_fn" not in rendered:
        # Template accepted tool_calls but dropped them; nothing to detect.
        return 1

    # Feed the parsers the assistant turn only, the way serving would: render the
    # same conversation without the assistant turn (generation prompt only) and
    # diff it against the full render. Some chat templates (Qwen's included) spell
    # out a <tool_call> *example* in the system instructions, before the real one;
    # a fixed lookbehind window can include that example and feed a parser
    # malformed placeholder JSON it then fails on. The diff is exact regardless of
    # how verbose those instructions are.
    prompt_only = tokenizer.apply_chat_template(
        _MESSAGES[:-1], tools=_TOOLS, tokenize=False, add_generation_prompt=True
    )
    if not rendered.startswith(prompt_only):
        # Template renders the assistant turn some other way this cannot isolate.
        return 1
    segment = rendered[len(prompt_only) :]

    expected_args = _CALL[0]["function"]["arguments"]
    names = sorted(set(ToolParserManager.tool_parsers) | set(ToolParserManager.lazy_parsers))
    matches: list[str] = []
    for name in names:
        try:
            parser = ToolParserManager.get_tool_parser(name)(tokenizer)
            info = parser.extract_tool_calls(segment, request=None)  # type: ignore[arg-type]
        except Exception:
            continue
        calls = getattr(info, "tool_calls", None) or []
        if not (getattr(info, "tools_called", False) and calls):
            continue
        call = calls[0]
        if call.function.name != "trusta_probe_fn":
            continue
        # Match the arguments too, not just the function name: a parser that merely
        # tolerates the wrapper but mangles the JSON is not a real match. This is
        # what disambiguates families (e.g. Qwen2.5) where several parsers loosely
        # accept the same <tool_call> envelope.
        try:
            parsed_args = json.loads(call.function.arguments or "{}")
        except (ValueError, TypeError):
            continue
        if parsed_args != expected_args:
            continue
        matches.append(name)

    if not matches:
        return 1
    if len(matches) > 1:
        # Hand the tie back to the backend, which owns the preference policy.
        sys.stdout.write("\nTRUSTA_TOOL_PARSER_AMBIGUOUS=" + ",".join(matches) + "\n")
        return 3
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
