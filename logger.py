from __future__ import annotations

import json
import re
import textwrap
import threading
import time
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

W = 78  # line width

# ─── Global turn state ────────────────────────────────────────────────────────
_lock = threading.Lock()
_global_step = 0
_turn_start_ts: float | None = None
_turn_breakdown: dict[str, float] = defaultdict(float)
_turn_calls: dict[str, int] = defaultdict(int)
_loggers: list["LLMLogger"] = []


def get_logger(agent_name: str) -> "LLMLogger | None":
    """Return the registered LLMLogger for agent_name, or None."""
    return next((l for l in _loggers if l.agent_name == agent_name), None)


def reset_steps() -> None:
    global _global_step, _turn_start_ts
    with _lock:
        _global_step = 0
        _turn_start_ts = time.perf_counter()
        _turn_breakdown.clear()
        _turn_calls.clear()
    for lgr in _loggers:
        lgr._reset_turn()


def _next_global_step() -> int:
    global _global_step
    with _lock:
        _global_step += 1
        return _global_step


# ─── Turn summary ─────────────────────────────────────────────────────────────
def turn_summary() -> str:
    if _turn_start_ts is None:
        return ""
    elapsed = time.perf_counter() - _turn_start_ts
    agents = {k: v for k, v in _turn_breakdown.items() if not k.startswith("tool:")}
    tools_total = sum(v for k, v in _turn_breakdown.items() if k.startswith("tool:"))
    parts = []
    for name, t in sorted(agents.items(), key=lambda x: -x[1]):
        calls = _turn_calls.get(name, 0)
        parts.append(f"{name} {t:.2f}s ({calls}x)")
    if tools_total > 0:
        parts.append(f"tools {tools_total:.2f}s")
    bar = " │ ".join(parts)
    sep = "─" * W
    return f"\n{sep}\nTOTAL {elapsed:.2f}s │ {bar}\n{sep}"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _strip_think(text: str) -> tuple[str, str]:
    think_parts = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    think = "\n".join(p.strip() for p in think_parts if p.strip())
    return think, visible


def _split_gemini_content(msg) -> tuple[str, str]:
    """Split an LLM message into (thought, visible) parts.

    - Ollama / plain string  → delegates to _strip_think (handles qwen3 <think> tags).
    - Gemini list blocks     → separates blocks where ``thought=True`` from normal text.
    - Mixed                  → merges both sources of thinking.
    Returns (thought_text, visible_text).
    """
    c = getattr(msg, "content", "")
    if not isinstance(c, list):
        # Plain string content — handle qwen3-style <think> tags
        return _strip_think(str(c).strip())

    # Gemini-style: content is a list of typed blocks
    thought_parts: list[str] = []
    visible_parts: list[str] = []
    for block in c:
        if isinstance(block, dict):
            text = block.get("text", "")
            if block.get("thought", False):
                thought_parts.append(text)
            else:
                visible_parts.append(text)
        else:
            visible_parts.append(str(block))

    thought = " ".join(p for p in thought_parts if p).strip()
    visible = " ".join(p for p in visible_parts if p).strip()

    # Also strip any qwen3-style tags that might appear inside visible text
    if "<think>" in visible:
        extra_think, visible = _strip_think(visible)
        if extra_think:
            thought = (thought + " " + extra_think).strip()

    return thought, visible


def _content(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c).strip()


def _fmt_args(args: dict) -> str:
    if not args:
        return "()"
    parts = []
    for k, v in args.items():
        vs = repr(v) if isinstance(v, str) else str(v)
        if len(vs) > 55:
            vs = vs[:52] + "…'"
        parts.append(f"{k}={vs}")
    s = ", ".join(parts)
    if len(s) > W - 18:
        s = s[:W - 21] + "…"
    return s


def _fmt_tool_input(input_str: str) -> str:
    """Parse tool args from JSON or Python-dict repr into key=val format."""
    # 1) Try JSON  (standard format, e.g. Ollama)
    try:
        args = json.loads(input_str)
        if isinstance(args, dict):
            return _fmt_args(args)
    except (json.JSONDecodeError, ValueError):
        pass
    # 2) Try Python literal  (Gemini serialises as {'key': 'val'})
    try:
        import ast
        args = ast.literal_eval(input_str)
        if isinstance(args, dict):
            return _fmt_args(args)
    except (ValueError, SyntaxError):
        pass
    return textwrap.shorten(str(input_str), width=W - 18, placeholder="…")


def _extract_text(raw) -> str:
    """Extract plain text from any tool-output shape.

    Handles:
    - plain str
    - LangChain ToolMessage / any object with .content
    - list of content blocks: [{'type': 'text', 'text': '...'}, ...]
      (Gemini and Anthropic both use this format for structured output)
    - nested: .content is itself a list of blocks
    """
    # Unwrap .content if present (ToolMessage, AIMessage, etc.)
    if hasattr(raw, "content"):
        raw = raw.content

    # List of content blocks → join the 'text' fields
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("value") or str(block))
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)

    return str(raw)


def _fmt_result(raw, max_len: int = 400) -> str:
    """Collapse whitespace and truncate tool output for display."""
    text = _extract_text(raw)
    text = " ".join(text.split())   # normalise whitespace
    if len(text) > max_len:
        return text[:max_len - 1] + "…"
    return text


def _get_user_question(messages) -> str:
    for msg_list in reversed(messages):
        for msg in reversed(msg_list):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                c = _content(msg)
                c = re.sub(r"\s*/no_think\s*$", "", c).strip()
                if c:
                    return c
    return ""


# ─── Callback ─────────────────────────────────────────────────────────────────
class LLMLogger(BaseCallbackHandler):
    def __init__(self, agent_name: str = "llm"):
        self.agent_name = agent_name
        self._reset_turn()
        _loggers.append(self)

    def _reset_turn(self):
        self._first_call = True
        self._t0_llm: float | None = None
        self._tool_t0: dict[str, tuple[float, str]] = {}
        self._agent_step = 0

    # ── LLM events ────────────────────────────────────────────────────────────

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._t0_llm = time.perf_counter()
        _next_global_step()

        if not self._first_call:
            return
        self._first_call = False

        if self.agent_name == "supervisor":
            question = _get_user_question(messages)
            sep = "─" * W
            print(f"\n{sep}")
            if question:
                print(f"TURN  {question}")
            print(f"{sep}\n")
        else:
            pad = max(W - 4 - len(self.agent_name), 2)
            print(f"\n── {self.agent_name} {'─' * pad}")

    def on_llm_end(self, response, **kwargs):
        elapsed = time.perf_counter() - (self._t0_llm or time.perf_counter())
        with _lock:
            _turn_breakdown[self.agent_name] += elapsed
            _turn_calls[self.agent_name] += 1

        if self.agent_name == "supervisor":
            return  # routing decision is printed by graph.py

        tool_calls: list[dict] = []
        thinking_text = ""
        visible_text = ""

        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                tool_calls.extend(getattr(msg, "tool_calls", []) if msg else [])
                if msg:
                    thought, visible = _split_gemini_content(msg)
                else:
                    text = getattr(gen, "text", "").strip()
                    thought, visible = _strip_think(text)
                if thought:
                    thinking_text += thought + "\n"
                if visible:
                    visible_text += visible + "\n"

        thinking_text = thinking_text.strip()
        visible_text = visible_text.strip()

        if thinking_text:
            # Show a single-line preview of the model's internal reasoning
            short = textwrap.shorten(
                thinking_text.replace("\n", " "), width=W - 6, placeholder="…"
            )
            print(f"  ┆ {short}")
        if tool_calls and visible_text:
            # Agent narrated its reasoning before the tool call
            short = textwrap.shorten(visible_text, width=W - 5, placeholder="…")
            print(f"  ┄ {short}")
        elif not tool_calls and visible_text:
            # Final answer — no more tool calls
            pad = max(W - 10, 2)
            print(f"\n── ANSWER {'─' * pad}")
            for raw_line in visible_text.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    print()
                    continue
                for chunk in textwrap.wrap(raw_line, width=W - 3) or [raw_line]:
                    print(f"   {chunk}")
            print("─" * W)
        # else: pure tool-call response — on_tool_start prints the step line

    def on_llm_error(self, error, **kwargs):
        elapsed = time.perf_counter() - (self._t0_llm or time.perf_counter())
        with _lock:
            _turn_breakdown[self.agent_name] += elapsed
            _turn_calls[self.agent_name] += 1
        print(f"\n  ✗ [{self.agent_name}] {elapsed:.2f}s  {error}")

    # ── Tool events ────────────────────────────────────────────────────────────

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?") if isinstance(serialized, dict) else "?"
        run_id = str(kwargs.get("run_id", ""))
        self._agent_step += 1
        step = self._agent_step
        self._tool_t0[run_id] = (time.perf_counter(), name)
        args = _fmt_tool_input(str(input_str))
        print(f" {step:>2}  {name}  {args}")

    def on_tool_end(self, output, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        info = self._tool_t0.pop(run_id, None)
        elapsed = (time.perf_counter() - info[0]) if info else 0.0
        name = info[1] if info else "?"
        with _lock:
            _turn_breakdown[f"tool:{name}"] += elapsed
            _turn_calls[f"tool:{name}"] += 1

        full = _fmt_result(output)     # handles ToolMessage, str, content-block lists
        timing = f"{elapsed:.2f}s"
        first_w = W - 7 - len(timing)  # chars available on the first line

        first = full[:first_w]
        rest  = full[first_w:]

        # First line: result + right-aligned timing (always W chars wide)
        pad = " " * (first_w - len(first)) if len(first) < first_w else " "
        print(f"     ╰ {first}{pad}{timing}")

        # Optional continuation line — shows the next slice of the result
        if rest:
            cont = rest[:first_w]
            ellipsis = "…" if len(rest) > first_w else ""
            print(f"       {cont}{ellipsis}")

    def on_tool_error(self, error, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        info = self._tool_t0.pop(run_id, None)
        elapsed = (time.perf_counter() - info[0]) if info else 0.0
        print(f"     ╰ ✗ {error}  ({elapsed:.2f}s)")


# ─── ToolOnlyLogger ───────────────────────────────────────────────────────────
class ToolOnlyLogger(BaseCallbackHandler):
    """Thin proxy that forwards ONLY tool events to a parent LLMLogger.

    Why this exists
    ---------------
    In LangGraph's create_react_agent the LLM node and the ToolNode are
    separate graph nodes. Callbacks attached to the LLM object (static) fire
    for LLM events but are NOT inherited by the ToolNode.  Passing this proxy
    in the *invocation config* (config={"callbacks": [ToolOnlyLogger(...)]})
    makes LangGraph propagate it to every child node — including ToolNode —
    so on_tool_start / on_tool_end fire correctly.

    All LLM-level handlers are intentional no-ops here to prevent the same
    event from being logged twice (once from the static callback on the model,
    once from this proxy in the invocation config).
    """

    def __init__(self, parent: LLMLogger) -> None:
        super().__init__()
        self._parent = parent

    # ── delegate tool events ──────────────────────────────────────────────────

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._parent.on_tool_start(serialized, input_str, **kwargs)

    def on_tool_end(self, output, **kwargs):
        self._parent.on_tool_end(output, **kwargs)

    def on_tool_error(self, error, **kwargs):
        self._parent.on_tool_error(error, **kwargs)

    # ── explicit no-ops for LLM events (prevent double logging) ──────────────

    def on_chat_model_start(self, *args, **kwargs): pass
    def on_llm_end(self, *args, **kwargs): pass
    def on_llm_error(self, *args, **kwargs): pass
