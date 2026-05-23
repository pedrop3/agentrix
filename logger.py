from __future__ import annotations

import re
import textwrap
import threading
import time
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# ---------------------------------------------------------------------------
# Estado global (turno + step counter)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_step = 0
_turn_start_ts: float | None = None
_turn_breakdown: dict[str, float] = defaultdict(float)
_turn_calls: dict[str, int] = defaultdict(int)


def reset_steps() -> None:
    """Chame no inicio de cada TURN do usuario (uma pergunta = um turno)."""
    global _step, _turn_start_ts
    with _lock:
        _step = 0
        _turn_start_ts = time.perf_counter()
        _turn_breakdown.clear()
        _turn_calls.clear()


def _next_step() -> int:
    global _step
    with _lock:
        _step += 1
        return _step


def turn_summary() -> str:
    """Imprime sumario do turno."""
    if _turn_start_ts is None:
        return ""
    elapsed = time.perf_counter() - _turn_start_ts
    lines = [f"\n┏━ TURN SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"]
    lines.append(f"┃  total wall:        {elapsed:>7.2f}s")
    if _turn_breakdown:
        lines.append("┃  por papel:")
        for agent, t in sorted(_turn_breakdown.items(), key=lambda x: -x[1]):
            calls = _turn_calls.get(agent, 0)
            pct = (t / elapsed * 100) if elapsed > 0 else 0
            lines.append(f"┃    {agent:<14} {t:>6.2f}s  ({calls} calls, {pct:>4.1f}%)")
    lines.append(f"┗{'━' * 61}┛")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatacao (caixas / mensagens)
# ---------------------------------------------------------------------------
W = 76  # largura da caixa

_ROLE_LABEL = {
    "system": "SYSTEM",
    "human": "USER",
    "ai": "AI",
    "tool": "TOOL",
    "HumanMessage": "USER",
    "AIMessage": "AI",
    "SystemMessage": "SYSTEM",
    "ToolMessage": "TOOL",
}


def _role(msg) -> str:
    t = getattr(msg, "type", None) or type(msg).__name__
    return _ROLE_LABEL.get(t, t.upper()[:6])


def _content(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c).strip()


def _strip_think(text: str) -> tuple[str, str]:
    """Separa <think>...</think> do resto. Retorna (think_content, visible)."""
    think_parts = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    think = "\n".join(p.strip() for p in think_parts if p.strip())
    return think, visible


def _wrap(text: str, indent: int = 4, max_lines: int = 8) -> list[str]:
    prefix = " " * indent
    available = W - indent - 2
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        chunks = textwrap.wrap(raw, width=available) or [raw[:available]]
        lines.extend(chunks)
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = lines[: max_lines] + [f"… (+{omitted} linhas)"]
    return [f"{prefix}{l}" for l in lines]


def _header(step: int, agent: str, tag: str) -> str:
    label = f" #{step} · {agent.upper()} · {tag} "
    pad = W - len(label) - 2
    return f"\n╔{label}{'═' * max(pad, 0)}╗"


def _footer() -> str:
    return f"╚{'═' * W}╝"


def _line(text: str = "") -> str:
    return f"║  {text}"


def _sep() -> str:
    return f"║  {'─' * (W - 4)}"


def _msg_block(msg) -> list[str]:
    role = _role(msg)
    content = _content(msg)
    if not content:
        return [_line(f"[{role}]  (empty)")]
    # primeira linha curta inline
    first_try = f"[{role:<6}] {content}"
    if len(first_try) <= W - 4 and "\n" not in content:
        return [_line(first_try)]
    out = [_line(f"[{role:<6}]")]
    out.extend(_line(l) for l in _wrap(content, indent=4, max_lines=5))
    return out


# ---------------------------------------------------------------------------
# Callback principal
# ---------------------------------------------------------------------------
class LLMLogger(BaseCallbackHandler):
    def __init__(self, agent_name: str = "llm"):
        self.agent_name = agent_name
        # Estado por step ativo (LangChain pode aninhar; usamos lista pra empilhar)
        self._step_stack: list[dict[str, Any]] = []

    # ----- LLM events -----
    def on_chat_model_start(self, serialized, messages, **kwargs):
        step = _next_step()
        t0 = time.perf_counter()
        self._step_stack.append({"step": step, "t0": t0, "n_msgs": 0})

        n_msgs = sum(len(ml) for ml in messages)
        self._step_stack[-1]["n_msgs"] = n_msgs

        print(_header(step, self.agent_name, f"INPUT ({n_msgs} msg)"))
        print(_sep())
        for msg_list in messages:
            for msg in msg_list:
                for line in _msg_block(msg):
                    print(line)
        print(_line())

    def on_llm_end(self, response, **kwargs):
        if not self._step_stack:
            return
        frame = self._step_stack.pop()
        elapsed = time.perf_counter() - frame["t0"]

        # Acumula no breakdown do turno
        _turn_breakdown[self.agent_name] += elapsed
        _turn_calls[self.agent_name] += 1

        print(_line("┌─ OUTPUT"))
        print(_sep())

        think_total = ""
        visible_total = ""
        tool_calls_all: list[dict] = []

        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                text = (_content(msg) if msg else getattr(gen, "text", "")).strip()
                tool_calls_all.extend(getattr(msg, "tool_calls", []) if msg else [])
                think, visible = _strip_think(text)
                if think:
                    think_total += think + "\n"
                if visible:
                    visible_total += visible + "\n"

        if visible_total.strip():
            for line in _wrap(visible_total.strip(), indent=2, max_lines=10):
                print(_line(line))
        elif not tool_calls_all and not think_total:
            print(_line("(empty)"))

        for tc in tool_calls_all:
            name = tc.get("name", "?")
            args = tc.get("args", {})
            args_str = str(args)
            if len(args_str) > W - 30:
                args_str = args_str[: W - 33] + "..."
            print(_line(f"⤷ tool_call  {name}({args_str})"))

        if think_total.strip():
            think_lines = think_total.strip().splitlines()
            think_chars = len(think_total)
            print(_line())
            print(_line(f"┌─ [THINK] {len(think_lines)} linhas, {think_chars} chars (gastou tokens)"))
            # mostra so as primeiras 3 linhas pra dar uma ideia do que tava pensando
            for line in _wrap(think_total.strip(), indent=2, max_lines=3):
                print(_line(line))

        # Tokens
        prompt_tok = compl_tok = 0
        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None)
                if um:
                    prompt_tok += um.get("input_tokens", 0)
                    compl_tok += um.get("output_tokens", 0)
                    continue
                gi = getattr(gen, "generation_info", None) or {}
                prompt_tok += gi.get("prompt_eval_count", 0)
                compl_tok += gi.get("eval_count", 0)
        if prompt_tok == 0 and compl_tok == 0:
            lo = getattr(response, "llm_output", None) or {}
            tu = lo.get("token_usage") or {}
            prompt_tok = tu.get("prompt_tokens", 0)
            compl_tok = tu.get("completion_tokens", 0)
        total = prompt_tok + compl_tok

        gen_rate = (compl_tok / elapsed) if elapsed > 0 and compl_tok > 0 else 0
        print(_line())
        print(_line("┌─ STATS"))
        print(_line(
            f"  elapsed = {elapsed:>5.2f}s   "
            f"tokens(in/out/total) = {prompt_tok}/{compl_tok}/{total}   "
            f"gen = {gen_rate:.1f} tok/s"
        ))
        print(_footer())

    def on_llm_error(self, error, **kwargs):
        if self._step_stack:
            frame = self._step_stack.pop()
            elapsed = time.perf_counter() - frame["t0"]
            print(_line())
            print(_line(f"✗ ERROR after {elapsed:.2f}s: {error}"))
            print(_footer())
        else:
            print(f"✗ LLM error (no step active): {error}")

    # ----- Tool events (capturam tools chamadas via create_react_agent) -----
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?") if isinstance(serialized, dict) else "?"
        short = textwrap.shorten(str(input_str), width=W - 16, placeholder="…")
        run_id = kwargs.get("run_id")
        with _lock:
            self._tool_t0 = getattr(self, "_tool_t0", {})
            self._tool_t0[str(run_id)] = (time.perf_counter(), name)
        print(f"\n  ► TOOL  {self.agent_name} → {name}({short})")

    def on_tool_end(self, output, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        t0_map = getattr(self, "_tool_t0", {})
        info = t0_map.pop(run_id, None)
        elapsed = (time.perf_counter() - info[0]) if info else 0.0
        name = info[1] if info else "?"
        short = textwrap.shorten(str(output), width=W - 16, placeholder="…")
        # contabiliza no breakdown como "tool:<nome>"
        if info:
            _turn_breakdown[f"tool:{name}"] += elapsed
            _turn_calls[f"tool:{name}"] += 1
        print(f"  ◄ TOOL  {name}  done in {elapsed:.2f}s   → {short}")

    def on_tool_error(self, error, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        t0_map = getattr(self, "_tool_t0", {})
        info = t0_map.pop(run_id, None)
        elapsed = (time.perf_counter() - info[0]) if info else 0.0
        print(f"  ✗ TOOL  failed in {elapsed:.2f}s: {error}")
