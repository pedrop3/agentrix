"""
Logger de invocações LLM — mostra step-by-step, input/output e consumo de tokens.
"""
import textwrap
import threading
from langchain_core.callbacks import BaseCallbackHandler

# ── Contador global de steps (compartilhado entre todos os agentes) ───────────
_lock = threading.Lock()
_step = 0

def reset_steps():
    global _step
    with _lock:
        _step = 0

def _next_step() -> int:
    global _step
    with _lock:
        _step += 1
        return _step

# ── Helpers de formatação ──────────────────────────────────────────────────────
W = 72  # largura total da caixa

_ROLE_LABEL = {
    "system":    "SYSTEM ",
    "human":     "USER   ",
    "ai":        "AI     ",
    "tool":      "TOOL   ",
    "HumanMessage":   "USER   ",
    "AIMessage":      "AI     ",
    "SystemMessage":  "SYSTEM ",
    "ToolMessage":    "TOOL   ",
}

def _role(msg) -> str:
    t = getattr(msg, "type", None) or type(msg).__name__
    return _ROLE_LABEL.get(t, t.upper()[:7].ljust(7))

def _content(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c).strip()

def _wrap(text: str, indent: int = 11, max_lines: int = 6) -> str:
    """Quebra o texto em linhas com indentação, limitando a max_lines."""
    prefix = " " * indent
    available = W - indent - 2
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        lines.extend(textwrap.wrap(raw, width=available) or [raw])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"… ({len(lines) - max_lines} linhas omitidas)"]
    return "\n".join(f"{prefix}{l}" for l in lines)

def _bar(char: str = "─") -> str:
    return f"  {char * (W - 4)}"

def _header(step: int, agent: str, tag: str) -> str:
    label = f" STEP {step} │ {agent.upper()} │ {tag} "
    pad = W - len(label) - 2
    return f"\n╔{label}{'═' * max(pad, 0)}╗"

def _footer() -> str:
    return f"╚{'═' * W}╝"

def _section(title: str) -> str:
    return f"║\n║  ┌─ {title}"

def _msg_line(msg) -> str:
    role = _role(msg)
    content = _content(msg)
    first_line = f"  [{role}]  {content}"
    if len(first_line) <= W + 2:
        return f"║{first_line}"
    # conteúdo longo: segunda linha indentada
    wrapped = _wrap(content)
    return f"║  [{role}]\n{chr(10).join('║' + l for l in wrapped.splitlines())}"


# ── Callback handler ───────────────────────────────────────────────────────────
class LLMLogger(BaseCallbackHandler):
    def __init__(self, agent_name: str = "llm"):
        self.agent_name = agent_name
        self._current_step = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._current_step = _next_step()
        n_msgs = sum(len(ml) for ml in messages)
        print(_header(self._current_step, self.agent_name, f"INPUT  ({n_msgs} msg)"))
        print(_bar())
        for msg_list in messages:
            for msg in msg_list:
                print(_msg_line(msg))
        print(f"║")

    def on_llm_end(self, response, **kwargs):
        print(f"║")
        print(f"║  ┌─ OUTPUT")
        print(_bar())

        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                text = (_content(msg) if msg else getattr(gen, "text", "")).strip()
                tool_calls = getattr(msg, "tool_calls", []) if msg else []

                if text:
                    for line in textwrap.wrap(text, width=W - 4) or [text]:
                        print(f"║  {line}")

                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    print(f"║  ⤷ tool_call  {name}({args})")

        # ── Tokens ────────────────────────────────────────────────────────────
        prompt_tok = compl_tok = 0
        for gens in response.generations:
            for gen in gens:
                # 1ª fonte: usage_metadata na mensagem (langchain-ollama >= 0.2)
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None)
                if um:
                    prompt_tok += um.get("input_tokens", 0)
                    compl_tok  += um.get("output_tokens", 0)
                    continue
                # 2ª fonte: generation_info (prompt_eval_count / eval_count)
                gi = getattr(gen, "generation_info", None) or {}
                prompt_tok += gi.get("prompt_eval_count", 0)
                compl_tok  += gi.get("eval_count", 0)
        # 3ª fonte: llm_output (fallback OpenAI-style)
        if prompt_tok == 0 and compl_tok == 0:
            lo = getattr(response, "llm_output", None) or {}
            if "token_usage" in lo:
                tu = lo["token_usage"]
                prompt_tok = tu.get("prompt_tokens", 0)
                compl_tok  = tu.get("completion_tokens", 0)

        total = prompt_tok + compl_tok
        print(f"║")
        print(f"║  ┌─ TOKENS")
        if total > 0:
            print(f"║  │  prompt: {prompt_tok:>6}   completion: {compl_tok:>6}   total: {total:>6}")
        else:
            print(f"║  │  (não disponível para este backend)")

        print(_footer())

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?")
        short = textwrap.shorten(str(input_str), width=W - 20, placeholder="…")
        print(f"║")
        print(f"║  ► TOOL CALL   {name}  ←  {short}")

    def on_tool_end(self, output, **kwargs):
        short = textwrap.shorten(str(output), width=W - 20, placeholder="…")
        print(f"║  ◄ TOOL RESULT {short}")

    def on_tool_error(self, error, **kwargs):
        print(f"║  ✗ TOOL ERROR  {error}")
