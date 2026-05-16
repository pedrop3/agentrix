"""
Callback de logging para inspecionar todas as invocacoes ao LLM e tool calls.
"""
import textwrap
from langchain_core.callbacks import BaseCallbackHandler


class LLMLogger(BaseCallbackHandler):
    def __init__(self, agent_name: str = "llm", max_width: int = 800):
        self.agent_name = agent_name
        self.max_width = max_width

    def _short(self, text: str) -> str:
        return textwrap.shorten(str(text), width=self.max_width, placeholder="…[truncado]")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"\n{'═' * 64}")
        print(f"  [{self.agent_name.upper()}] INPUT")
        print(f"{'═' * 64}")
        for msg_list in messages:
            for msg in msg_list:
                role = getattr(msg, "type", type(msg).__name__)
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                if content:
                    print(f"[{role}]\n{self._short(content)}\n")

    def on_llm_end(self, response, **kwargs):
        print(f"{'─' * 64}")
        print(f"  [{self.agent_name.upper()}] OUTPUT")
        print(f"{'─' * 64}")
        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                text = getattr(msg, "content", "") if msg else getattr(gen, "text", "")
                if isinstance(text, list):
                    text = str(text)
                if text:
                    print(self._short(text))
                tool_calls = getattr(msg, "tool_calls", []) if msg else []
                for tc in tool_calls:
                    print(f"  → tool_call: {tc.get('name')}({tc.get('args')})")
        print()

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?")
        print(f"  → tool call : {name}({self._short(input_str)})")

    def on_tool_end(self, output, **kwargs):
        print(f"  ← tool result: {self._short(str(output))}\n")

    def on_tool_error(self, error, **kwargs):
        print(f"  ✗ tool error : {error}\n")
