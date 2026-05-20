"""Mathy sub-agent: so tem acesso a calc."""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from logger import LLMLogger


PROMPT = """You are a math agent. Your ONLY job is to compute math expressions.

Rules:
- ALWAYS use the `calc` tool. Never compute mentally, even for simple math.
- Pass the full expression as a single string (e.g., "23 * 47 + 100").
- Return only the final result with a brief explanation.
"""


def build_mathy(tools, model_name: str = "qwen3:14b"):
    model = ChatOllama(model=model_name, temperature=0, num_ctx=8192, callbacks=[LLMLogger("mathy")])
    my_tools = [t for t in tools if t.name == "calc"]
    return create_react_agent(model, tools=my_tools, prompt=PROMPT)
