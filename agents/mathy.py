"""Mathy sub-agent: so tem acesso a calc."""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config import config
from logger import LLMLogger


PROMPT = """/no_think
You are a math agent. Your ONLY job is to compute math expressions.

Rules:
- ALWAYS use the `calc` tool. Never compute mentally, even for simple math.
- Pass the full expression as a single string (e.g., "23 * 47 + 100").
- Return only the final result with a brief explanation.
"""


def build_mathy(tools, model_name: str = None):
    name = model_name or config.ollama.mathy_model
    model = ChatOllama(
        model=name,
        temperature=config.ollama.temperature,
        num_ctx=config.ollama.num_ctx,
        callbacks=[LLMLogger("mathy")],
    )
    my_tools = [t for t in tools if t.name == "calc"]
    return create_react_agent(model, tools=my_tools, prompt=PROMPT)
