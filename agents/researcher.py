"""Researcher sub-agent: so tem acesso a search_notes."""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from logger import LLMLogger

PROMPT = """You are a research agent. Your ONLY job is to search the user's saved notes
to find information they're asking about.

Rules:
- Always use the `search_notes` tool. Never answer from memory.
- After getting results, return a concise summary of what was found.
- If nothing relevant turns up, say so plainly.
"""


def build_researcher(tools, model_name: str = "qwen3:14b"):
    model = ChatOllama(model=model_name, temperature=0, num_ctx=8192, callbacks=[LLMLogger("researcher")])
    my_tools = [t for t in tools if t.name == "search_notes"]
    return create_react_agent(model, tools=my_tools, prompt=PROMPT)
