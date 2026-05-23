"""Researcher sub-agent: so tem acesso a search_notes."""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config import config
from logger import LLMLogger

PROMPT = """/no_think
You are a research agent. Your ONLY job is to search the user's saved notes
to find information they're asking about.

Rules:
- Always use the `search_notes` tool. Never answer from memory.
- After getting results, return a concise summary of what was found.
- If nothing relevant turns up, say so plainly.
"""


def build_researcher(tools, model_name: str = None):
    name = model_name or config.ollama.researcher_model
    model = ChatOllama(
        model=name,
        temperature=config.ollama.temperature,
        num_ctx=config.ollama.num_ctx,
        callbacks=[LLMLogger("researcher")],
    )
    my_tools = [t for t in tools if t.name == "search_notes"]
    return create_react_agent(model, tools=my_tools, prompt=PROMPT)
