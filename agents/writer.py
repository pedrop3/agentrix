"""Writer sub-agent: so tem acesso a save_note."""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from logger import LLMLogger

PROMPT = """You are a writer agent. Your ONLY job is to save notes for the user.

Rules:
- Use the `save_note` tool with a short, descriptive title and well-formatted
  markdown content.
- Pick a clear title yourself based on the user's request.
- After saving, confirm what you saved (title + a one-line summary).
"""


def build_writer(tools, model_name: str = "qwen3:14b"):
    model = ChatOllama(model=model_name, temperature=0, num_ctx=8192, callbacks=[LLMLogger("writer")])
    my_tools = [t for t in tools if t.name == "save_note"]
    return create_react_agent(model, tools=my_tools, prompt=PROMPT)
