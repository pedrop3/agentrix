"""Writer sub-agent: so tem acesso a save_note."""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config import config
from logger import LLMLogger

PROMPT = """/no_think
You are a writer agent. Your ONLY job is to save notes for the user.

Rules:
- Use the `save_note` tool with a short, descriptive title and well-formatted
  markdown content.
- Pick a clear title yourself based on the user's request.
- After saving, confirm what you saved (title + a one-line summary).
"""


def build_writer(tools, model_name: str = None):
    name = model_name or config.ollama.writer_model
    model = ChatOllama(
        model=name,
        temperature=config.ollama.temperature,
        num_ctx=config.ollama.num_ctx,
        callbacks=[LLMLogger("writer")],
    )
    my_tools = [t for t in tools if t.name == "save_note"]
    return create_react_agent(model, tools=my_tools, prompt=PROMPT)
