"""
Entry point. Sobe o MCP server via stdio, carrega as tools no LangChain,
constroi os sub-agents + supervisor, e abre um REPL no terminal.

Roda: python main.py
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.mathy import build_mathy
from agents.researcher import build_researcher
from agents.writer import build_writer
from graph import build_graph, build_supervisor

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
PROJECT_ROOT = Path(__file__).parent


async def main():
    print(f"Carregando modelo Ollama: {MODEL}")
    print("Subindo MCP server via stdio...")

    client = MultiServerMCPClient(
        {
            "switchboard": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "cwd": str(PROJECT_ROOT),
            }
        }
    )
    tools = await client.get_tools()
    print(f"Tools carregadas do MCP: {[t.name for t in tools]}")

    researcher = build_researcher(tools, MODEL)
    writer = build_writer(tools, MODEL)
    mathy = build_mathy(tools, MODEL)
    supervisor = build_supervisor(MODEL)
    graph = build_graph(researcher, writer, mathy, supervisor)

    print("\nSwitchboard pronto. Digite 'quit' para sair.\n")
    print("Exemplos:")
    print("  - salve uma nota sobre minha reuniao com pontos X, Y, Z")
    print("  - o que eu salvei sobre reunioes?")
    print("  - quanto e 23 * 47 + 100?\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "sair"}:
            break

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": 10},
        )
        last = result["messages"][-1]
        print(f"\nbot> {last.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
