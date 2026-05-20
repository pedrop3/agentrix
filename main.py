"""
Entry point. Sobe o MCP server via stdio, carrega as tools no LangChain,
constroi os sub-agents + supervisor, e abre um REPL no terminal.

Roda: python main.py
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.knower import build_knower
from agents.mathy import build_mathy
from agents.researcher import build_researcher
from agents.writer import build_writer
from config import config
from graph import build_graph, build_direct, build_supervisor
from logger import reset_steps

THREAD_ID = "default"


async def main():
    print(f"Carregando modelo Ollama: {config.ollama.model}")
    print("Subindo MCP server via stdio...")

    client = MultiServerMCPClient(
        {
            "switchboard": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "cwd": str(config.project_root),
            }
        }
    )
    tools = await client.get_tools()
    print(f"Tools carregadas do MCP: {[t.name for t in tools]}")

    researcher = build_researcher(tools, config.ollama.researcher_model)
    writer = build_writer(tools, config.ollama.writer_model)
    mathy = build_mathy(tools, config.ollama.mathy_model)
    knower = build_knower(tools, config.ollama.knower_model)
    supervisor = build_supervisor(config.ollama.supervisor_model)
    direct = build_direct(config.ollama.direct_model)

    async with AsyncSqliteSaver.from_conn_string(config.memory_db) as checkpointer:
        graph = build_graph(researcher, writer, mathy, supervisor, direct, knower, checkpointer)

        print("\nSwitchboard pronto. Digite 'quit' para sair.\n")
        print("Exemplos:")
        print("  - salve uma nota sobre minha reuniao com pontos X, Y, Z")
        print("  - o que eu salvei sobre reunioes?")
        print("  - quanto e 23 * 47 + 100?\n")

        run_config = {
            "configurable": {"thread_id": THREAD_ID},
            "recursion_limit": config.recursion_limit,
        }

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

            reset_steps()
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=run_config,
            )
            last = result["messages"][-1]
            print(f"\nbot> {last.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
