"""
Smoke test pra validar que o Ollama + LangChain conseguem fazer tool calling
ANTES de voce investir tempo construindo o resto do projeto.

Roda: python test_ollama.py
"""
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

load_dotenv()
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


def main():
    print(f"Testando tool calling com modelo: {MODEL}")
    model = ChatOllama(model=MODEL, temperature=0, num_ctx=8192)
    model_with_tools = model.bind_tools([multiply])

    response = model_with_tools.invoke(
        "What is 23 times 47? You must use the multiply tool."
    )

    print(f"\nConteudo da resposta: {response.content!r}")
    print(f"Tool calls: {response.tool_calls}")

    if response.tool_calls:
        print("\n[OK] Tool calling funcionou. Pode prosseguir.")
    else:
        print("\n[FALHOU] O modelo NAO chamou a tool.")
        print("Tente outro modelo: ollama pull llama3.1:8b")


if __name__ == "__main__":
    main()
