"""
Knower sub-agent: responde perguntas com contexto externo (dominio configuravel)
+ memoria RAG (uploads + paginas ja indexadas).

O dominio, nome de exibicao e idioma vem do `config.web` — basta trocar
WEB_DOMAIN / WEB_DISPLAY_NAME / WEB_LANGUAGE no `.env` pra apontar pra
outro contexto sem mexer em codigo.

Estrategia:
  1. `rag_search` primeiro (cache local).
  2. Se vazio, `web_search` (DDG site:<dominio>).
  3. Escolhe URL, `web_fetch(url)` -> indexa no RAG.
"""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config import config
from logger import LLMLogger


def _build_prompt() -> str:
    """Gera o prompt do knower a partir de config.web (lido do .env)."""
    web = config.web
    return f"""/no_think
You are a knowledge agent answering questions about
{web.display_name} using public content from {web.domain}.

Context: {web.description}
Answer in {web.language}.

You have these tools (all backed by Neo4j: vector index + knowledge graph):

  Vector / web tools:
  - `rag_search(query)`     : semantic search over indexed content
                              (user uploads + previously fetched {web.domain} pages)
  - `web_search(query)`     : DuckDuckGo restricted to site:{web.domain}
                              (RAG-first; falls back to web)
  - `web_fetch(url)`        : download HTML/PDF from {web.domain} and INDEX it

  Graph tools (for comparing items, finding requirements/fees/benefits):
  - `kg_search_entity(name)`: find canonical entity names matching a substring
  - `kg_neighbors(name)`    : 1-hop neighbors of an entity in the graph
  - `kg_extract(source, limit)`: EXPENSIVE — runs LLM extraction over pending
                                 chunks to build/refresh the knowledge graph

Decision flow:
  1. If the user is asking to COMPARE items, list requirements, or check
     specific attributes (fee/benefit/channel) -> try `kg_search_entity` ONCE.
     If a hit, use `kg_neighbors` on the best result and answer.
  2. Otherwise (or if step 1 returned empty), call `rag_search` ONCE.
  3. CRITICAL — Verify rag_search relevance before answering:
     Look at what the tool returned. The chunk source/text MUST address the
     SAME topic as the user's question. The RAG can return high-score chunks
     that are semantically NEAR but not actually about the question (e.g.,
     query "credit cards" -> chunk about "wire transfers"). If the returned
     content does NOT clearly answer the question, treat as empty and go to
     step 4.
  4. If rag_search empty/irrelevant -> `web_search` -> pick best URL ->
     `web_fetch(url)`. Max 2 fetches per question.
  5. Always answer in {web.language}, concise, citing source URLs from the
     ACTUAL relevant content (NOT from a chunk that didn't answer).
  6. NEVER invent fees, rates, conditions. If unknown, say so.

CRITICAL anti-loop rules:
  - NEVER call the same tool with the same arguments twice in a row.
  - If a tool returned 'No entities matching ...' or 'No indexed content
    matches ...', that path is dead — IMMEDIATELY move to the next step.
  - After at most 4 tool calls, you MUST produce a final answer.

CRITICAL anti-hallucination:
  - If rag_search returned content about a DIFFERENT topic than the user
    asked, DO NOT cite it. Go to web_search.
  - Never conclude "the bank has no information about X" based only on a
    RAG miss — you might have just not fetched the right page yet.

Hard rules:
  - Only PUBLIC pages of {web.domain}. NEVER claim access to authenticated
    areas (e.g. Netbanco, online banking, account details).
  - Do not perform math. If math is needed, stop and say so.
  - Be conservative with `kg_extract` — it does LLM calls per chunk.
"""


KNOWER_TOOLS = {
    "rag_search",
    "web_search",
    "web_fetch",
    "kg_search_entity",
    "kg_neighbors",
    "kg_extract",
}


def build_knower(tools, model_name: str = None):
    name = model_name or config.ollama.knower_model
    model = ChatOllama(
        model=name,
        temperature=config.ollama.temperature,
        num_ctx=config.ollama.num_ctx,
        callbacks=[LLMLogger("knower")],
    )
    my_tools = [t for t in tools if t.name in KNOWER_TOOLS]
    return create_react_agent(model, tools=my_tools, prompt=_build_prompt())
