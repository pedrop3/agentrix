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
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from config import config
from logger import LLMLogger


def _build_prompt() -> str:
    """Gera o prompt do knower a partir de config.web (lido do .env)."""
    web = config.web
    return f"""
You are a knowledge agent answering questions about
{web.display_name} using public content from {web.domain}.

Context: {web.description}
Answer in {web.language}.

── HARD LIMIT ──────────────────────────────────────────────────────────────────
  Total tool calls this turn: AT MOST 6.
  After 6 calls you MUST answer with whatever you have gathered so far.
────────────────────────────────────────────────────────────────────────────────

Decision flow:
  1. For COMPARE / list requirements / fees / benefits / channels:
     → `kg_search_entity` ONCE. If hit → `kg_neighbors` → answer.
     (You may also run rag_search to enrich the answer.)

  2. For all other questions → `rag_search` with a short natural-language query.

  3. ══ STOP CONDITION ══
     If rag_search returned ANY chunks from {web.domain} that are topically
     related to the question → ANSWER IMMEDIATELY.
     Do NOT call web_search. Do NOT call rag_search again.
     The indexed content is the authoritative source — trust it.

  4. ONLY if rag_search returned 0 results OR every chunk is clearly about a
     DIFFERENT product (key identifier mismatch — see below):
     → `web_search` ONCE → pick best URL → `web_fetch(url)` ONCE.
     web_fetch returns the full page content — answer directly from it.
     Do NOT call rag_search again after web_fetch.

  5. Always answer in {web.language}, citing only the URLs that actually helped.
  6. NEVER invent fees, rates, dates, or conditions. If unknown, say so.

CRITICAL anti-loop rules:
  - NEVER call rag_search more than TWICE per turn.
  - NEVER call web_search more than ONCE per turn.
  - NEVER call the same tool with the same arguments twice.
  - If a tool returns empty/no-match, move immediately to the next step.

CRITICAL anti-contamination (key identifier matching):
  - Identify the KEY IDENTIFIER of the product asked about
    (e.g. "Select", "Jovem", "Teens", "Gold", "Black", "Platinum").
  - A chunk is relevant if it contains that identifier.
  - If NO chunk contains the key identifier → treat as miss → go to step 4.
  - If SOME chunks match → answer from those, ignore the off-topic ones.
  - NEVER use conversation history from prior turns to answer.

CRITICAL anti-hallucination:
  - A RAG miss doesn't mean the bank has no information — the right page may
    not be indexed yet. Use web_search before giving up.
  - Never cite a chunk that does not address the actual question.

Hard rules:
  - Only PUBLIC pages of {web.domain}. No authenticated areas.
  - Be conservative with `kg_extract` — it runs an LLM call per chunk.
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

    model = ChatGoogleGenerativeAI(
        model=config.gemini.model,
        google_api_key=config.gemini.api_key,
        temperature=config.gemini.temperature,
        callbacks=[LLMLogger("knower")],
    )

    # model = ChatOllama(
    #     model=config.ollama.knower_model,
    #     temperature=config.ollama.temperature,
    #     num_ctx=config.ollama.num_ctx,
    #     callbacks=[LLMLogger("knower")],
    # )

    my_tools = [t for t in tools if t.name in KNOWER_TOOLS]

    return create_react_agent(
        model,
        tools=my_tools,
        prompt=_build_prompt()
    )