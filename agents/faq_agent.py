"""
FAQ agent: answers questions from the local RAG cache only.

No web access — fastest possible path. Called first for every single-product
domain question. Signals FAQ_MISS when the cache has nothing relevant so the
supervisor can re-route to web_agent without the LLM guessing.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import config
from logger import LLMLogger


def _build_prompt() -> str:
    web = config.web
    return f"""You are a fast FAQ agent for {web.display_name}.
Your ONLY tool is rag_search. You have NO access to the web.

Context: {web.description}
Answer in {web.language}.

── STRATEGY ────────────────────────────────────────────────────────────────────
1. Call rag_search ONCE with a focused natural-language query.
2. If the returned chunks directly address the question → answer concisely.
3. If results are empty, off-topic, or lack the key product identifier →
   respond with EXACTLY this format (one line, nothing else):
   FAQ_MISS: <brief reason>

   Examples of valid FAQ_MISS responses:
     FAQ_MISS: no chunks about Cartão Gold found in local cache
     FAQ_MISS: cached content is about Conta Corrente, not about the asked product
     FAQ_MISS: cache returned 0 results for this query
────────────────────────────────────────────────────────────────────────────────

Hard rules:
- Call rag_search AT MOST ONCE. Never twice.
- NEVER invent fees, rates, dates, or conditions.
- NEVER answer from memory when rag_search returned nothing — use FAQ_MISS.
- If in doubt, FAQ_MISS is always the right choice. A web agent handles fallback.
"""


FAQ_TOOLS = {"rag_search"}


def build_faq_agent(tools, model_name: str = None):
    model = ChatGoogleGenerativeAI(
        model=config.gemini.model,
        google_api_key=config.gemini.api_key,
        temperature=0.0,  # deterministic — cache lookups should be consistent
        callbacks=[LLMLogger("faq")],
    )
    my_tools = [t for t in tools if t.name in FAQ_TOOLS]
    return create_react_agent(model, tools=my_tools, prompt=_build_prompt())
