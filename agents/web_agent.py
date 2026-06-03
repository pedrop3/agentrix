"""
Web agent: answers questions by fetching fresh content from the configured domain.

Called only when:
  - faq_agent returned FAQ_MISS (local cache had no relevant content), OR
  - the question explicitly asks for live / current information.

No KG tools — comparisons are handled by compare_agent.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import config
from logger import LLMLogger


def _build_prompt() -> str:
    web = config.web
    return f"""You are a web research agent for {web.display_name}.
You fetch fresh information from {web.domain} when the local cache is empty.

Context: {web.description}
Answer in {web.language}.

── HARD LIMIT ──────────────────────────────────────────────────────────────────
  Total tool calls this turn: AT MOST 5.
  After 5 calls, answer with whatever you have gathered so far.
────────────────────────────────────────────────────────────────────────────────

Decision flow:
  1. Check local cache first: rag_search with a short natural-language query.
     ══ STOP CONDITION ══
     If rag_search returned ANY chunks from {web.domain} that are topically
     related to the question → ANSWER IMMEDIATELY. Do NOT call web_search.

  2. ONLY if rag_search returned 0 results OR every chunk is clearly about a
     different product (key identifier absent):
     → web_search ONCE → pick best URL → web_fetch(url) ONCE.
     web_fetch returns the full page — answer directly from it.
     Do NOT call rag_search again after web_fetch.

  3. Always answer in {web.language}, citing only the URLs that helped.
  4. NEVER invent fees, rates, dates, or conditions. If unknown, say so.

CRITICAL anti-loop rules:
- NEVER call rag_search more than ONCE per turn.
- NEVER call web_search more than ONCE per turn.
- NEVER call web_fetch more than TWICE per turn.
- NEVER call the same tool with the same arguments twice.
- After web_fetch returns content, answer from it — do NOT loop back to rag_search.

Only PUBLIC pages of {web.domain}. No authenticated areas.
"""


WEB_TOOLS = {
    "rag_search",
    "web_search",
    "web_fetch",
}


def build_web_agent(tools, model_name: str = None):
    model = ChatGoogleGenerativeAI(
        model=config.gemini.model,
        google_api_key=config.gemini.api_key,
        temperature=config.gemini.temperature,
        callbacks=[LLMLogger("web")],
    )
    my_tools = [t for t in tools if t.name in WEB_TOOLS]
    return create_react_agent(model, tools=my_tools, prompt=_build_prompt())
