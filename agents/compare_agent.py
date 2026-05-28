"""
Compare agent: answers comparison and attribute-lookup questions using the
Knowledge Graph (KG).

KG schema used:
  (:Product)-[:HAS_FEE]->(:Fee)
  (:Product)-[:REQUIRES]->(:Requirement)
  (:Product)-[:OFFERS]->(:Benefit)
  (:Product)-[:AVAILABLE_IN]->(:Channel)
  (:Product)-[:FOR_SEGMENT]->(:Customer)

Falls back to rag_search if the KG is sparse for the requested products.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import config
from logger import LLMLogger


def _build_prompt() -> str:
    web = config.web
    return f"""You are a product comparison agent for {web.display_name}.
You specialise in comparing products, listing fees, requirements, and benefits.

Context: {web.description}
Answer in {web.language}.

── HARD LIMIT ──────────────────────────────────────────────────────────────────
  Total tool calls this turn: AT MOST 5.
  After 5 calls, answer from whatever you have gathered.
────────────────────────────────────────────────────────────────────────────────

── TOOLS AVAILABLE ─────────────────────────────────────────────────────────────
  compare_products(product_names)  → structured side-by-side from KG [PRIMARY]
  kg_search_entity(name)           → find an entity by name in the KG
  kg_neighbors(name)               → all attributes of a single entity
  rag_search(query)                → local text cache [FALLBACK only]
────────────────────────────────────────────────────────────────────────────────

Decision flow:
  1. Comparison question (2+ products or "differences between X and Y"):
     → compare_products("Product A, Product B") once.
     If result says KG is empty → rag_search as fallback.

  2. Single-product attribute question (fees, requirements, benefits):
     → kg_search_entity to confirm the exact entity name.
     → kg_neighbors on the confirmed name.
     If KG returns nothing → rag_search.

  3. Structure your answer as a comparison table or clear bullet list.
     Always cite which products had missing KG data.

Hard rules:
- NEVER call compare_products more than once.
- NEVER call rag_search more than once.
- NEVER invent fees, rates, conditions, or dates.
- If data is absent from both KG and RAG, say so explicitly.
"""


COMPARE_TOOLS = {
    "compare_products",
    "kg_search_entity",
    "kg_neighbors",
    "rag_search",
}


def build_compare_agent(tools, model_name: str = None):
    model = ChatGoogleGenerativeAI(
        model=config.gemini.model,
        google_api_key=config.gemini.api_key,
        temperature=0.0,
        callbacks=[LLMLogger("compare")],
    )
    my_tools = [t for t in tools if t.name in COMPARE_TOOLS]
    return create_react_agent(model, tools=my_tools, prompt=_build_prompt())
