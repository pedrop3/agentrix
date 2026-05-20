"""
MCP server que expoe tools: search_notes, save_note, calc,
web_fetcher, rag_search, rag_sources.

Roda standalone: python -m mcp_server.server
Ou e iniciado automaticamente pelo MultiServerMCPClient em main.py
(via transport stdio).
"""
import ast
import operator
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Correção do Ambiente: Garante o carregamento do .env dentro do MCP
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("switchboard-tools")

# Pasta onde as notas vivem (criada na primeira execucao)
NOTES_DIR = Path(__file__).parent / "notes"
NOTES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tool 1: buscar notas
# ---------------------------------------------------------------------------
@mcp.tool()
def search_notes(query: str) -> str:
    """Search saved notes by keyword in title or content.

    Returns matching notes with title and a content snippet.
    Use this to find information the user previously saved.
    """
    q = query.lower().strip()
    results = []
    for note in NOTES_DIR.glob("*.md"):
        content = note.read_text(encoding="utf-8")
        if q in content.lower() or q in note.stem.lower():
            snippet = content[:300] + ("..." if len(content) > 300 else "")
            results.append(f"## {note.stem}\n{snippet}")

    if not results:
        return f"No notes found matching '{query}'."
    return "\n\n---\n\n".join(results)


# ---------------------------------------------------------------------------
# Tool 2: salvar nota
# ---------------------------------------------------------------------------
@mcp.tool()
def save_note(title: str, content: str) -> str:
    """Save a note to disk with the given title and markdown content.

    Use this when the user wants to save, write, or create a new note.
    """
    safe = "".join(c for c in title if c.isalnum() or c in " -_").strip()
    if not safe:
        return "Error: title must contain at least one alphanumeric character."
    note_path = NOTES_DIR / f"{safe}.md"
    note_path.write_text(content, encoding="utf-8")
    return f"Note saved at {note_path.name} ({len(content)} chars)."


# ---------------------------------------------------------------------------
# Tool 3: calculadora segura (sem eval)
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@mcp.tool()
def calc(expression: str) -> str:
    """Evaluate a math expression safely.

    Supports +, -, *, /, **, %, //, and parentheses.
    Example: '23 * 47 + 100' returns '23 * 47 + 100 = 1181'.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

# ---------------------------------------------------------------------------
# Tool 4: busca web restrita ao dominio configurado em WEB_DOMAIN
# ---------------------------------------------------------------------------
@mcp.tool()
def web_search(query: str) -> str:
    """Search the configured web domain for public information.

    Strategy:
      1. Try the local RAG first (cached pages + uploads). If hits, return them.
      2. Otherwise, fall back to DuckDuckGo restricted to `site:<WEB_DOMAIN>`.
         Snippet summary is auto-indexed in the RAG for future hits.

    The domain is set by `WEB_DOMAIN` in `.env` (default: bank.pt).
    Returns a numbered list of {title, url, snippet}.
    """
    from config import config
    from rag import get_rag
    from web_fetcher import search_site

    domain = config.web.domain

    # 1) RAG-first
    try:
        rag = get_rag()
        local = rag.search(query, k=3)
        if local:
            lines = [f"# [Origem: RAG local] Resultados na base de conhecimento sobre {domain}:"]
            for i, h in enumerate(local, 1):
                snippet = h.text[:200].replace("\n", " ")
                lines.append(f"{i}. {h.source}\n   {snippet}...")
            return "\n\n".join(lines)
    except Exception as e:
        print(f"[web_search] RAG fallback: {e}")

    # 2) Web fallback (DDG + indexa snippets pro RAG)
    try:
        results = search_site(query, max_results=6)
    except Exception as e:
        return f"Error searching {domain}: {e}"

    if not results:
        return f"No results found on {domain} for '{query}'."

    try:
        rag = get_rag()
        summary = "\n".join(
            f"Result: {r.title} - {r.snippet} (URL: {r.url})" for r in results
        )
        rag.index_text(
            summary,
            source=f"search://{query}",
            kind="search_results",
            extra_metadata={"query": query, "domain": domain},
        )
    except Exception as e:
        print(f"[web_search] failed to index search results: {e}")

    lines = [f"# [Origem: DuckDuckGo] Resultados em {domain}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}\n   {r.url}\n   {r.snippet}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: fetch de uma URL no dominio configurado e indexa no RAG
# ---------------------------------------------------------------------------
@mcp.tool()
def web_fetch(url: str) -> str:
    """Fetch a page (HTML or PDF) inside the configured domain and return its
    clean text. The content is automatically indexed in the RAG so future
    questions hit the cache.

    The URL must belong to one of the allowed hosts (WEB_ALLOWED_HOSTS).
    """
    from config import config
    from rag import get_rag
    from web_fetcher import fetch_url

    try:
        page = fetch_url(url)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error fetching {url}: {e}"

    try:
        rag = get_rag()
        chunks = rag.index_text(
            page.text,
            source=page.url,
            kind="web",
            extra_metadata={
                "title": page.title,
                "type": page.kind,
                "domain": config.web.domain,
            },
        )
        indexing_note = f"(indexed {chunks} chunks in RAG, source={page.url})"
    except Exception as e:
        indexing_note = f"(RAG indexing failed: {e})"

    body = page.text if len(page.text) <= 6000 else page.text[:6000] + "\n…[truncado]"
    return f"# {page.title}\n{page.url} ({page.kind})\n{indexing_note}\n\n{body}"


# ---------------------------------------------------------------------------
# Tool 6: busca semantica no RAG (uploads + paginas ja fetchadas)
# ---------------------------------------------------------------------------
@mcp.tool()
def rag_search(query: str, k: int = 5) -> str:
    """Semantic search over indexed content (user uploads + bank pages).

    Returns the top-k chunks ranked by similarity. Use this BEFORE going to
    the web — if there's a confident hit, answer from it. If nothing
    relevant comes back, fall back to `web_search`.
    """
    from rag import get_rag

    try:
        rag = get_rag()
        hits = rag.search(query, k=k)
    except Exception as e:
        return f"Error querying RAG: {e}"
    if not hits:
        return f"No indexed content matches '{query}'."
    return "\n\n---\n\n".join(h.to_block() for h in hits)


# ---------------------------------------------------------------------------
# Tool 7: lista as fontes indexadas (debug / inspeção)
# ---------------------------------------------------------------------------
@mcp.tool()
def rag_sources() -> str:
    """List all unique sources currently indexed in the RAG store."""
    from rag import get_rag

    try:
        rag = get_rag()
        sources = rag.sources()
        total = rag.count()
    except Exception as e:
        return f"Error reading RAG: {e}"
    if not sources:
        return "RAG is empty."
    listing = "\n".join(f"- {s}" for s in sources)
    return f"{total} chunks total across {len(sources)} sources:\n{listing}"


# ---------------------------------------------------------------------------
# Tool 8: extrai entidades bancárias dos chunks pendentes (LLM -> Neo4j)
# ---------------------------------------------------------------------------
@mcp.tool()
def kg_extract(source: str = "", limit: int = 20) -> str:
    """Extract banking entities (Product/Fee/Requirement/Benefit/Channel/Customer)
    from indexed chunks that haven't been processed yet.

    - If `source` is given, only chunks of that source are processed.
    - Otherwise, processes any pending chunk (max `limit`).

    This is heavy (one LLM call per chunk). Use sparingly. Returns a summary
    with counts of nodes and relationships created.
    """
    from kg_extract import extract_for_source

    try:
        totals = extract_for_source(source=source or None, limit=limit)
    except Exception as e:
        return f"Error during extraction: {e}"
    return (
        f"Extracted from {totals['chunks']} chunks: "
        f"{totals['nodes']} nodes, {totals['rels']} relationships."
    )


# ---------------------------------------------------------------------------
# Tool 9: busca uma entidade por nome aproximado
# ---------------------------------------------------------------------------
@mcp.tool()
def kg_search_entity(name: str, limit: int = 10) -> str:
    """Find banking entities whose name matches the given substring (case-insensitive).

    Returns a list of {labels, name, sources_mentioning_it_count}.
    Use this to discover canonical product/fee/requirement names before
    asking for neighbors.
    """
    from rag import get_rag

    rag = get_rag()
    if not name.strip():
        return "Provide a name (or substring) to search."
    with rag.driver.session(database=rag.database) as s:
        res = s.run(
            """
            MATCH (e)
            WHERE any(l IN labels(e) WHERE l IN $allowed)
              AND toLower(e.name) CONTAINS toLower($name)
            OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
            WITH e, count(DISTINCT c) AS mentions
            RETURN labels(e) AS labels, e.name AS name, mentions
            ORDER BY mentions DESC LIMIT $limit
            """,
            name=name,
            allowed=["Product", "Fee", "Requirement", "Benefit", "Channel", "Customer"],
            limit=limit,
        )
        rows = [(r["labels"], r["name"], r["mentions"]) for r in res]
    if not rows:
        return f"No entities matching '{name}'."
    return "\n".join(f"- {':'.join(labels)} '{n}' (mentions: {m})" for labels, n, m in rows)


# ---------------------------------------------------------------------------
# Tool 10: vizinhos no grafo (1 hop) de uma entidade
# ---------------------------------------------------------------------------
@mcp.tool()
def kg_neighbors(name: str) -> str:
    """Return the 1-hop neighbors of an entity in the knowledge graph.

    Example output for name="Cartão Gold":
      Cartão Gold (:Product)
        -[HAS_FEE]->  Anuidade 35€ (:Fee)
        -[OFFERS]->   Seguro de viagem (:Benefit)
        -[REQUIRES]-> Rendimento 1000€ (:Requirement)
    """
    from rag import get_rag

    rag = get_rag()
    if not name.strip():
        return "Provide an entity name."
    with rag.driver.session(database=rag.database) as s:
        res = s.run(
            """
            MATCH (e {name: $name})
            OPTIONAL MATCH (e)-[r]->(n)
            WHERE NOT type(r) IN ['MENTIONS','CONTAINS']
            RETURN labels(e) AS src_labels, e.name AS src_name,
                   type(r) AS rel, labels(n) AS dst_labels, n.name AS dst_name
            """,
            name=name,
        )
        rows = list(res)
    if not rows:
        return f"No entity named '{name}' (or it has no outgoing relations)."
    header = f"{rows[0]['src_name']} (:{':'.join(rows[0]['src_labels'])})"
    lines = [header]
    for r in rows:
        if not r["rel"]:
            continue
        lines.append(
            f"  -[{r['rel']}]-> {r['dst_name']} (:{':'.join(r['dst_labels'])})"
        )
    return "\n".join(lines) if len(lines) > 1 else f"{header}\n  (no outgoing relations)"


if __name__ == "__main__":
    # Stdio transport: o cliente (langchain-mcp-adapters) sobe esse processo
    mcp.run(transport="stdio")