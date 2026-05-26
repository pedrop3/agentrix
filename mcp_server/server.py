"""
MCP server que expoe tools: web_search, web_fetch, rag_search, rag_sources,
rag_stats, kg_extract, kg_search_entity, kg_neighbors.

Roda standalone: python -m mcp_server.server
Ou e iniciado automaticamente pelo MultiServerMCPClient em server.py
(via transport stdio).
"""
import logging
import sys
import threading
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ─── Suprimir logs verbosos do protocolo MCP (stdio) ─────────────────────────
# O FastMCP loga "Processing request of type …" em INFO por defeito —
# polui o output do agente intercalado com os steps numerados.
for _mcp_logger in ("mcp", "mcp.server", "mcp.server.fastmcp",
                    "mcp.server.session", "mcp.shared"):
    logging.getLogger(_mcp_logger).setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("switchboard-tools")

NOTES_DIR = Path(__file__).parent / "notes"
NOTES_DIR.mkdir(exist_ok=True)

# ─── Circuit breaker: web_search ─────────────────────────────────────────────
# Se a rede externa estiver inacessível, bloqueia novas tentativas por
# _WEB_BACKOFF_SECS para não desperdiçar tempo com timeouts repetidos.
_web_unavailable_until: float = 0.0
_WEB_BACKOFF_SECS: float = 120.0   # 2 minutos entre retentativas

# ─── Background KG extraction ─────────────────────────────────────────────────
# Lock não-bloqueante: se já houver extracção a correr, a nova é saltada
# em vez de ficar em fila (evita saturar o Ollama com pedidos paralelos).
_kg_extract_lock = threading.Lock()


def _bg_kg_extract(source: str) -> None:
    """Extrai entidades para o grafo em background (thread daemon).

    Corre após cada web_fetch bem-sucedido para manter o knowledge graph
    actualizado sem bloquear a resposta ao utilizador.
    """
    if not _kg_extract_lock.acquire(blocking=False):
        print(f"[kg] ⬡ skipped (busy): {source[:60]}")
        return
    try:
        from kg_extract import extract_for_source
        totals = extract_for_source(source=source, limit=20)
        if totals["nodes"] or totals["rels"]:
            print(
                f"[kg] ⬡ {source[:55]}"
                f"  → {totals['nodes']} nodes  {totals['rels']} rels"
                f"  ({totals['chunks']} chunks)"
            )
    except Exception as e:
        print(f"[kg] ⬡ extract error ({source[:40]}): {e}")
    finally:
        _kg_extract_lock.release()


# ─── Lexical-overlap guard (shared by rag_search) ─────────────────────────────
_STOPWORDS_LEX = {
    # PT — function words
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "para", "pra", "por", "com", "sem", "sobre", "ate",
    "é", "e", "ou", "que", "qual", "quais", "como", "onde", "quando", "porque", "porquê",
    "mais", "menos", "muito", "muita", "muitos", "muitas", "ser", "estar", "ter", "tem",
    "são", "seu", "sua", "seus", "suas", "meu", "minha", "voce", "vocês", "este",
    "esta", "esse", "essa", "isto", "isso", "aquele", "aquela", "aqui", "ali", "lá",
    # EN — function words
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from", "and",
    "or", "but", "is", "are", "was", "were", "be", "been", "this", "that", "what",
    "which", "how", "why", "when", "where", "who", "i", "you", "he", "she", "it", "we",
    "they",
    # Domain-generic banking / Portuguese terms — appear on every page and
    # therefore carry no discriminating signal between different products.
    # Without these, queries like "conta select criterios" share the token
    # "conta" with ANY indexed page, causing false-positive RAG matches.
    "conta", "banco", "abrir", "abertura", "titular", "deposito", "montante",
    "cliente", "clientes", "produto", "produtos", "servico", "servicos",
    "informacao", "informacoes", "documentos", "documentacao", "ficha",
    "criterios", "requisitos", "condicoes", "condicao", "taxas", "custos",
    "santander", "portugal", "bancario", "bancaria",
}


def _strip_accents(s: str) -> str:
    """Remove diacritics: 'cartão' → 'cartao', 'crédito' → 'credito'."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


def _lexical_overlap_ok(query: str, text: str, min_overlap: int = 1) -> bool:
    """True if at least `min_overlap` significant query terms appear in text.

    Accent-insensitive: 'cartao' matches 'cartão', 'credito' matches 'crédito'.
    Users often type without accents; chunks always have them — without this
    normalisation every query without accents produces a false miss.
    """
    import re

    def tokens(s: str) -> set[str]:
        ws = re.findall(r"\w+", _strip_accents(s).lower())
        return {w for w in ws if len(w) >= 4 and w not in _STOPWORDS_LEX}

    q = tokens(query)
    if not q:
        return True  # query is all stopwords — disable the filter
    return len(q & tokens(text)) >= min_overlap


# ─── web_search ───────────────────────────────────────────────────────────────
@mcp.tool(
    title="Web Search (domain-restricted)",
    description=(
        "Search the configured web domain via DuckDuckGo (site:<domain>). "
        "Call ONLY when rag_search returns no relevant results — web_search is "
        "slower and has external side effects. "
        "Results are auto-indexed in RAG so the same query is answered locally next time. "
        "Returns: numbered list of {title, URL, snippet}. "
        "If a result looks relevant, follow up with web_fetch to get the full content."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,    # writes snippets to RAG
        destructiveHint=False, # only additive (new RAG chunks)
        idempotentHint=False,  # DDG results can vary over time
        openWorldHint=True,    # calls external DuckDuckGo
    ),
)
def web_search(query: str) -> str:
    global _web_unavailable_until

    from config import config
    from rag import get_rag
    from web_fetcher import search_site

    domain = config.web.domain

    # Circuit breaker: se houve erro de rede recente, não tenta de novo
    if time.time() < _web_unavailable_until:
        remaining = int(_web_unavailable_until - time.time())
        return (
            f"External search is currently unavailable (network error; retry in ~{remaining}s). "
            f"If you know a relevant URL on {domain}, call web_fetch directly."
        )

    print(f"[web_search] querying DDG site:{domain!r} for: {query!r}")

    try:
        results = search_site(query, max_results=6)
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("unreachable", "errno 51", "timed out", "timeout", "connect")):
            _web_unavailable_until = time.time() + _WEB_BACKOFF_SECS
            print(f"[web_search] network unavailable — circuit open for {_WEB_BACKOFF_SECS:.0f}s")
        return f"Error searching {domain}: {e}"

    if not results:
        return (
            f"No results found on {domain} for '{query}'. "
            f"If you know a relevant URL on {domain}, call `web_fetch(url)` directly."
        )

    print(f"[web_search] found {len(results)} results.")

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


# ─── web_fetch ────────────────────────────────────────────────────────────────
@mcp.tool(
    title="Web Fetch (full page)",
    description=(
        "Download the full text of a specific URL (HTML or PDF) from the configured domain. "
        "Use after web_search reveals a relevant URL, or when you already know the URL. "
        "Limit to 2 fetches per question — each fetch is a network call. "
        "Content is automatically indexed in RAG so future questions hit the local cache. "
        "Returns: page title + full text (up to 6000 chars). "
        "After fetching, answer from the returned content — do NOT call rag_search again."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,   # writes page content to RAG
        destructiveHint=False,
        idempotentHint=True,  # fetching the same URL twice is safe
        openWorldHint=True,   # makes an outbound HTTP request
    ),
)
def web_fetch(url: str) -> str:
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
        # ── Background KG extraction ────────────────────────────────────────
        # Corre em daemon thread para não atrasar a resposta ao utilizador.
        # O knowledge graph fica actualizado para perguntas futuras.
        threading.Thread(
            target=_bg_kg_extract, args=(page.url,), daemon=True
        ).start()
    except Exception as e:
        indexing_note = f"(RAG indexing failed: {e})"

    body = page.text if len(page.text) <= 6000 else page.text[:6000] + "\n…[truncado]"
    return f"# {page.title}\n{page.url} ({page.kind})\n{indexing_note}\n\n{body}"


# ─── rag_search ───────────────────────────────────────────────────────────────
@mcp.tool(
    title="RAG Search (local cache)",
    description=(
        "Hybrid search (vector + fulltext) over locally cached content "
        "(user uploads + previously fetched pages). "
        "ALWAYS call this FIRST before going to the web — it is fast, free, and offline. "
        "Query must be a short natural-language question or phrase — "
        "do NOT use DDG-style operators (site:, OR, AND, quotes). "
        "Uses vector similarity + Lucene fulltext fused with RRF, then a domain-aware "
        "lexical guard to reject off-topic chunks. "
        "If the returned chunks clearly answer the question, STOP and use them — "
        "do not escalate to web_search. "
        "Only fall through to web_search if results are empty or every chunk is "
        "clearly about a different product (key identifier absent)."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def rag_search(query: str, k: int = 7) -> str:
    from rag import get_rag

    try:
        rag = get_rag()
        # hybrid_search fuses vector cosine + Lucene fulltext via RRF, then keeps
        # only chunks that appear in the fulltext results OR have a very strong
        # vector score (>= 0.80). This eliminates pure-vector false positives
        # without discarding chunks that are lexically on-topic.
        raw_hits = rag.hybrid_search(
            query,
            k=k * 2,
            require_lexical=True,
            min_vector_score=0.5,           # pre-filter for the vector arm of RRF
            exclude_kinds=["search_results"],  # snippets DDG são ruído — só conteúdo real
        )
    except Exception as e:
        return f"Error querying RAG: {e}"

    if not raw_hits:
        return f"No indexed content matches '{query}'."

    # Secondary domain-aware guard: removes chunks that share no significant
    # product-specific term with the query (e.g. rejects "Conta Jovem" content
    # when the query is about "Conta Select", since after stripping generic
    # banking stopwords the only discriminating token is the product identifier).
    filtered = [h for h in raw_hits if _lexical_overlap_ok(query, h.text, min_overlap=1)]
    discarded = len(raw_hits) - len(filtered)

    if not filtered:
        return (
            f"No indexed content matches '{query}' "
            f"({len(raw_hits)} candidate(s) found but none shared the key product "
            f"identifier with the query). Try `web_search` instead."
        )

    returned = filtered[:k]
    extra = len(filtered) - len(returned)
    extra_note = f", {extra} more omitted" if extra else ""
    head = f"# {len(returned)} hit(s) (discarded {discarded} off-topic{extra_note}):"
    body = "\n\n---\n\n".join(h.to_block() for h in returned)
    return f"{head}\n\n{body}"


# ─── rag_sources ──────────────────────────────────────────────────────────────
@mcp.tool(
    title="RAG Sources (debug)",
    description=(
        "List all unique source URLs currently indexed in the local RAG store. "
        "Use for debugging to check what has already been fetched/uploaded. "
        "NOT intended for answering user questions."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def rag_sources() -> str:
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


# ─── rag_stats ────────────────────────────────────────────────────────────────
@mcp.tool(
    title="RAG Stats (debug)",
    description=(
        "Diagnostic: chunk counts by kind, total sources, entity and mention counts. "
        "Use when the agent's answers look wrong or to verify the knowledge graph was built. "
        "NOT intended for answering user questions."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def rag_stats() -> str:
    from rag import get_rag

    try:
        rag = get_rag()
        s = rag.stats()
    except Exception as e:
        return f"Error reading RAG stats: {e}"
    by_kind = "\n".join(f"    {k}: {v}" for k, v in s["chunks_by_kind"].items()) or "    (none)"
    return (
        f"chunks_total: {s['chunks_total']}\n"
        f"chunks_by_kind:\n{by_kind}\n"
        f"sources: {s['sources']}\n"
        f"entities: {s['entities']}\n"
        f"mentions: {s['mentions']}"
    )


# ─── kg_extract ───────────────────────────────────────────────────────────────
@mcp.tool(
    title="Knowledge Graph Extract (expensive)",
    description=(
        "EXPENSIVE: runs one LLM call per chunk to extract entities "
        "(Product/Fee/Requirement/Benefit/Channel/Customer) into the knowledge graph. "
        "NOTE: extraction runs automatically in the background after every web_fetch — "
        "you usually do NOT need to call this manually. "
        "Only call explicitly when kg_search_entity returns nothing for a source that "
        "was indexed before background extraction was active. "
        "Already-processed chunks are skipped, so re-running the same source is safe. "
        "Returns: counts of nodes and relationships created."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,    # writes to the knowledge graph
        destructiveHint=False, # additive only
        idempotentHint=True,   # re-processing same chunks is a no-op
        openWorldHint=False,
    ),
)
def kg_extract(source: str = "", limit: int = 20) -> str:
    from kg_extract import extract_for_source

    try:
        totals = extract_for_source(source=source or None, limit=limit)
    except Exception as e:
        return f"Error during extraction: {e}"
    return (
        f"Extracted from {totals['chunks']} chunks: "
        f"{totals['nodes']} nodes, {totals['rels']} relationships."
    )


# ─── kg_search_entity ─────────────────────────────────────────────────────────
@mcp.tool(
    title="Knowledge Graph — Find Entity",
    description=(
        "Find entities in the knowledge graph by name substring (case-insensitive). "
        "Best first step when the question is about comparing products, listing "
        "requirements, or looking up fees/benefits for a named product. "
        "Returns entity type labels and mention counts. "
        "If a match is found, call kg_neighbors on the exact name to get its attributes. "
        "If empty, skip to rag_search — the KG may not have been populated yet."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def kg_search_entity(name: str, limit: int = 10) -> str:
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


# ─── kg_neighbors ─────────────────────────────────────────────────────────────
@mcp.tool(
    title="Knowledge Graph — Entity Neighbors",
    description=(
        "Get all outgoing relationships of a named entity (1-hop neighbors). "
        "Call after kg_search_entity confirms the exact entity name. "
        "Returns linked fees, requirements, benefits, and channels as labeled triples — "
        "ideal for questions about what a product costs, requires, or offers. "
        "If the entity has no neighbors, the KG is sparse — fall back to rag_search."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def kg_neighbors(name: str) -> str:
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
    mcp.run(transport="stdio")
