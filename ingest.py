"""
Script de ingestão: percorre o sitemap de WEB_DOMAIN, indexa páginas no RAG
(Neo4j vector + fulltext) e extrai entidades para o Knowledge Graph.

Pré-requisitos:
  - Neo4j a correr (docker-compose up)
  - Ollama a correr com nomic-embed-text (para embeddings)
  - Ollama com o modelo de extracção (para --kg; ver OLLAMA_MODEL_KG_EXTRACT no .env)

Uso:
  python ingest.py                     # indexa até 100 páginas novas
  python ingest.py --limit 50          # máximo 50 páginas novas
  python ingest.py --no-kg             # só RAG, sem extracção de KG
  python ingest.py --check             # mostra páginas pendentes sem indexar
  python ingest.py --url <url>         # força (re-)indexar uma URL específica
  python ingest.py --reindex           # re-indexa tudo (mesmo já indexado)
  python ingest.py --delay 2.0         # pausa entre pedidos HTTP (default: 1s)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config import config
from rag import get_rag
from web_fetcher import fetch_url, get_sitemap_urls

W = 72  # largura da linha de output

# ─── Helpers de display ───────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def _short(s: str, n: int = 55) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


# ─── Core ─────────────────────────────────────────────────────────────────────

def ingest_url(
    rag,
    url: str,
    *,
    run_kg: bool = True,
    label: str = "",
) -> dict | None:
    """Fetcha e indexa uma URL. Devolve stats ou None em caso de erro."""
    try:
        page = fetch_url(url)
    except Exception as e:
        print(f"  {label}FAIL  {_short(url, 65)}  ({e})")
        return None

    kind = "pdf" if page.kind == "pdf" else "web"
    try:
        chunks = rag.index_text(
            page.text,
            source=page.url,
            kind=kind,
            extra_metadata={
                "title": page.title,
                "type": page.kind,
                "domain": config.web.domain,
            },
        )
    except Exception as e:
        print(f"  {label}RAG FAIL  {_short(url, 60)}  ({e})")
        return None

    result: dict = {"chunks": chunks, "nodes": 0, "rels": 0}

    if run_kg and chunks > 0:
        try:
            from kg_extract import extract_for_source
            kg = extract_for_source(source=page.url, limit=50)
            result.update({"nodes": kg["nodes"], "rels": kg["rels"]})
            kg_info = f"  KG:{kg['nodes']}n/{kg['rels']}r"
        except Exception as e:
            kg_info = f"  KG:err"
    else:
        kg_info = ""

    title_str = f"{_short(page.title or url, 42)!r:44s}"
    print(f"  {label}OK  {chunks:3d} chunks  {title_str}{kg_info}")
    return result


def run_check(rag) -> None:
    """Mostra estado actual: indexado vs pendente."""
    print(f"Reading sitemap: {config.web.base_url}/sitemap.xml …")
    urls = get_sitemap_urls(limit=500)
    if not urls:
        print("No URLs found. Check WEB_BASE_URL and network.")
        return

    indexed = set(rag.sources())
    pending = [u for u in urls if u not in indexed]

    sep = "─" * W
    print(f"\n{sep}")
    print(f"Sitemap : {len(urls):4d} URLs")
    print(f"Indexed : {len(indexed):4d} sources in Neo4j")
    print(f"Pending : {len(pending):4d} pages")
    print(sep)

    if pending:
        print("\nPending pages:")
        for u in pending:
            print(f"  {u}")
    else:
        print("\nAll sitemap pages are already indexed.")

    # Quick stats
    stats = rag.stats()
    print(f"\n{sep}")
    print(f"chunks_total : {stats['chunks_total']}")
    by_kind = "  ".join(f"{k}={v}" for k, v in stats["chunks_by_kind"].items())
    print(f"by_kind      : {by_kind}")
    print(f"entities     : {stats['entities']}")
    print(f"mentions     : {stats['mentions']}")
    print(sep)


def run_ingest(
    rag,
    *,
    limit: int = 100,
    run_kg: bool = True,
    reindex: bool = False,
    delay: float = 1.0,
) -> None:
    print(f"Reading sitemap: {config.web.base_url}/sitemap.xml …")
    urls = get_sitemap_urls(limit=500)
    if not urls:
        print("No URLs found in sitemap. Check WEB_BASE_URL and network.")
        sys.exit(1)
    print(f"Found {len(urls)} URLs in sitemap.")

    if reindex:
        pending = urls
        print(f"Re-indexing all {len(pending)} URLs (--reindex).")
    else:
        indexed = set(rag.sources())
        pending = [u for u in urls if u not in indexed]
        print(f"Already indexed: {len(indexed)}  |  Pending: {len(pending)}")

    pending = pending[:limit]
    if not pending:
        print("Nothing to index.")
        return

    kg_note = "RAG + KG" if run_kg else "RAG only"
    print(f"\nIngesting {len(pending)} pages ({kg_note}) …\n")

    ok = fail = total_nodes = total_rels = total_chunks = 0

    for i, url in enumerate(pending, 1):
        label = f"[{i:3d}/{len(pending)}] "
        result = ingest_url(rag, url, run_kg=run_kg, label=label)
        if result:
            ok += 1
            total_chunks += result["chunks"]
            total_nodes += result["nodes"]
            total_rels += result["rels"]
        else:
            fail += 1

        # Progress bar a cada 10 URLs
        if i % 10 == 0 or i == len(pending):
            print(f"\n  {_bar(i, len(pending))}  ok={ok} fail={fail}\n")

        if i < len(pending):
            time.sleep(delay)

    sep = "─" * W
    print(f"\n{sep}")
    print(f"Done.  {ok} indexed  |  {fail} failed  |  {total_chunks} chunks")
    if run_kg:
        print(f"KG:    {total_nodes} nodes  |  {total_rels} relationships")
    print(sep)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest sitemap pages into RAG + Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit",   type=int,   default=100,  help="Max new pages to ingest (default: 100)")
    parser.add_argument("--no-kg",   action="store_true",      help="Skip KG extraction (RAG only)")
    parser.add_argument("--check",   action="store_true",      help="Dry run: show pending pages without indexing")
    parser.add_argument("--url",                               help="Force re-ingest a specific URL")
    parser.add_argument("--reindex", action="store_true",      help="Re-ingest already indexed pages too")
    parser.add_argument("--delay",   type=float, default=1.0,  help="Delay between HTTP requests in seconds (default: 1.0)")
    args = parser.parse_args()

    rag = get_rag()

    if args.url:
        print(f"Ingesting single URL: {args.url}")
        result = ingest_url(rag, args.url, run_kg=not args.no_kg)
        if result:
            print(f"Done. {result['chunks']} chunks indexed.")
        return

    if args.check:
        run_check(rag)
        return

    run_ingest(
        rag,
        limit=args.limit,
        run_kg=not args.no_kg,
        reindex=args.reindex,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
