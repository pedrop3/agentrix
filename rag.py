"""
RAG sobre Neo4j

- Embeddings: Ollama (`nomic-embed-text`, 768 dim)
- Vector index nativo do Neo4j 5 (`db.index.vector.queryNodes`)
- Knowledge graph na mesma instância: nós :Chunk + :Source + entidades
  bancárias (:Product / :Fee / :Requirement / :Benefit / :Channel / :Customer)

Esquema Cypher (criado em get_rag().init_schema()):
    (:Source {url})-[:CONTAINS]->(:Chunk {id, text, kind, source, embedding})
    (:Chunk)-[:MENTIONS]->(:Entity)
    (:Product)-[:HAS_FEE|REQUIRES|OFFERS|AVAILABLE_IN|FOR_SEGMENT]->...
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Optional

from config import config

VECTOR_INDEX = config.neo4j.vector_index

_singleton_lock = threading.Lock()
_singleton: Optional["RAG"] = None


@dataclass
class Hit:
    text: str
    source: str
    kind: str
    score: float

    def to_block(self) -> str:
        return f"[{self.kind} · {self.source} · score={self.score:.3f}]\n{self.text}"


def _chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _silence_neo4j_notifications() -> dict:
    """Tenta desabilitar notifications informacionais do Neo4j no driver."""
    import logging
    import warnings

    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"neo4j(\..*)?")

    out: dict = {}
    try:
        from neo4j import (
            NotificationMinimumSeverity,
            NotificationDisabledClassification,
        )

        out["notifications_min_severity"] = NotificationMinimumSeverity.WARNING
        out["notifications_disabled_classifications"] = [
            NotificationDisabledClassification.UNRECOGNIZED,
            NotificationDisabledClassification.SCHEMA,
        ]
        return out
    except ImportError:
        pass
    try:
        from neo4j import (
            NotificationMinimumSeverity,
            NotificationDisabledCategory,
        )

        out["notifications_min_severity"] = NotificationMinimumSeverity.WARNING
        out["notifications_disabled_categories"] = [
            NotificationDisabledCategory.UNRECOGNIZED,
            NotificationDisabledCategory.SCHEMA,
        ]
        return out
    except ImportError:
        return out


class RAG:
    def __init__(self):
        from neo4j import GraphDatabase
        from langchain_ollama import OllamaEmbeddings

        driver_kwargs: dict = {"auth": (config.neo4j.user, config.neo4j.password)}
        driver_kwargs.update(_silence_neo4j_notifications())

        self._driver = GraphDatabase.driver(config.neo4j.uri, **driver_kwargs)
        self._embeddings = OllamaEmbeddings(model=config.ollama.embed_model)
        self._db = config.neo4j.database
        self.init_schema()

    # -----------------------------------------------------------------------
    # Schema (constraints + vector index)
    # -----------------------------------------------------------------------
    def init_schema(self) -> None:
        with self._driver.session(database=self._db) as s:
            s.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT source_url IF NOT EXISTS "
                "FOR (s:Source) REQUIRE s.url IS UNIQUE"
            )
            for label in ("Product", "Fee", "Requirement", "Benefit", "Channel", "Customer"):
                s.run(
                    f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                )
            s.run(
                f"""
                CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
                FOR (c:Chunk) ON c.embedding
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {config.ollama.embed_dim},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """
            )

    # -----------------------------------------------------------------------
    # Indexação
    # -----------------------------------------------------------------------
    def index_text(
        self,
        text: str,
        *,
        source: str,
        kind: str = "doc",
        extra_metadata: Optional[dict] = None,
    ) -> int:
        chunks = _chunk(text)
        if not chunks:
            return 0
        embeddings = self._embeddings.embed_documents(chunks)
        meta = extra_metadata or {}
        title = meta.get("title") or source

        params = {
            "source": source,
            "title": title,
            "kind": kind,
            "rows": [
                {
                    "id": f"{source}::{i}",
                    "text": chunks[i],
                    "embedding": embeddings[i],
                    "chunk_index": i,
                    # Mete chaves do metadata como propriedades do nó (Neo4j nao
                    # aceita Map aninhado direto; precisa "achatar")
                    "metadata": {k: v for k, v in meta.items() if k != "title"},
                }
                for i in range(len(chunks))
            ],
        }
        with self._driver.session(database=self._db) as s:
            s.run(
                """
                MERGE (src:Source {url: $source})
                ON CREATE SET src.title = $title, src.kind = $kind, src.indexed_at = datetime()
                ON MATCH  SET src.title = $title, src.kind = $kind, src.updated_at = datetime()
                WITH src
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c.text        = row.text,
                    c.embedding   = row.embedding,
                    c.kind        = $kind,
                    c.source      = $source,
                    c.chunk_index = row.chunk_index
                SET c += row.metadata
                MERGE (src)-[:CONTAINS]->(c)
                """,
                params,
            )
        return len(chunks)

    # -----------------------------------------------------------------------
    # Busca vetorial
    # -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        kind: Optional[str] = None,
        min_score: float = 0.0,
        exclude_kinds: Optional[Iterable[str]] = None,
    ) -> list[Hit]:
        """Top-K chunks por similaridade coseno, com filtros opcionais.

        - `kind`           : retorna so chunks desse kind
        - `min_score`      : descarta hits com score < min_score (cosine, 0..1)
        - `exclude_kinds`  : descarta hits cujo kind esteja nesse set
        """
        if not query.strip():
            return []
        query_emb = self._embeddings.embed_query(query)
        excluded = list(exclude_kinds) if exclude_kinds else []
        with self._driver.session(database=self._db) as s:
            res = s.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k2, $emb)
                YIELD node, score
                WITH node AS c, score
                WHERE ($kind IS NULL OR c.kind = $kind)
                  AND (size($excluded) = 0 OR NOT c.kind IN $excluded)
                  AND score >= $min_score
                RETURN c.text AS text,
                       c.source AS source,
                       c.kind AS kind,
                       score
                ORDER BY score DESC
                LIMIT $k
                """,
                emb=query_emb,
                k=k,
                k2=k * 6,  # over-fetch porque pode descartar muito
                kind=kind,
                min_score=float(min_score),
                excluded=excluded,
            )
            return [
                Hit(text=r["text"], source=r["source"], kind=r["kind"], score=float(r["score"]))
                for r in res
            ]

    # -----------------------------------------------------------------------
    # Busca híbrida: vetor + expansão de grafo (1 hop)
    # -----------------------------------------------------------------------
    def search_with_entities(self, query: str, k: int = 5) -> dict:
        if not query.strip():
            return {"hits": []}
        query_emb = self._embeddings.embed_query(query)
        with self._driver.session(database=self._db) as s:
            res = s.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k2, $emb)
                YIELD node AS c, score
                WITH c, score ORDER BY score DESC LIMIT $k
                OPTIONAL MATCH (c)-[:MENTIONS]->(e)
                OPTIONAL MATCH (e)-[r]->(neighbor)
                WITH c, score, e,
                     collect(DISTINCT {{rel: type(r), name: neighbor.name, labels: labels(neighbor)}}) AS entity_neighbors
                WITH c, score,
                     collect(DISTINCT {{
                         name: e.name,
                         labels: labels(e),
                         neighbors: [n IN entity_neighbors WHERE n.name IS NOT NULL]
                     }}) AS entities
                RETURN c.text AS text,
                       c.source AS source,
                       c.kind AS kind,
                       score,
                       entities
                """,
                emb=query_emb,
                k=k,
                k2=k * 4,
            )
            return {
                "hits": [
                    {
                        "text": r["text"],
                        "source": r["source"],
                        "kind": r["kind"],
                        "score": float(r["score"]),
                        "entities": [e for e in r["entities"] if e.get("name")],
                    }
                    for r in res
                ]
            }

    # -----------------------------------------------------------------------
    # Stats / inspeção
    # -----------------------------------------------------------------------
    def count(self) -> int:
        with self._driver.session(database=self._db) as s:
            r = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()
            return int(r["n"]) if r else 0

    def stats(self) -> dict:
        """Conta chunks por kind + lista fontes (debug)."""
        with self._driver.session(database=self._db) as s:
            by_kind = {
                r["kind"]: r["n"]
                for r in s.run(
                    "MATCH (c:Chunk) RETURN c.kind AS kind, count(c) AS n ORDER BY n DESC"
                )
            }
            total = sum(by_kind.values())
            sources_n = s.run("MATCH (s:Source) RETURN count(s) AS n").single()["n"]
            entities_n = s.run(
                "MATCH (e) WHERE any(l IN labels(e) WHERE l IN $allowed) RETURN count(e) AS n",
                allowed=["Product", "Fee", "Requirement", "Benefit", "Channel", "Customer"],
            ).single()["n"]
            mentions_n = s.run(
                "MATCH ()-[m:MENTIONS]->() RETURN count(m) AS n"
            ).single()["n"]
        return {
            "chunks_total": total,
            "chunks_by_kind": by_kind,
            "sources": sources_n,
            "entities": entities_n,
            "mentions": mentions_n,
        }

    def sources(self) -> list[str]:
        with self._driver.session(database=self._db) as s:
            return [r["url"] for r in s.run("MATCH (s:Source) RETURN s.url AS url ORDER BY s.url")]

    def chunks_for_source(self, source: str) -> list[dict]:
        with self._driver.session(database=self._db) as s:
            return [
                {"id": r["id"], "text": r["text"]}
                for r in s.run(
                    """
                    MATCH (s:Source {url: $url})-[:CONTAINS]->(c:Chunk)
                    RETURN c.id AS id, c.text AS text ORDER BY c.chunk_index
                    """,
                    url=source,
                )
            ]

    def delete_source(self, source: str) -> int:
        """Remove uma fonte e todos os seus chunks. Útil pra limpar lixo de testes."""
        with self._driver.session(database=self._db) as s:
            r = s.run(
                """
                MATCH (src:Source {url: $url})-[:CONTAINS]->(c:Chunk)
                WITH src, collect(c) AS chunks
                FOREACH (c IN chunks | DETACH DELETE c)
                DETACH DELETE src
                RETURN size(chunks) AS removed
                """,
                url=source,
            ).single()
            return int(r["removed"]) if r else 0

    def delete_by_kind(self, kind: str) -> int:
        """Remove TODOS os chunks de um kind (ex: 'search_results'). Cuidado."""
        with self._driver.session(database=self._db) as s:
            r = s.run(
                """
                MATCH (c:Chunk {kind: $kind})
                WITH count(c) AS n
                MATCH (c:Chunk {kind: $kind}) DETACH DELETE c
                RETURN n AS removed
                """,
                kind=kind,
            ).single()
            return int(r["removed"]) if r else 0

    # -----------------------------------------------------------------------
    # Acessores (RESTAURADOS — kg_extract e tools de grafo dependem disto)
    # -----------------------------------------------------------------------
    @property
    def driver(self):
        return self._driver

    @property
    def database(self) -> str:
        return self._db


def get_rag() -> RAG:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = RAG()
    return _singleton
