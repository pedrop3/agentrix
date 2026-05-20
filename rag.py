"""
RAG sobre Neo4j

- Embeddings: Ollama (`nomic-embed-text`, 768 dim)
- Vector index nativo do Neo4j 5 (`db.index.vector.queryNodes`)
- Knowledge graph na mesma instância: nós :Chunk + :Source + entidades
  bancárias (:Product / :Fee / :Requirement / :Benefit / :Channel / :Customer)

Mantém a mesma API publica do módulo antigo:
    from rag import get_rag, Hit
    rag = get_rag()
    rag.index_text("conteudo...", source="...", kind="web")
    hits = rag.search("query", k=5)

Esquema Cypher (criado em get_rag().init_schema()):
    (:Source {url})-[:CONTAINS]->(:Chunk {id, text, kind, source, embedding})
    (:Chunk)-[:MENTIONS]->(:Entity)
    (:Product)-[:HAS_FEE|REQUIRES|OFFERS|AVAILABLE_IN|FOR_SEGMENT]->...
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("OLLAMA_EMBED_DIM", "768"))  # nomic-embed-text = 768

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "agentrix-dev-pwd")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

VECTOR_INDEX = "chunk_embedding_index"

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
    """Tenta desabilitar notifications informacionais do Neo4j no driver.

    O nome do parametro/enum mudou entre versoes do driver:
      - >= 5.16  : notifications_disabled_classifications + NotificationDisabledClassification
      - 5.x  old : notifications_disabled_categories + NotificationDisabledCategory

    Tambem suprime via logging/warnings como rede de seguranca.
    """
    import logging
    import warnings

    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"neo4j(\..*)?")

    out: dict = {}
    # Tentativa 1: API nova (5.16+, baseada em GQL classifications)
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
    # Tentativa 2: API antiga (categories)
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

        driver_kwargs: dict = {"auth": (NEO4J_USER, NEO4J_PASSWORD)}
        driver_kwargs.update(_silence_neo4j_notifications())

        self._driver = GraphDatabase.driver(NEO4J_URI, **driver_kwargs)
        self._embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self._db = NEO4J_DATABASE
        self.init_schema()

    # -----------------------------------------------------------------------
    # Schema (constraints + vector index)
    # -----------------------------------------------------------------------
    def init_schema(self) -> None:
        with self._driver.session(database=self._db) as s:
            # Constraints
            s.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT source_url IF NOT EXISTS "
                "FOR (s:Source) REQUIRE s.url IS UNIQUE"
            )
            # Constraints pras entidades de banking (chave = name normalizado)
            for label in ("Product", "Fee", "Requirement", "Benefit", "Channel", "Customer"):
                s.run(
                    f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                )

            # Vector index nativo (Neo4j 5+). cosine é o padrão pra similaridade semântica
            s.run(
                f"""
                CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
                FOR (c:Chunk) ON c.embedding
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {EMBED_DIM},
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
                    "metadata": {k: v for k, v in meta.items() if k != "title"},
                }
                for i in range(len(chunks))
            ],
        }
        # Cypher: cria/atualiza Source, depois faz upsert dos Chunks ligados a ele.
        # `MERGE` na Chunk{id} garante reindex sem duplicar.
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
                
                // Em vez de SET c.metadata = row.metadata, usamos APOC ou fundimos as propriedades de forma plana
                // Isso garante que se row.metadata contiver chaves primitivas, elas viram propriedades diretas do nó c
                SET c += row.metadata
                
                MERGE (src)-[:CONTAINS]->(c)
                """,
                params,
            )
        return len(chunks)

    # -----------------------------------------------------------------------
    # Busca
    # -----------------------------------------------------------------------
    def search(self, query: str, k: int = 5, kind: Optional[str] = None) -> list[Hit]:
        if not query.strip():
            return []
        query_emb = self._embeddings.embed_query(query)
        # Pega top-K * 2 do vector index e filtra por kind no Cypher.
        # (db.index.vector.queryNodes não aceita filtros direto.)
        with self._driver.session(database=self._db) as s:
            res = s.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k2, $emb)
                YIELD node, score
                WITH node AS c, score
                WHERE $kind IS NULL OR c.kind = $kind
                RETURN c.text AS text,
                       c.source AS source,
                       c.kind AS kind,
                       score
                ORDER BY score DESC
                LIMIT $k
                """,
                emb=query_emb,
                k=k,
                k2=k * 4,
                kind=kind,
            )
            return [
                Hit(text=r["text"], source=r["source"], kind=r["kind"], score=float(r["score"]))
                for r in res
            ]

    # -----------------------------------------------------------------------
    # Busca híbrida: vetor + expansão de grafo
    # -----------------------------------------------------------------------
    def search_with_entities(self, query: str, k: int = 5) -> dict:
        """
        Busca chunks por similaridade e devolve também as entidades mencionadas
        e seus vizinhos imediatos. Permite o knower fazer 'GraphRAG light'.
        """
        if not query.strip():
            return {"hits": [], "entities": []}
        query_emb = self._embeddings.embed_query(query)
        with self._driver.session(database=self._db) as s:
            res = s.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k2, $emb)
                YIELD node AS c, score
                WITH c, score ORDER BY score DESC LIMIT $k
                OPTIONAL MATCH (c)-[:MENTIONS]->(e)
                OPTIONAL MATCH (e)-[r]->(neighbor)
                RETURN c.text AS text,
                       c.source AS source,
                       c.kind AS kind,
                       score,
                       collect(DISTINCT {{
                         name: e.name,
                         labels: labels(e),
                         neighbors: collect(DISTINCT {{
                           rel: type(r),
                           name: neighbor.name,
                           labels: labels(neighbor)
                         }})
                       }}) AS entities
                """,
                emb=query_emb,
                k=k,
                k2=k * 4,
            )
            hits = []
            for r in res:
                hits.append(
                    {
                        "text": r["text"],
                        "source": r["source"],
                        "kind": r["kind"],
                        "score": float(r["score"]),
                        "entities": [e for e in r["entities"] if e["name"]],
                    }
                )
            return {"hits": hits}

    def search_with_entities2(self, query: str, k: int = 5) -> dict:
        """
        Busca chunks por similaridade e devolve também as entidades mencionadas
        e seus vizinhos imediatos. Permite o knower fazer 'GraphRAG light'.
        """
        if not query.strip():
            return {"hits": [], "entities": []}

        query_emb = self._embeddings.embed_query(query)

        with self._driver.session(database=self._db) as s:
            res = s.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k2, $emb)
                YIELD node AS c, score
                WITH c, score ORDER BY score DESC LIMIT $k
                
                // 1. Encontra as entidades mencionadas
                OPTIONAL MATCH (c)-[:MENTIONS]->(e)
                
                // 2. Encontra os vizinhos dessas entidades
                OPTIONAL MATCH (e)-[r]->(neighbor)
                
                // 3. Primeiro agrupamos os vizinhos para cada entidade
                WITH c, score, e, 
                     collect(DISTINCT {{
                         rel: type(r),
                         name: neighbor.name,
                         labels: labels(neighbor)
                     }}) AS entity_neighbors
                
                // 4. Agora agrupamos as entidades para o chunk c
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

            hits = []
            for r in res:
                # Filtramos entidades que possam ter vindo nulas devido ao OPTIONAL MATCH
                clean_entities = [e for e in r["entities"] if e.get("name") is not None]

                hits.append(
                    {
                        "text": r["text"],
                        "source": r["source"],
                        "kind": r["kind"],
                        "score": float(r["score"]),
                        "entities": clean_entities,
                    }
                )
            return {"hits": hits}



def get_rag() -> RAG:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = RAG()
    return _singleton
