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
FULLTEXT_INDEX = "chunk_text_fulltext"

# Caracteres reservados do Lucene que precisam ser escapados em queries livres.
# (https://lucene.apache.org/core/9_0_0/queryparser/org/apache/lucene/queryparser/classic/package-summary.html)
_LUCENE_SPECIALS = r'+-&|!(){}[]^"~*?:\/'

# Pesos default da fusao RRF; constante mantida pra debug
RRF_K = 60

_singleton_lock = threading.Lock()
_singleton: Optional["RAG"] = None


def _escape_lucene(query: str) -> str:
    """Escapa chars reservados do Lucene pra busca livre, sem quebrar a query.

    Tambem remove caracteres de operadores que ficariam isolados.
    """
    out = []
    for ch in query:
        if ch in _LUCENE_SPECIALS:
            out.append("\\")
        out.append(ch)
    cleaned = "".join(out).strip()
    return cleaned or "*"  # fallback inocuo se vier vazio


def _rrf_fuse(
    vector_ranked: list[tuple[str, float]],
    fulltext_ranked: list[tuple[str, float]],
    rrf_k: int = RRF_K,
) -> dict[str, dict]:
    """Reciprocal Rank Fusion: combina dois rankings em um score unico.

    Score de cada id = sum(1/(rrf_k + rank_i)) em todos os rankings onde aparece.
    Retorna dict {chunk_id -> {rrf, vector_score, fulltext_score, vector_rank, fulltext_rank}}.
    """
    out: dict[str, dict] = {}
    for rank, (cid, score) in enumerate(vector_ranked, start=1):
        out.setdefault(cid, {
            "rrf": 0.0, "vector_score": None, "fulltext_score": None,
            "vector_rank": None, "fulltext_rank": None,
        })
        out[cid]["rrf"] += 1.0 / (rrf_k + rank)
        out[cid]["vector_score"] = score
        out[cid]["vector_rank"] = rank
    for rank, (cid, score) in enumerate(fulltext_ranked, start=1):
        out.setdefault(cid, {
            "rrf": 0.0, "vector_score": None, "fulltext_score": None,
            "vector_rank": None, "fulltext_rank": None,
        })
        out[cid]["rrf"] += 1.0 / (rrf_k + rank)
        out[cid]["fulltext_score"] = score
        out[cid]["fulltext_rank"] = rank
    return out


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
            # Fulltext (Lucene) index sobre o texto do chunk — usado pelo
            # hybrid_search pra combinar com a similaridade semantica.
            # Analyzer 'portuguese' faz stemming PT, normaliza acentos e
            # aplica IDF — resolve plurais e dá peso por raridade.
            try:
                s.run(
                    f"""
                    CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS
                    FOR (c:Chunk) ON EACH [c.text]
                    OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'portuguese' }} }}
                    """
                )
            except Exception as e:
                # Fallback: alguma versao antiga do Neo4j pode nao aceitar
                # analyzer custom. Cria com default (ainda funciona, so sem stemming PT).
                print(f"[rag] portuguese analyzer falhou ({e}); usando default")
                s.run(
                    f"""
                    CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS
                    FOR (c:Chunk) ON EACH [c.text]
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
    # Hybrid search: vector + fulltext, fundidos via RRF
    # -----------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        kind: Optional[str] = None,
        exclude_kinds: Optional[Iterable[str]] = None,
        min_vector_score: float = 0.0,
        require_lexical: bool = True,
    ) -> list[Hit]:
        """Vector + fulltext fundidos com Reciprocal Rank Fusion.

        Resolve dois problemas do vector puro:
        - Falsos positivos em indice pequeno (vector achava chunks tematicamente
          proximos mas semanticamente errados).
        - Falsos negativos do lexical puro (plurais, sinonimos).

        Estratégia:
        1. Roda vector search (top 3*k) e fulltext search (top 3*k).
        2. Funde ranks via RRF: `score = sum(1/(60+rank_i))` pra cada lista
           onde o chunk aparece.
        3. Filtro de qualidade (`require_lexical=True`): mantem so chunks que
           OU apareceram no fulltext (= ao menos um termo significativo match)
           OU tem score vetorial >= 0.80 (vector forte sozinho ja vale).
        4. Devolve top-k ordenados por RRF.

        Em cada Hit, `score` carrega o RRF combinado (normalizado pra 0..1
        dividindo pelo max teorico). Hits ainda expoem o vector_score original
        via metadata se precisar — aqui simplificamos pra Hit padrao.

        - `min_vector_score`: piso pra vector hits entrarem na fusao.
                              (0.0 = aceita tudo do vector)
        - `kind`/`exclude_kinds`: filtros (mesma semantica do search() puro)
        """
        if not query.strip():
            return []

        excluded = list(exclude_kinds) if exclude_kinds else []
        over_fetch = max(k * 3, 15)

        # 1) Vector search
        query_emb = self._embeddings.embed_query(query)
        with self._driver.session(database=self._db) as s:
            vec_res = list(
                s.run(
                    f"""
                    CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k2, $emb)
                    YIELD node AS c, score
                    WHERE ($kind IS NULL OR c.kind = $kind)
                      AND (size($excluded) = 0 OR NOT c.kind IN $excluded)
                      AND score >= $min_score
                    RETURN c.id AS id, c.text AS text, c.source AS source,
                           c.kind AS kind, score
                    ORDER BY score DESC
                    LIMIT $k2
                    """,
                    emb=query_emb,
                    k2=over_fetch,
                    kind=kind,
                    excluded=excluded,
                    min_score=float(min_vector_score),
                )
            )

            # 2) Fulltext search (Lucene) com o mesmo filtro de kind
            lucene_q = _escape_lucene(query)
            ft_res = list(
                s.run(
                    f"""
                    CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $q, {{limit: $k2}})
                    YIELD node AS c, score
                    WHERE ($kind IS NULL OR c.kind = $kind)
                      AND (size($excluded) = 0 OR NOT c.kind IN $excluded)
                    RETURN c.id AS id, c.text AS text, c.source AS source,
                           c.kind AS kind, score
                    """,
                    q=lucene_q,
                    k2=over_fetch,
                    kind=kind,
                    excluded=excluded,
                )
            )

        # 3) Indexa metadata dos rows pra reaproveitar text/source/kind
        by_id: dict[str, dict] = {}
        vector_ranked: list[tuple[str, float]] = []
        for r in vec_res:
            cid = r["id"]
            by_id[cid] = {"text": r["text"], "source": r["source"], "kind": r["kind"]}
            vector_ranked.append((cid, float(r["score"])))

        fulltext_ranked: list[tuple[str, float]] = []
        fulltext_ids: set[str] = set()
        for r in ft_res:
            cid = r["id"]
            by_id.setdefault(cid, {"text": r["text"], "source": r["source"], "kind": r["kind"]})
            fulltext_ranked.append((cid, float(r["score"])))
            fulltext_ids.add(cid)

        # 4) Fusao RRF
        fused = _rrf_fuse(vector_ranked, fulltext_ranked)

        # 5) Filtro de qualidade: chunk so e mantido se passa em uma das duas
        #    condicoes (defesa contra false-positive do vector):
        STRONG_VECTOR = 0.80
        kept: list[tuple[str, dict]] = []
        for cid, scores in fused.items():
            if not require_lexical:
                kept.append((cid, scores))
                continue
            ft_match = cid in fulltext_ids
            vec_strong = (scores["vector_score"] or 0.0) >= STRONG_VECTOR
            if ft_match or vec_strong:
                kept.append((cid, scores))

        # 6) Ordena por RRF desc e normaliza score pra 0..1
        if not kept:
            return []
        max_rrf = max(s["rrf"] for _, s in kept) or 1.0
        kept.sort(key=lambda x: -x[1]["rrf"])

        hits: list[Hit] = []
        for cid, scores in kept[:k]:
            meta = by_id[cid]
            normalized = scores["rrf"] / max_rrf
            hits.append(
                Hit(
                    text=meta["text"],
                    source=meta["source"],
                    kind=meta["kind"],
                    score=normalized,
                )
            )
        return hits

    # -----------------------------------------------------------------------
    # Busca semantica + expansão de grafo (1 hop)
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
