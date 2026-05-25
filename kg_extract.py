"""
Extração de entidades + relações em texto bancário e inserção no grafo Neo4j.

Schema FIXO (`ALLOWED_NODES` / `ALLOWED_RELS`) — propositadamente pequeno
pra evitar grafo barulhento. Cada chunk processado vira:

    (:Chunk)-[:MENTIONS]->(:Product|:Fee|:Requirement|:Benefit|:Channel|:Customer)

E relações de domínio entre as próprias entidades, ex.:

    (:Product)-[:HAS_FEE]->(:Fee)
    (:Product)-[:REQUIRES]->(:Requirement)
    (:Product)-[:OFFERS]->(:Benefit)
    (:Product)-[:AVAILABLE_IN]->(:Channel)
    (:Product)-[:FOR_SEGMENT]->(:Customer)

LLM: Gemini (ChatGoogleGenerativeAI) com response_mime_type=application/json
para forçar output JSON estruturado sem depender de Ollama.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from config import config

ALLOWED_NODES = ["Product", "Fee", "Requirement", "Benefit", "Channel", "Customer"]
ALLOWED_RELS = [
    "HAS_FEE",
    "REQUIRES",
    "OFFERS",
    "AVAILABLE_IN",
    "FOR_SEGMENT",
]


def _extraction_prompt() -> str:
    """Prompt da extracção com display_name/domínio/idioma vindos do config."""
    web = config.web
    return (
        "You extract a small product knowledge graph from text about "
        f"{web.display_name} (source: {web.domain}).\n\n"
        "Allowed node types (use ONLY these):\n"
        "- Product      (a concrete offering, e.g. a card, account, loan, plan)\n"
        "- Fee          (a price/cost; include amount/currency/frequency when known)\n"
        "- Requirement  (a condition the customer must meet)\n"
        "- Benefit      (an advantage or perk)\n"
        "- Channel      (where the product is sold or used, e.g. App, Web, Branch)\n"
        "- Customer     (the target audience / segment)\n\n"
        "Allowed relationships (use ONLY these, all from Product):\n"
        "- (Product)-[HAS_FEE]->(Fee)\n"
        "- (Product)-[REQUIRES]->(Requirement)\n"
        "- (Product)-[OFFERS]->(Benefit)\n"
        "- (Product)-[AVAILABLE_IN]->(Channel)\n"
        "- (Product)-[FOR_SEGMENT]->(Customer)\n\n"
        "Rules:\n"
        "- Be conservative. If unsure, omit. Empty result is fine.\n"
        f"- Use the canonical name in {web.language} as `name`. No quotes, no fluff.\n"
        "- For Fee, include amount/currency/frequency in `properties` if mentioned.\n"
        "- Return ONLY valid JSON matching the schema below. No commentary.\n\n"
        "Output schema:\n"
        '{ "nodes": [ {"label": "Product", "name": "<product name>", "properties": {}} ],\n'
        '  "relationships": [ {"source": {"label": "Product", "name": "<product>"},\n'
        '                       "type": "HAS_FEE",\n'
        '                       "target": {"label": "Fee", "name": "<fee>"}} ] }\n\n'
        "Text to analyze:\n---\n{TEXT}\n---\n"
    )


@dataclass
class GraphDoc:
    nodes: list[dict]
    relationships: list[dict]


def _llm(model_name: str = None) -> ChatGoogleGenerativeAI:
    """Devolve o LLM de extracção (Gemini).

    `response_mime_type=application/json` força o modelo a devolver JSON
    válido sem markdown fences — substitui o `format='json'` do Ollama.
    O parâmetro `model_name` é mantido por compatibilidade mas ignorado.
    """
    return ChatGoogleGenerativeAI(
        model=config.gemini.model,
        google_api_key=config.gemini.api_key,
        temperature=0,
    )

def _normalize_name(name: str) -> str:
    """Reduz variações triviais pra que MERGE não duplique 'Cartão Gold' e 'Cartao Gold '. """
    if not isinstance(name, str):
        return ""
    n = name.strip()
    n = re.sub(r"\s+", " ", n)
    return n


def extract_from_text(text: str, model_name: str = None) -> GraphDoc:
    """Roda o LLM e devolve nodes/relationships filtrados pelo schema."""
    if not text or not text.strip():
        return GraphDoc(nodes=[], relationships=[])

    llm = _llm(model_name)
    prompt = _extraction_prompt().replace("{TEXT}", text[:3500])
    resp = llm.invoke(prompt)

    # Normaliza o conteúdo da resposta para string:
    # Gemini devolve lista de blocos [{"type":"text","text":"..."}] quando
    # response_mime_type=application/json está activo.
    raw = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(raw, list):
        raw = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        ).strip()
    else:
        raw = str(raw).strip()

    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: tenta caçar o primeiro JSON object dentro do texto
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return GraphDoc(nodes=[], relationships=[])
        try:
            data = json.loads(m.group(0))
        except Exception:
            return GraphDoc(nodes=[], relationships=[])

    nodes_in = data.get("nodes", []) if isinstance(data, dict) else []
    rels_in = data.get("relationships", []) if isinstance(data, dict) else []

    nodes: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for n in nodes_in:
        if not isinstance(n, dict):
            continue
        label = n.get("label")
        name = _normalize_name(n.get("name", ""))
        if label not in ALLOWED_NODES or not name:
            continue
        key = (label, name.lower())
        if key in seen:
            continue
        seen.add(key)
        nodes.append({"label": label, "name": name, "properties": n.get("properties") or {}})

    rels: list[dict] = []
    for r in rels_in:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type")
        if rtype not in ALLOWED_RELS:
            continue
        src = r.get("source") or {}
        tgt = r.get("target") or {}
        slabel, sname = src.get("label"), _normalize_name(src.get("name", ""))
        tlabel, tname = tgt.get("label"), _normalize_name(tgt.get("name", ""))
        if slabel not in ALLOWED_NODES or tlabel not in ALLOWED_NODES:
            continue
        if not sname or not tname:
            continue
        rels.append(
            {
                "type": rtype,
                "source": {"label": slabel, "name": sname},
                "target": {"label": tlabel, "name": tname},
            }
        )

    return GraphDoc(nodes=nodes, relationships=rels)


def write_to_neo4j(driver, database: str, chunk_id: str, doc: GraphDoc) -> dict:
    """Escreve nodes + relações no Neo4j e liga as entidades ao chunk fonte."""
    if not doc.nodes and not doc.relationships:
        return {"nodes": 0, "rels": 0, "mentions": 0}

    with driver.session(database=database) as s:
        # nodes
        s.run(
            """
            UNWIND $nodes AS n
            CALL apoc.merge.node([n.label], {name: n.name}, n.properties, n.properties) YIELD node
            RETURN count(node)
            """,
            nodes=doc.nodes,
        ) if _apoc_available(s) else _merge_nodes_plain(s, doc.nodes)

        # MENTIONS
        s.run(
            """
            UNWIND $nodes AS n
            MATCH (c:Chunk {id: $chunk_id})
            CALL {
              WITH c, n
              MATCH (e {name: n.name})
              WHERE n.label IN labels(e)
              MERGE (c)-[:MENTIONS]->(e)
            }
            """,
            chunk_id=chunk_id,
            nodes=doc.nodes,
        )

        # relações entre entidades
        for r in doc.relationships:
            s.run(
                f"""
                MATCH (src {{name: $s_name}}) WHERE $s_label IN labels(src)
                MATCH (tgt {{name: $t_name}}) WHERE $t_label IN labels(tgt)
                MERGE (src)-[:{r["type"]}]->(tgt)
                """,
                s_label=r["source"]["label"],
                s_name=r["source"]["name"],
                t_label=r["target"]["label"],
                t_name=r["target"]["name"],
            )

    return {
        "nodes": len(doc.nodes),
        "rels": len(doc.relationships),
        "mentions": len(doc.nodes),
    }


def _apoc_available(session) -> bool:
    try:
        session.run("CALL apoc.help('apoc') YIELD name RETURN name LIMIT 1").consume()
        return True
    except Exception:
        return False


def _merge_nodes_plain(session, nodes: list[dict]) -> None:
    """Fallback se APOC não estiver instalado: faz MERGE por label conhecido."""
    for n in nodes:
        label = n["label"]
        session.run(
            f"MERGE (e:{label} {{name: $name}}) SET e += $props",
            name=n["name"],
            props=n.get("properties") or {},
        )


# ---------------------------------------------------------------------------
# Entry point usado pelo MCP tool kg_extract
# ---------------------------------------------------------------------------
def extract_for_source(source: Optional[str] = None, limit: int = 50) -> dict:
    """Extrai entidades em chunks ainda não processados.

    - Se `source` for dado, processa apenas chunks dessa fonte.
    - Caso contrário, processa qualquer chunk sem flag `entities_extracted=true`.
    """
    from rag import get_rag

    rag = get_rag()
    driver = rag.driver
    db = rag.database

    # busca chunks pendentes
    with driver.session(database=db) as s:
        if source:
            res = s.run(
                """
                MATCH (src:Source {url: $url})-[:CONTAINS]->(c:Chunk)
                WHERE coalesce(c.entities_extracted, false) = false
                RETURN c.id AS id, c.text AS text
                ORDER BY c.chunk_index LIMIT $limit
                """,
                url=source, limit=limit,
            )
        else:
            res = s.run(
                """
                MATCH (c:Chunk)
                WHERE coalesce(c.entities_extracted, false) = false
                RETURN c.id AS id, c.text AS text
                ORDER BY c.id LIMIT $limit
                """,
                limit=limit,
            )
        chunks = [{"id": r["id"], "text": r["text"]} for r in res]

    totals = {"chunks": 0, "nodes": 0, "rels": 0}
    for c in chunks:
        doc = extract_from_text(c["text"])
        stats = write_to_neo4j(driver, db, c["id"], doc)
        totals["chunks"] += 1
        totals["nodes"] += stats["nodes"]
        totals["rels"] += stats["rels"]
        # marca como processado
        with driver.session(database=db) as s:
            s.run(
                "MATCH (c:Chunk {id: $id}) SET c.entities_extracted = true",
                id=c["id"],
            )

    return totals
