"""
Configuracao centralizada do agentrix.

Le `.env` UMA vez e expoe `config` (singleton) com tudo tipado:
    from config import config
    config.ollama.model
    config.neo4j.uri
    config.recursion_limit

Quaisquer modulos (main.py, server.py, rag.py, kg_extract.py, agents/*)
devem ler dali. Nao reler env var direto pra evitar defaults divergentes.

Overrides por papel (opcionais):
- OLLAMA_MODEL_SUPERVISOR  : routing (structured output simples)
- OLLAMA_MODEL_DIRECT      : conversa solta
- OLLAMA_MODEL_KG_EXTRACT  : extracao de JSON estruturado pro grafo
- OLLAMA_MODEL_KNOWER      : agent que faz tool calls + RAG/grafo
- OLLAMA_MODEL_MATHY       : agent de matematica
- OLLAMA_MODEL_RESEARCHER  : agent de busca em notas
- OLLAMA_MODEL_WRITER      : agent que salva notas

Se nao setados, caem no `OLLAMA_MODEL` principal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Carrega .env do diretorio do projeto, nao importa o cwd de quem importou.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    """Le env var; se vier vazia ('') trata como nao setada."""
    val = os.getenv(key, "")
    return val if val else default


def _model_or_default(key: str, default_model: str) -> str:
    """Override de modelo por papel; cai no default se nao tiver."""
    return _env(key, default_model)


# Modelo "base" — usado como fallback pros papeis especificos
_DEFAULT_MODEL = _env("OLLAMA_MODEL", "qwen3:8b")


@dataclass(frozen=True)
class OllamaConfig:
    # Base
    model: str = _DEFAULT_MODEL

    # Por papel (caem no `model` se nao setados)
    supervisor_model: str = field(
        default_factory=lambda: _model_or_default("OLLAMA_MODEL_SUPERVISOR", _DEFAULT_MODEL)
    )
    direct_model: str = field(
        default_factory=lambda: _model_or_default("OLLAMA_MODEL_DIRECT", _DEFAULT_MODEL)
    )
    extract_model: str = field(
        default_factory=lambda: _model_or_default("OLLAMA_MODEL_KG_EXTRACT", _DEFAULT_MODEL)
    )
    knower_model: str = field(
        default_factory=lambda: _model_or_default("OLLAMA_MODEL_KNOWER", _DEFAULT_MODEL)
    )
    mathy_model: str = field(
        default_factory=lambda: _model_or_default("OLLAMA_MODEL_MATHY", _DEFAULT_MODEL)
    )
    researcher_model: str = field(
        default_factory=lambda: _model_or_default("OLLAMA_MODEL_RESEARCHER", _DEFAULT_MODEL)
    )
    writer_model: str = field(
        default_factory=lambda: _model_or_default("OLLAMA_MODEL_WRITER", _DEFAULT_MODEL)
    )

    # Embeddings
    embed_model: str = field(
        default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    embed_dim: int = field(
        default_factory=lambda: int(_env("OLLAMA_EMBED_DIM", "768"))
    )

    # Defaults do ChatOllama
    temperature: float = field(
        default_factory=lambda: float(_env("OLLAMA_TEMPERATURE", "0"))
    )
    num_ctx: int = field(
        default_factory=lambda: int(_env("OLLAMA_NUM_CTX", "8192"))
    )


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "agentrix-dev-pwd"))
    database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j"))
    vector_index: str = "chunk_embedding_index"


def _default_allowed_hosts(domain: str) -> tuple[str, ...]:
    domain = domain.lower().strip().lstrip(".")
    if not domain:
        return ()
    # se ja veio com subdominio (ex: "www.foo.pt"), inclui sem o "www" tambem
    hosts = {domain}
    if domain.startswith("www."):
        hosts.add(domain[4:])
    else:
        hosts.add(f"www.{domain}")
    return tuple(sorted(hosts))


def _parse_hosts_env(raw: str, fallback_domain: str) -> tuple[str, ...]:
    if not raw:
        return _default_allowed_hosts(fallback_domain)
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return tuple(parts) if parts else _default_allowed_hosts(fallback_domain)


_DEFAULT_DOMAIN = _env("WEB_DOMAIN", "")


@dataclass(frozen=True)
class WebContextConfig:
    """Dominio externo + nome humano que o agent usa em prompts.

    Tudo configuravel via env:
      WEB_DOMAIN
      WEB_BASE_URL        URL base com protocolo (default: https://www.<dominio>)
      WEB_ALLOWED_HOSTS   csv de hosts permitidos no scraping
                          (default: '<dominio>, www.<dominio>')
      WEB_DISPLAY_NAME    nome humano nos prompts (ex: 'Portugal')
      WEB_LANGUAGE        idioma de resposta (default 'PT-PT')
      WEB_DESCRIPTION     descricao curta usada em prompts
    """

    domain: str = _DEFAULT_DOMAIN
    base_url: str = field(
        default_factory=lambda: _env("WEB_BASE_URL", f"https://www.{_DEFAULT_DOMAIN}")
    )
    allowed_hosts: tuple[str, ...] = field(
        default_factory=lambda: _parse_hosts_env(
            os.getenv("WEB_ALLOWED_HOSTS", ""), _DEFAULT_DOMAIN
        )
    )
    display_name: str = field(
        default_factory=lambda: _env("WEB_DISPLAY_NAME", "Portugal")
    )
    language: str = field(default_factory=lambda: _env("WEB_LANGUAGE", "PT-PT"))
    description: str = field(
        default_factory=lambda: _env(
            "WEB_DESCRIPTION",
            "the public website of the configured organization",
        )
    )

@dataclass(frozen=True)
class GeminiConfig:
    model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash")
    )

    api_key: str = field(
        default_factory=lambda: _env("GEMINI_API_KEY", "")
    )

    temperature: float = field(
        default_factory=lambda: float(_env("GEMINI_TEMPERATURE", "0"))
    )

@dataclass(frozen=True)
class AppConfig:
    # LLM providers
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)

    # Infra
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    web: WebContextConfig = field(default_factory=WebContextConfig)

    # App paths
    project_root: Path = PROJECT_ROOT
    memory_db: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "memory.db")
    )

    # LangGraph / runtime
    recursion_limit: int = field(
        default_factory=lambda: int(_env("AGENTRIX_RECURSION_LIMIT", "10"))
    )

    port: int = field(
        default_factory=lambda: int(_env("PORT", "8000"))
    )

    reload: bool = field(
        default_factory=lambda: _env("AGENTRIX_RELOAD", "").lower() in {"1", "true", "yes"}
    )


# Singleton publico
config = AppConfig()
