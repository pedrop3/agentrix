"""
Long-term user memory backed by Neo4j.

Schema:
    (:UserProfile {user_id, created_at, updated_at})
    (:UserProfile)-[:REMEMBERS]->(:Fact {key, value, created_at})

Usage:
    mem = get_memory()
    mem.save_fact(user_id, "name", "Carlos")
    facts = mem.load_facts(user_id)   # → {"name": "Carlos", ...}
    text  = mem.as_context(user_id)   # → "User context:\n- name: Carlos\n..."
"""
from __future__ import annotations

import threading
from typing import Optional

from config import config

_singleton_lock = threading.Lock()
_singleton: Optional["UserMemory"] = None


class UserMemory:
    def __init__(self):
        from neo4j import GraphDatabase

        driver_kwargs: dict = {"auth": (config.neo4j.user, config.neo4j.password)}
        try:
            from rag import _silence_neo4j_notifications
            driver_kwargs.update(_silence_neo4j_notifications())
        except Exception:
            pass

        self._driver = GraphDatabase.driver(config.neo4j.uri, **driver_kwargs)
        self._db = config.neo4j.database
        self._init_schema()

    def _init_schema(self) -> None:
        with self._driver.session(database=self._db) as s:
            s.run(
                "CREATE CONSTRAINT userprofile_id IF NOT EXISTS "
                "FOR (u:UserProfile) REQUIRE u.user_id IS UNIQUE"
            )

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    def save_fact(self, user_id: str, key: str, value: str) -> None:
        """Upsert a single fact for a user."""
        import datetime
        now = datetime.datetime.utcnow().isoformat()
        with self._driver.session(database=self._db) as s:
            s.run(
                """
                MERGE (u:UserProfile {user_id: $uid})
                ON CREATE SET u.created_at = $now
                SET u.updated_at = $now
                WITH u
                MERGE (u)-[:REMEMBERS]->(f:Fact {key: $key, user_id: $uid})
                SET f.value = $value, f.updated_at = $now
                """,
                uid=user_id,
                key=key,
                value=value,
                now=now,
            )

    def save_facts(self, user_id: str, facts: dict[str, str]) -> None:
        """Upsert multiple facts at once."""
        for k, v in facts.items():
            self.save_fact(user_id, k, v)

    def delete_fact(self, user_id: str, key: str) -> None:
        with self._driver.session(database=self._db) as s:
            s.run(
                """
                MATCH (:UserProfile {user_id: $uid})-[:REMEMBERS]->(f:Fact {key: $key})
                DETACH DELETE f
                """,
                uid=user_id,
                key=key,
            )

    # -----------------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------------
    def load_facts(self, user_id: str) -> dict[str, str]:
        """Return all facts for a user as a {key: value} dict."""
        if not user_id:
            return {}
        with self._driver.session(database=self._db) as s:
            rows = list(
                s.run(
                    """
                    MATCH (:UserProfile {user_id: $uid})-[:REMEMBERS]->(f:Fact)
                    RETURN f.key AS key, f.value AS value
                    ORDER BY f.key
                    """,
                    uid=user_id,
                )
            )
        return {r["key"]: r["value"] for r in rows}

    def as_context(self, user_id: str) -> str:
        """Return facts as a short context string for injection into prompts.

        Returns empty string if no facts exist.
        """
        facts = self.load_facts(user_id)
        if not facts:
            return ""
        lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        return f"Known facts about this user:\n{lines}"


def get_memory() -> UserMemory:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = UserMemory()
    return _singleton
