"""Graph-memory integration for STEER using Helix DB (or any graph+vector DB).

Provides a minimal abstraction so STEER specs can reference persistent,
queryable agent memory.  The `HelixMemory` class wraps the Helix DB
Python SDK and exposes the operations agents need: write facts, query
related context, and traverse collaboration graphs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    import helixdb  # type: ignore
    HELIX_AVAILABLE = True
except Exception:
    HELIX_AVAILABLE = False


@dataclass
class MemoryFact:
    """A single fact stored in the agent's knowledge graph."""
    entity_type: str          # e.g. "Threat", "Incident", "MeetingNote"
    properties: dict[str, Any]  # typed properties, may include embeddings
    embedding: list[float] | None = None
    source_agent: str = ""


@dataclass
class QueryResult:
    """Result of a memory query."""
    entities: list[dict[str, Any]]
    raw: Any = None


class GraphMemoryBackend(ABC):
    """Abstract base for any graph+vector memory store."""

    @abstractmethod
    def upsert(self, fact: MemoryFact) -> str:
        """Insert or update a fact; return the entity ID."""

    @abstractmethod
    def query(
        self,
        entity_type: str | None = None,
        embedding: list[float] | None = None,
        top_k: int = 10,
        filter_props: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Query the graph, optionally with vector similarity."""

    @abstractmethod
    def traverse(
        self,
        start_id: str,
        edge_type: str,
        depth: int = 2,
    ) -> QueryResult:
        """Traverse relationships from a starting node."""


class HelixMemory(GraphMemoryBackend):
    """Helix DB backend implementation."""

    def __init__(
        self,
        uri: str = "helix://localhost:6969",
        username: str | None = None,
        password: str | None = None,
        namespace: str = "steer",
    ):
        if not HELIX_AVAILABLE:
            raise RuntimeError("helixdb Python SDK not installed. `pip install helixdb`")
        self.client = helixdb.Client(uri, username, password)
        self.namespace = namespace

    def upsert(self, fact: MemoryFact) -> str:
        props = fact.properties | {"__source_agent": fact.source_agent}
        if fact.embedding:
            props["__embedding"] = fact.embedding

        # HelixQL upsert: MATCH or CREATE
        query = f"""
        MATCH (n:{fact.entity_type} {{id: $id}})
        SET n += $props
        RETURN n.id
        """
        result = self.client.query(
            query,
            params={"id": fact.properties.get("id"), "props": props},
            namespace=self.namespace,
        )
        if result.rows:
            return result.rows[0]["n.id"]

        # Create new
        create_query = f"""
        CREATE (n:{fact.entity_type} $props)
        RETURN n.id
        """
        result = self.client.query(
            create_query,
            params={"props": props},
            namespace=self.namespace,
        )
        return result.rows[0]["n.id"]

    def query(
        self,
        entity_type: str | None = None,
        embedding: list[float] | None = None,
        top_k: int = 10,
        filter_props: dict[str, Any] | None = None,
    ) -> QueryResult:
        where_clauses = []
        params: dict[str, Any] = {"top_k": top_k}

        if entity_type:
            where_clauses.append("n:$entity_type")

        if filter_props:
            for k, v in filter_props.items():
                where_clauses.append(f"n.{k} = ${k}")
                params[k] = v

        where_str = " AND ".join(where_clauses) if where_clauses else "true"

        if embedding:
            # Hybrid graph + vector search (HelixQL VECTOR SEARCH)
            query = f"""
            MATCH (n)
            WHERE {where_str}
            VECTOR SEARCH n.__embedding SIMILAR TO $embedding TOP $top_k
            RETURN n, score
            """
            params["embedding"] = embedding
        else:
            query = f"""
            MATCH (n)
            WHERE {where_str}
            RETURN n LIMIT $top_k
            """

        result = self.client.query(query, params=params, namespace=self.namespace)
        return QueryResult(
            entities=[{"properties": r["n"], "score": r.get("score")} for r in result.rows],
            raw=result,
        )

    def traverse(
        self,
        start_id: str,
        edge_type: str,
        depth: int = 2,
    ) -> QueryResult:
        query = f"""
        MATCH (start {{id: $start_id}})-[:{edge_type}*1..{depth}]->(target)
        RETURN DISTINCT target
        """
        result = self.client.query(
            query,
            params={"start_id": start_id},
            namespace=self.namespace,
        )
        return QueryResult(
            entities=[r["target"] for r in result.rows],
            raw=result,
        )


class InMemoryGraphMemory(GraphMemoryBackend):
    """In-memory fallback for testing / local dev (no Helix DB required)."""

    def __init__(self):
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[str, list[tuple[str, str]]] = {}  # (start, end)
        self._embeddings: dict[str, list[float]] = {}

    def upsert(self, fact: MemoryFact) -> str:
        eid = fact.properties.get("id", f"{fact.entity_type}-{len(self._nodes)}")
        fact.properties["id"] = eid
        self._nodes[eid] = fact.properties | {"__entity_type": fact.entity_type, "__source_agent": fact.source_agent}
        if fact.embedding:
            self._embeddings[eid] = fact.embedding
        return eid

    def query(
        self,
        entity_type: str | None = None,
        embedding: list[float] | None = None,
        top_k: int = 10,
        filter_props: dict[str, Any] | None = None,
    ) -> QueryResult:
        results = []
        for eid, props in self._nodes.items():
            if entity_type and props.get("__entity_type") != entity_type:
                continue
            if filter_props:
                if not all(props.get(k) == v for k, v in filter_props.items()):
                    continue
            score = None
            if embedding and eid in self._embeddings:
                # cosine similarity
                import math
                a = embedding
                b = self._embeddings[eid]
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(y * y for y in b))
                score = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
            results.append({"id": eid, "properties": props, "score": score})

        if embedding:
            results.sort(key=lambda x: x["score"] or -1.0, reverse=True)
        return QueryResult(entities=results[:top_k])

    def traverse(
        self,
        start_id: str,
        edge_type: str,
        depth: int = 2,
    ) -> QueryResult:
        visited = set()
        current = {start_id}

        for _ in range(depth):
            next_level = set()
            for node in current:
                for (s, e), etype in self._edges.items():
                    if etype == edge_type:
                        if s == node:
                            next_level.add(e)
                        elif e == node:
                            next_level.add(s)
            visited.update(current)
            current = next_level - visited

        return QueryResult(entities=[self._nodes[eid] for eid in current if eid in self._nodes])


def create_memory_backend(uri: str) -> GraphMemoryBackend:
    """Factory that picks the right backend from a URI."""
    if uri.startswith("helix://"):
        return HelixMemory(uri)
    return InMemoryGraphMemory()


# --- Example usage ---
if __name__ == "__main__":
    mem = create_memory_backend("memory://")

    # Store some facts
    mem.upsert(MemoryFact(
        entity_type="Threat",
        properties={"id": "threat-001", "name": "CVE-2024-1234", "severity": "critical"},
        embedding=[0.1] * 384,
        source_agent="CyberAnalyst",
    ))
    mem.upsert(MemoryFact(
        entity_type="Threat",
        properties={"id": "threat-002", "name": "CVE-2024-5678", "severity": "high"},
        embedding=[0.2] * 384,
        source_agent="CyberAnalyst",
    ))

    # Query with vector similarity
    results = mem.query(entity_type="Threat", embedding=[0.15] * 384, top_k=5)
    for r in results.entities:
        print(f"{r['properties']['name']}: {r.get('score')}")