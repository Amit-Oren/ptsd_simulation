"""
Typed knowledge graph for the interviewer's cross-session patient memory.

A graph holds specific things the patient has mentioned (nodes) and the
relationships between them (edges), accumulated across every session for
one patient. 
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

NODE_TYPES = (
    "Triggers",
    "Negative Appraisals",
    "Memory",
    "Threat",
    "Maladaptive Strategies",
)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "item"


@dataclass
class MemoryNode:
    id: str
    type: str
    label: str
    quotes: list[str] = field(default_factory=list)
    sessions_seen: list[int] = field(default_factory=list)

    def add_observation(self, quote: str, session_idx: int, max_quotes: int = 3) -> None:
        if session_idx not in self.sessions_seen:
            self.sessions_seen.append(session_idx)
        if quote and quote not in self.quotes:
            self.quotes.append(quote)
            if len(self.quotes) > max_quotes:
                self.quotes = self.quotes[-max_quotes:]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "quotes": list(self.quotes),
            "sessions_seen": list(self.sessions_seen),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        return cls(
            id=d["id"], type=d["type"], label=d["label"],
            quotes=list(d.get("quotes", [])),
            sessions_seen=list(d.get("sessions_seen", [])),
        )


@dataclass
class MemoryEdge:
    source: str
    target: str
    relation: str
    weight: float
    sessions_seen: list[int] = field(default_factory=list)

    def observe(self, relation: str, weight: float, session_idx: int) -> None:
        n = len(self.sessions_seen)
        self.weight = (self.weight * n + weight) / (n + 1)
        self.relation = relation  # most recent wording wins
        if session_idx not in self.sessions_seen:
            self.sessions_seen.append(session_idx)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "sessions_seen": list(self.sessions_seen),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEdge":
        return cls(
            source=d["source"], target=d["target"], relation=d["relation"],
            weight=d["weight"], sessions_seen=list(d.get("sessions_seen", [])),
        )


class TypedKnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[tuple[str, str], MemoryEdge] = {}

    # ---- nodes ----

    def _match_existing_node(self, type_: str, label: str) -> Optional[str]:
        same_type = [n for n in self.nodes.values() if n.type == type_]
        for n in same_type:
            if n.label.strip().lower() == label.strip().lower():
                return n.id
        return None

    def add_or_merge_node(self, type_: str, label: str, quote: str, session_idx: int,
                           existing_id: Optional[str] = None) -> str:
        node_id = existing_id if existing_id and existing_id in self.nodes else None
        if node_id is None:
            node_id = self._match_existing_node(type_, label)
        if node_id is None:
            node_id = f"{_slugify(type_)}_{_slugify(label)}_{uuid.uuid4().hex[:6]}"
            self.nodes[node_id] = MemoryNode(id=node_id, type=type_, label=label)
        self.nodes[node_id].add_observation(quote, session_idx)
        return node_id

    # ---- edges ----

    def add_or_merge_edge(self, source_id: str, target_id: str, relation: str,
                           weight: float, session_idx: int) -> None:
        key = (source_id, target_id)
        if key in self.edges:
            self.edges[key].observe(relation, weight, session_idx)
        else:
            edge = MemoryEdge(source=source_id, target=target_id, relation=relation, weight=weight)
            edge.sessions_seen.append(session_idx)
            self.edges[key] = edge

    # ---- ingest ----

    def ingest(self, extraction: dict, session_idx: int) -> None:
        """extraction = {"entities": [...], "relations": [...]} from the analyst's structured output."""
        temp_to_final: dict[str, str] = {}
        for entity in extraction.get("entities", []):
            node_id = self.add_or_merge_node(
                type_=entity["type"], label=entity["label"], quote=entity.get("quote", ""),
                session_idx=session_idx, existing_id=entity.get("existing_id"),
            )
            temp_to_final[entity["temp_id"]] = node_id

        for rel in extraction.get("relations", []):
            source_id = temp_to_final.get(rel["source_temp_id"])
            target_id = temp_to_final.get(rel["target_temp_id"])
            if not source_id or not target_id:
                continue
            weight = rel.get("weight", 5) / 10.0
            self.add_or_merge_edge(source_id, target_id, rel["relation"], weight, session_idx)

    # ---- rendering ----

    def nodes_by_type(self) -> dict[str, list[MemoryNode]]:
        grouped: dict[str, list[MemoryNode]] = {t: [] for t in NODE_TYPES}
        for n in self.nodes.values():
            grouped.setdefault(n.type, []).append(n)
        return grouped

    def render_for_coreference(self) -> str:
        """Existing entities shown to the extractor so it can resolve coreference itself."""
        if self.is_empty():
            return "(no existing entities yet — this is the first session)"
        lines = []
        grouped = self.nodes_by_type()
        for t in NODE_TYPES:
            nodes = grouped.get(t, [])
            if not nodes:
                continue
            lines.append(f"{t}:")
            for n in nodes:
                lines.append(f'  - id={n.id}  label="{n.label}"')
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return not self.nodes

    # ---- serialization ----

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TypedKnowledgeGraph":
        graph = cls()
        for nd in d.get("nodes", []):
            node = MemoryNode.from_dict(nd)
            graph.nodes[node.id] = node
        for ed in d.get("edges", []):
            edge = MemoryEdge.from_dict(ed)
            graph.edges[(edge.source, edge.target)] = edge
        return graph
