"""
EdgeGraph is the single format for all edge data — in memory, in prompts, in JSON output.
"""

# {parent_node: {child_node: weight}}  — weight is 0.0–1.0
EdgeGraph = dict

# {node_name: active}  — active is bool or 0/1
NodeGraph = dict


def from_schema(direct_edges: dict) -> EdgeGraph:
    """Create a zero-weighted EdgeGraph from DIRECT_EDGES schema {(p,c): 0}."""
    result: EdgeGraph = {}
    for (p, c) in direct_edges:
        result.setdefault(p, {})[c] = 0.0
    return result


def iter_edges(graph: EdgeGraph):
    """Yield (parent, child, weight) for every edge in the graph."""
    for p, children in graph.items():
        for c, v in children.items():
            yield p, c, v


def get_weight(graph: EdgeGraph, parent: str, child: str, default: float = 0.0) -> float:
    return graph.get(parent, {}).get(child, default)


def set_weight(graph: EdgeGraph, parent: str, child: str, weight: float) -> None:
    graph.setdefault(parent, {})[child] = weight


def count_active(graph: EdgeGraph, threshold: float = 0.0) -> int:
    """Count edges with weight strictly above threshold."""
    return sum(1 for _, _, v in iter_edges(graph) if v > threshold)


def count_by_strength(graph: EdgeGraph) -> tuple[int, int]:
    """Return (strong, moderate) counts: strong ≥ 0.7, moderate 0.4–0.7."""
    strong = sum(1 for _, _, v in iter_edges(graph) if v >= 0.7)
    moderate = sum(1 for _, _, v in iter_edges(graph) if 0.4 <= v < 0.7)
    return strong, moderate


def format_for_prompt(graph: EdgeGraph, only_active: bool = True) -> str:
    """Format edges as a bullet list for LLM prompts."""
    lines = [
        f"  - {p} → {c}: {v:.4f}"
        for p, c, v in iter_edges(graph)
        if not only_active or v > 0
    ]
    return "\n".join(lines) or "  (none)"


def active_edge_set(graph: EdgeGraph) -> set[tuple[str, str]]:
    """Return set of (parent, child) tuples for edges with weight > 0."""
    return {(p, c) for p, c, v in iter_edges(graph) if v > 0}


