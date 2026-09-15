"""Renders the interviewer's passive, factual memory context from a
TypedKnowledgeGraph. Never suggests topics or questions — only reports
what's already been covered, so the interviewer doesn't repeat ground."""
from simulation.memory.kg import NODE_TYPES, TypedKnowledgeGraph


def build_interviewer_memory_context(graph: TypedKnowledgeGraph | None) -> str:
    if graph is None or graph.is_empty():
        return "(No prior sessions — this is the first time meeting this patient.)"

    lines = ["Topics already discussed with this patient in prior sessions:"]
    grouped = graph.nodes_by_type()
    id_to_label = {n.id: n.label for n in graph.nodes.values()}

    for node_type in NODE_TYPES:
        nodes = grouped.get(node_type, [])
        if not nodes:
            continue
        lines.append(f"\n{node_type}:")
        for n in nodes:
            sessions = ", ".join(str(s) for s in sorted(n.sessions_seen))
            lines.append(f"  - {n.label} (sessions: {sessions})")

    if graph.edges:
        lines.append("\nRelationships noted between these:")
        for edge in graph.edges.values():
            src = id_to_label.get(edge.source, edge.source)
            tgt = id_to_label.get(edge.target, edge.target)
            sessions = ", ".join(str(s) for s in sorted(edge.sessions_seen))
            lines.append(f"  - {src} --{edge.relation}--> {tgt} (sessions: {sessions})")

    return "\n".join(lines)
