"""
memory_graph.py — Plotly figure for a patient's memory knowledge graph.

Unlike the fixed 5-node cognitive model graph (components/graph.py), the
memory graph grows over time and has no fixed layout, so positions are
computed with a NetworkX spring layout and nodes are colored by their
coarse type (Triggers/Memory/Negative Appraisals/Threat/Maladaptive
Strategies).
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import networkx as nx
import plotly.graph_objects as go

TYPE_COLORS = {
    "Triggers":               "#e67e22",
    "Negative Appraisals":    "#9b59b6",
    "Memory":                 "#3498db",
    "Threat":                 "#e74c3c",
    "Maladaptive Strategies": "#2ecc71",
}
DEFAULT_COLOR = "#95a5a6"


def render_memory_graph(graph: dict) -> go.Figure:
    """
    Build and return a Plotly figure of a patient's memory knowledge graph.

    Parameters
    ----------
    graph : dict
        {"nodes": [{"id","type","label","quotes","sessions_seen"}, ...],
         "edges": [{"source","target","relation","weight","sessions_seen"}, ...]}
        — the same shape as TypedKnowledgeGraph.to_dict().
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            height=400, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="No memory yet", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=16, color="#888"))],
        )
        return fig

    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], **n)
    for e in edges:
        if e["source"] in g and e["target"] in g:
            g.add_edge(e["source"], e["target"], **e)

    pos = nx.spring_layout(g, seed=42, k=1.1 / max(len(nodes) ** 0.5, 1))

    # ── Edge lines + arrowheads ─────────────────────────────────────────────
    edge_x, edge_y = [], []
    annotations = []
    edge_label_x, edge_label_y, edge_label_text = [], [], []

    for e in edges:
        if e["source"] not in pos or e["target"] not in pos:
            continue
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        annotations.append(dict(
            ax=x0, ay=y0, axref="x", ayref="y",
            x=x1, y=y1, xref="x", yref="y",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=max(0.5, min(2.5, e.get("weight", 0.5) * 2.5)),
            arrowcolor="#7f8c8d", opacity=0.7,
            standoff=14, startstandoff=14,
        ))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        edge_label_x.append(mx)
        edge_label_y.append(my)
        edge_label_text.append(e.get("relation", ""))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#bbb"), hoverinfo="none",
    )
    edge_label_trace = go.Scatter(
        x=edge_label_x, y=edge_label_y, mode="text",
        text=edge_label_text, textfont=dict(size=9, color="#666"),
        hoverinfo="none",
    )

    # ── Nodes ────────────────────────────────────────────────────────────────
    node_x = [pos[n["id"]][0] for n in nodes]
    node_y = [pos[n["id"]][1] for n in nodes]
    node_colors = [TYPE_COLORS.get(n["type"], DEFAULT_COLOR) for n in nodes]
    node_labels = [n["label"] for n in nodes]
    node_hover = [
        f"<b>{n['label']}</b><br>{n['type']}<br>"
        f"sessions: {', '.join(str(s) for s in n.get('sessions_seen', []))}<br>"
        + "<br>".join(f"&ldquo;{q}&rdquo;" for q in n.get("quotes", []))
        for n in nodes
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=26, color=node_colors, line=dict(width=1.5, color="#2c3e50")),
        text=node_labels, textposition="bottom center", textfont=dict(size=10, color="#2c3e50"),
        hovertext=node_hover, hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace])
    fig.update_layout(
        annotations=annotations,
        showlegend=False,
        height=650,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
    )
    return fig
