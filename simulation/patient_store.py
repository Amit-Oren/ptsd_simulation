import json
import logging
from pathlib import Path

from config.formulation_config import DIRECT_EDGES, NODES
from simulation.graph import EdgeGraph, from_schema

logger = logging.getLogger(__name__)


def format_session_history(session_summaries: list) -> str:
    """Build a readable context string from past session summaries for agent prompts."""
    if not session_summaries:
        return "(No prior sessions.)"
    lines = []
    for s in session_summaries:
        idx = s["session_idx"]
        lines.append(f"Session {idx}:")
        if s.get("summary"):
            lines.append(f"  {s['summary']}")
        active = [
            f"  - {p} → {c}"
            for p, children in s.get("graph", {}).items()
            for c, data in children.items()
            if data.get("score", 0) == 1
        ]
        if active:
            lines.append("  Active cognitive edges:")
            lines.extend(active)
        else:
            lines.append("  Active cognitive edges: none observed.")
        lines.append("")
    return "\n".join(lines).strip()


def _empty_inferred_graph() -> dict:
    """Empty accumulator: one list per node/edge, appended to each session."""
    edges: dict = {}
    for (p, c) in DIRECT_EDGES:
        edges.setdefault(p, {})[c] = []
    return {"nodes": {n: [] for n in NODES}, "edges": edges}


def _patient_path(experiment_dir: Path, patient_idx: int) -> Path:
    return experiment_dir / f"patient_{patient_idx}_state.json"


def load_patient_state(experiment_dir: Path, patient_idx: int, initial_state: dict) -> dict:
    """Return patient state for the session loop.

    On first session: wraps initial_state with empty accumulators.
    On subsequent sessions: restores evolved agg_edges and accumulated inferred_graph from disk.
    """
    state = dict(initial_state)
    path = _patient_path(experiment_dir, patient_idx)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        state["agg_edges"]         = stored["agg_edges"]
        state["vignette_analysis"] = stored.get("vignette_analysis", {})
        state["self_report"]       = stored.get("self_report", {})
        state["inferred_graph"]    = stored["inferred_graph"]
        state["session_summaries"] = stored["session_summaries"]
        state["session_count"]     = stored["session_count"]
        logger.info("[patient_%d] loaded state — session_count=%d", patient_idx, state["session_count"])
    else:
        state["inferred_graph"]   = _empty_inferred_graph()
        state["session_summaries"] = []
        state["session_count"]    = 0

    return state


def update_and_save(experiment_dir: Path, patient_idx: int,
                    patient_state: dict, session_state: dict,
                    session_idx: int) -> dict:
    """Append session analysis to the longitudinal graph and persist to disk.

    Returns the updated patient_state (used as baseline for the next session).
    """
    analysis = session_state.get("analysis")
    inferred = {
        "nodes": {k: list(v) for k, v in patient_state["inferred_graph"]["nodes"].items()},
        "edges": {p: {c: list(scores) for c, scores in children.items()}
                  for p, children in patient_state["inferred_graph"]["edges"].items()},
    }

    expressed_nodes: list[str] = []
    expressed_edges: list[str] = []

    if isinstance(analysis, dict):
        for node, data in analysis.get("nodes", {}).items():
            if node in inferred["nodes"]:
                inferred["nodes"][node].append(data["score"])
                if data["score"] == 1:
                    expressed_nodes.append(node)

        for p, children in analysis.get("edges", {}).items():
            for c, data in children.items():
                if p in inferred["edges"] and c in inferred["edges"][p]:
                    inferred["edges"][p][c].append(data["score"])
                    if data["score"] == 1:
                        expressed_edges.append(f"{p}--{c}")

    session_summary = {
        "session_idx":     session_idx,
        "summary":         analysis.get("summary", "") if isinstance(analysis, dict) else "",
        "nodes_expressed": expressed_nodes,
        "edges_expressed": expressed_edges,
        "graph":           analysis.get("edges", {}) if isinstance(analysis, dict) else {},
    }

    updated = dict(patient_state)
    updated["inferred_graph"]    = inferred
    updated["session_summaries"] = patient_state.get("session_summaries", []) + [session_summary]
    updated["session_count"]     = patient_state.get("session_count", 0) + 1

    payload = {
        "patient_idx":       patient_idx,
        "session_count":     updated["session_count"],
        "demographics":      updated.get("demographics", {}),
        "self_report":       updated.get("self_report", {}),
        "active_nodes":      updated.get("active_nodes", []),
        "agg_edges":         updated.get("agg_edges", {}),
        "vignette_analysis": updated.get("vignette_analysis", {}),
        "inferred_graph":    updated["inferred_graph"],
        "session_summaries": updated["session_summaries"],
    }
    path = _patient_path(experiment_dir, patient_idx)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("[patient_%d] state saved — session %d complete, %d nodes / %d edges expressed",
                patient_idx, session_idx, len(expressed_nodes), len(expressed_edges))
    return updated
