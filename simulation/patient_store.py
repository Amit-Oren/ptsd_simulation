import json
import logging
import uuid
from pathlib import Path

from config.formulation_config import DIRECT_EDGES, NODES
from simulation.graph import EdgeGraph, from_schema
from simulation.memory.store import load_memory_graph

logger = logging.getLogger(__name__)

_RECENT_TRANSCRIPTS_KEPT = 2


def format_session_history(session_summaries: list) -> str:
    """Build a readable context string from past session summaries for agent prompts.

    NOTE: kept for the inferred_graph/session_summaries bookkeeping this module still
    maintains, but never used to build the patient's prompt — this data is derived from
    the analyst's evaluation, which a real patient would never see. The patient's own
    cross-session memory is format_session_memory().
    """
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


def format_session_memory(session_topics: list, max_sessions: int = _RECENT_TRANSCRIPTS_KEPT) -> str:
    """What the patient remembers of earlier conversations.

    What was talked about, not what was said. Verbatim transcripts were tried
    first and killed the mechanism: a patient's own past words contain their
    symptoms in plain language, so from the second session onward the model had
    the whole profile back in its prompt and stopped calling tools entirely -
    zero calls in every session after the first. The same failure as handing it
    a clinical vignette, arriving through memory, and it compounds, since each
    session carries all the ones before it.

    Topics preserve the continuity that matters - these conversations are not
    standalone, and the patient knows they have spoken before - without
    re-supplying the symptom list the mechanism exists to produce.
    """
    if not session_topics:
        return "(No earlier conversations.)"
    lines = []
    for entry in session_topics[-max_sessions:]:
        gist = entry.get("gist") or "something that had happened"
        lines.append(f"  - {gist}")
    # The gists are present-tense clauses, so they are listed rather than
    # wrapped in "you talked about ...", which produces a tense mismatch.
    header = "You have messaged this chatbot before. What you talked about:"
    return header + "\n" + "\n".join(lines)


def _empty_inferred_graph() -> dict:
    """Empty accumulator: one list per node/edge, appended to each session."""
    edges: dict = {}
    for (p, c) in DIRECT_EDGES:
        edges.setdefault(p, {})[c] = []
    return {"nodes": {n: [] for n in NODES}, "edges": edges}


def _patient_dir(experiment_dir: Path, patient_idx: int) -> Path:
    return experiment_dir / "patients" / f"patient_{patient_idx}"


def _patient_path(experiment_dir: Path, patient_idx: int) -> Path:
    return _patient_dir(experiment_dir, patient_idx) / "state.json"


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
        state["used_scenario_events"] = stored.get("used_scenario_events", [])
        state["session_topics"]      = stored.get("session_topics", [])
        state["self_report"]       = stored.get("self_report", {})
        state["inferred_graph"]    = stored["inferred_graph"]
        state["session_summaries"] = stored["session_summaries"]
        state["session_count"]     = stored["session_count"]
        # memory_id is stable across sessions within this experiment_dir (resumed run);
        # fall back to a fresh id only if this state predates the memory_id fix.
        state["memory_id"] = stored.get("memory_id") or uuid.uuid4().hex[:8]
        logger.info("[patient_%d] loaded state — session_count=%d", patient_idx, state["session_count"])
    else:
        state["inferred_graph"]   = _empty_inferred_graph()
        state["session_summaries"] = []
        state["session_count"]    = 0
        state["recent_transcripts"] = []
        # patient_idx is only unique WITHIN this run (it's just position 0, 1, 2, ...) —
        # two separate `main.py` invocations both produce a "patient 0". Persistent
        # memory in data/memory/ needs a genuinely unique key, so each new patient gets
        # a real random id here, generated once and persisted alongside the rest of
        # this patient's state so it stays stable across sessions of the same run.
        state["memory_id"] = uuid.uuid4().hex[:8]

    # Memory graph is genuinely persistent state, not part of this run's output —
    # it lives in data/memory/, loaded/saved separately from patient_{idx}_state.json,
    # keyed by memory_id rather than patient_idx.
    state["memory_graph"] = load_memory_graph(state["memory_id"])

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

    recent_transcripts = patient_state.get("recent_transcripts", []) + [
        {"session_idx": session_idx, "conversation": session_state.get("conversation", [])}
    ]

    updated = dict(patient_state)
    updated["inferred_graph"]     = inferred
    updated["session_summaries"]  = patient_state.get("session_summaries", []) + [session_summary]
    updated["session_count"]      = patient_state.get("session_count", 0) + 1
    updated["recent_transcripts"] = recent_transcripts[-_RECENT_TRANSCRIPTS_KEPT:]
    updated["memory_graph"]       = session_state.get("memory_graph", patient_state.get("memory_graph"))
    updated["memory_id"]          = patient_state.get("memory_id")
    # Carried forward so a patient does not report the same kind of week twice:
    # session state is rebuilt from this dict each session, so anything left
    # only in session state is silently discarded.
    updated["used_scenario_events"] = session_state.get(
        "used_scenario_events", patient_state.get("used_scenario_events", []))

    scenario = session_state.get("scenario") or {}
    material = scenario.get("material") or {}
    # A short, still camera-only line - not the full account - so memory
    # carries continuity without re-supplying everything the scenario said.
    # event is always present in material; place is optional context.
    gist = (f"{material['event']} at {material['place']}"
            if material.get("place") else material.get("event"))
    updated["session_topics"] = patient_state.get("session_topics", []) + [{
        "session_idx": session_idx,
        "event_key":   scenario.get("event_key"),
        "gist":        gist,
    }]

    payload = {
        "patient_idx":       patient_idx,
        "memory_id":         updated["memory_id"],
        "session_count":     updated["session_count"],
        "demographics":      updated.get("demographics", {}),
        "self_report":       updated.get("self_report", {}),
        "active_nodes":      updated.get("active_nodes", []),
        "agg_edges":         updated.get("agg_edges", {}),
        "inferred_graph":    updated["inferred_graph"],
        "used_scenario_events": updated["used_scenario_events"],
        "session_topics":    updated["session_topics"],
        "session_summaries": updated["session_summaries"],
        "recent_transcripts": updated["recent_transcripts"],
    }
    path = _patient_path(experiment_dir, patient_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("[patient_%d] state saved — session %d complete, %d nodes / %d edges expressed",
                patient_idx, session_idx, len(expressed_nodes), len(expressed_edges))
    return updated
