"""
Core unit tests for the simulation.

Run: python -m pytest tests/ -x -q
"""
import pytest
from config.formulation_config import DIRECT_EDGES, NODES, EDGE_MAP_TRIPLES
from simulation.graph import (
    EdgeGraph, from_schema, iter_edges, get_weight, set_weight,
    count_active, count_by_strength, active_edge_set,
)
from simulation.sampling import sample_cognitive_model, sample_session_state
from simulation.patient_store import format_session_history


# ── graph.py ──────────────────────────────────────────────────────────────────

def test_from_schema_covers_all_edges():
    g = from_schema(DIRECT_EDGES)
    for (p, c) in DIRECT_EDGES:
        assert p in g
        assert c in g[p]
        assert g[p][c] == 0.0


def test_iter_edges_round_trip():
    g = from_schema(DIRECT_EDGES)
    set_weight(g, "Triggers", "Memory", 0.8)
    pairs = {(p, c) for p, c, _ in iter_edges(g)}
    expected = set(DIRECT_EDGES.keys())
    assert pairs == expected


def test_get_weight_default():
    g = from_schema(DIRECT_EDGES)
    assert get_weight(g, "Triggers", "Memory") == 0.0
    assert get_weight(g, "Triggers", "Memory", default=0.5) == 0.0  # key exists, value is 0.0
    assert get_weight(g, "Triggers", "NONEXISTENT", default=99.0) == 99.0


def test_count_active():
    g = from_schema(DIRECT_EDGES)
    assert count_active(g) == 0
    set_weight(g, "Triggers", "Memory", 0.8)
    set_weight(g, "Memory", "Threat", 0.3)
    assert count_active(g) == 2


def test_count_by_strength():
    g = from_schema(DIRECT_EDGES)
    set_weight(g, "Triggers", "Memory", 0.8)    # strong
    set_weight(g, "Memory", "Threat", 0.5)      # moderate
    set_weight(g, "Threat", "Triggers", 0.2)    # weak (not counted)
    strong, moderate = count_by_strength(g)
    assert strong == 1
    assert moderate == 1


def test_active_edge_set():
    g = from_schema(DIRECT_EDGES)
    set_weight(g, "Triggers", "Memory", 0.8)
    set_weight(g, "Memory", "Threat", 0.0)
    active = active_edge_set(g)
    assert ("Triggers", "Memory") in active
    assert ("Memory", "Threat") not in active


# ── formulation_config.py ─────────────────────────────────────────────────────

def test_edge_map_triples_covers_all_direct_edges():
    triple_pairs = {(p, c) for p, c, _ in EDGE_MAP_TRIPLES}
    direct_pairs = set(DIRECT_EDGES.keys())
    assert triple_pairs == direct_pairs


def test_edge_map_triples_field_names():
    for p, c, field in EDGE_MAP_TRIPLES:
        expected = f"{p.replace(' ', '_')}__{c.replace(' ', '_')}"
        assert field == expected, f"Bad field for {p}→{c}: got {field!r}"


# ── sampling.py ───────────────────────────────────────────────────────────────

def test_sample_cognitive_model_structure():
    model = sample_cognitive_model(node_prob=1.0, edge_prob=1.0)
    assert "edges" in model
    assert "active_nodes" in model
    g = model["edges"]
    for (p, c) in DIRECT_EDGES:
        assert p in g and c in g[p]


def test_sample_session_state_respects_inactive_nodes():
    model = sample_cognitive_model(node_prob=1.0, edge_prob=1.0)
    agg_edges = model["edges"]
    active_nodes = model["active_nodes"]

    for _ in range(20):
        nodes, obs = sample_session_state(agg_edges, active_nodes)
        for p, children in obs.items():
            for c, w in children.items():
                if w > 0:
                    assert nodes.get(p), f"Edge {p}→{c} active but node {p} is inactive"
                    assert nodes.get(c), f"Edge {p}→{c} active but node {c} is inactive"


def test_sample_session_state_zero_graph_stays_zero():
    g = from_schema(DIRECT_EDGES)
    all_nodes = list(NODES.keys())
    _, obs = sample_session_state(g, all_nodes)
    assert count_active(obs) == 0


# ── patient_store.py ──────────────────────────────────────────────────────────

def test_format_session_history_empty():
    result = format_session_history([])
    assert "No prior sessions" in result


def test_format_session_history_with_data():
    summaries = [
        {
            "session_idx": 0,
            "summary": "Patient expressed avoidance.",
            "nodes_expressed": ["Triggers"],
            "edges_expressed": ["Triggers--Memory"],
            "graph": {
                "Triggers": {"Memory": {"score": 1, "explanation": "x", "quote": "y"}},
                "Memory": {"Threat": {"score": 0, "explanation": "", "quote": ""}},
            },
        }
    ]
    result = format_session_history(summaries)
    assert "Session 0" in result
    assert "Patient expressed avoidance." in result
    assert "Triggers → Memory" in result
    assert "Memory → Threat" not in result  # score=0 should be excluded


def test_format_session_history_multiple_sessions():
    summaries = [
        {"session_idx": i, "summary": f"Session {i} text.", "graph": {}} for i in range(3)
    ]
    result = format_session_history(summaries)
    assert "Session 0" in result
    assert "Session 1" in result
    assert "Session 2" in result


# ── scenario continuity across sessions ───────────────────────────────────────

def test_used_scenario_seeds_survive_a_session(tmp_path):
    """Session state is rebuilt from the patient store each session, so anything
    left only in session state is silently discarded — which is how the same
    scenario came up three sessions running."""
    from simulation.patient_store import load_patient_state, update_and_save

    initial = {"demographics": {}, "self_report": {}, "active_nodes": [],
               "agg_edges": {}}
    patient_state = load_patient_state(tmp_path, 0, initial)

    patient_state = update_and_save(
        tmp_path, 0, patient_state,
        {"conversation": [], "used_scenario_seeds": ["called_in"]}, 0)
    assert patient_state["used_scenario_seeds"] == ["called_in"]

    patient_state = update_and_save(
        tmp_path, 0, patient_state,
        {"conversation": [], "used_scenario_seeds": ["called_in", "detour"]}, 1)
    assert patient_state["used_scenario_seeds"] == ["called_in", "detour"]

    # and it survives a reload from disk, not just the in-memory handoff
    reloaded = load_patient_state(tmp_path, 0, initial)
    assert reloaded["used_scenario_seeds"] == ["called_in", "detour"]
