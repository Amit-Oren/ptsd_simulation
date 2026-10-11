"""
Unit tests for the memory engine (simulation/memory/).

Run: python -m pytest tests/ -x -q
"""
from simulation.memory.kg import TypedKnowledgeGraph
from simulation.memory.render import build_interviewer_memory_context
from simulation.patient_store import format_session_memory


# ── kg.py — node merging ───────────────────────────────────────────────────────

def test_add_or_merge_node_creates_new():
    g = TypedKnowledgeGraph()
    node_id = g.add_or_merge_node(type_="Triggers", label="loud noises", quote="q1", session_idx=0)
    assert node_id in g.nodes
    node = g.nodes[node_id]
    assert node.type == "Triggers"
    assert node.label == "loud noises"
    assert node.quotes == ["q1"]
    assert node.sessions_seen == [0]


def test_add_or_merge_node_exact_label_match_merges():
    g = TypedKnowledgeGraph()
    id1 = g.add_or_merge_node(type_="Triggers", label="loud noises", quote="q1", session_idx=0)
    id2 = g.add_or_merge_node(type_="Triggers", label="Loud Noises", quote="q2", session_idx=1)
    assert id1 == id2
    assert len(g.nodes) == 1
    node = g.nodes[id1]
    assert node.sessions_seen == [0, 1]
    assert node.quotes == ["q1", "q2"]


def test_add_or_merge_node_same_label_different_type_does_not_merge():
    g = TypedKnowledgeGraph()
    id1 = g.add_or_merge_node(type_="Triggers", label="avoidance", quote="q1", session_idx=0)
    id2 = g.add_or_merge_node(type_="Maladaptive Strategies", label="avoidance", quote="q2", session_idx=0)
    assert id1 != id2
    assert len(g.nodes) == 2


def test_add_or_merge_node_explicit_existing_id_merges():
    g = TypedKnowledgeGraph()
    id1 = g.add_or_merge_node(type_="Memory", label="the crash", quote="q1", session_idx=0)
    id2 = g.add_or_merge_node(type_="Memory", label="a completely different phrasing",
                               quote="q2", session_idx=1, existing_id=id1)
    assert id1 == id2
    assert len(g.nodes) == 1


def test_add_or_merge_node_invalid_existing_id_falls_back():
    g = TypedKnowledgeGraph()
    node_id = g.add_or_merge_node(type_="Memory", label="the crash", quote="q1",
                                   session_idx=0, existing_id="not_a_real_id")
    assert node_id in g.nodes
    assert len(g.nodes) == 1


def test_node_quotes_capped_at_three_most_recent():
    g = TypedKnowledgeGraph()
    node_id = None
    for i in range(5):
        node_id = g.add_or_merge_node(type_="Threat", label="hypervigilance",
                                       quote=f"q{i}", session_idx=i)
    node = g.nodes[node_id]
    assert node.quotes == ["q2", "q3", "q4"]
    assert node.sessions_seen == [0, 1, 2, 3, 4]


# ── kg.py — edge merging ───────────────────────────────────────────────────────

def test_add_or_merge_edge_creates_new():
    g = TypedKnowledgeGraph()
    g.add_or_merge_edge("a", "b", "reactivates", 0.7, session_idx=0)
    edge = g.edges[("a", "b")]
    assert edge.relation == "reactivates"
    assert edge.weight == 0.7
    assert edge.sessions_seen == [0]


def test_add_or_merge_edge_running_average_weight():
    g = TypedKnowledgeGraph()
    g.add_or_merge_edge("a", "b", "reactivates", 1.0, session_idx=0)
    g.add_or_merge_edge("a", "b", "still reactivates", 0.5, session_idx=1)
    edge = g.edges[("a", "b")]
    assert edge.weight == 0.75  # (1.0 + 0.5) / 2
    assert edge.relation == "still reactivates"  # most recent wording wins
    assert edge.sessions_seen == [0, 1]


def test_add_or_merge_edge_direction_matters():
    g = TypedKnowledgeGraph()
    g.add_or_merge_edge("a", "b", "reactivates", 0.7, session_idx=0)
    g.add_or_merge_edge("b", "a", "amplifies", 0.4, session_idx=0)
    assert len(g.edges) == 2
    assert ("a", "b") in g.edges
    assert ("b", "a") in g.edges


# ── kg.py — ingest (entity/relation extraction merge) ──────────────────────────

def test_ingest_full_extraction_creates_nodes_and_edges():
    g = TypedKnowledgeGraph()
    extraction = {
        "entities": [
            {"temp_id": "e1", "existing_id": None, "type": "Triggers", "label": "loud noises", "quote": "q1"},
            {"temp_id": "e2", "existing_id": None, "type": "Memory", "label": "car accident", "quote": "q2"},
        ],
        "relations": [
            {"source_temp_id": "e1", "target_temp_id": "e2", "relation": "reactivates", "weight": 7, "quote": "q3"},
        ],
    }
    g.ingest(extraction, session_idx=0)
    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    edge = next(iter(g.edges.values()))
    assert edge.relation == "reactivates"
    assert edge.weight == 0.7  # 7/10


def test_ingest_reuses_existing_id_across_sessions():
    g = TypedKnowledgeGraph()
    g.ingest({"entities": [{"temp_id": "e1", "existing_id": None, "type": "Triggers",
                            "label": "loud noises", "quote": "q1"}], "relations": []}, session_idx=0)
    existing_id = next(iter(g.nodes.keys()))

    g.ingest({"entities": [{"temp_id": "e1", "existing_id": existing_id, "type": "Triggers",
                            "label": "loud noises again", "quote": "q2"}], "relations": []}, session_idx=1)
    assert len(g.nodes) == 1
    node = g.nodes[existing_id]
    assert node.sessions_seen == [0, 1]


def test_ingest_skips_relation_with_unknown_temp_id():
    g = TypedKnowledgeGraph()
    extraction = {
        "entities": [{"temp_id": "e1", "existing_id": None, "type": "Triggers", "label": "x", "quote": "q1"}],
        "relations": [{"source_temp_id": "e1", "target_temp_id": "does_not_exist",
                       "relation": "leads to", "weight": 5, "quote": "q2"}],
    }
    g.ingest(extraction, session_idx=0)  # must not raise
    assert len(g.nodes) == 1
    assert len(g.edges) == 0


# ── kg.py — serialization ──────────────────────────────────────────────────────

def test_to_dict_from_dict_round_trip():
    g = TypedKnowledgeGraph()
    g.ingest({
        "entities": [
            {"temp_id": "e1", "existing_id": None, "type": "Triggers", "label": "loud noises", "quote": "q1"},
            {"temp_id": "e2", "existing_id": None, "type": "Memory", "label": "car accident", "quote": "q2"},
        ],
        "relations": [
            {"source_temp_id": "e1", "target_temp_id": "e2", "relation": "reactivates", "weight": 8, "quote": "q3"},
        ],
    }, session_idx=0)

    restored = TypedKnowledgeGraph.from_dict(g.to_dict())
    assert set(restored.nodes.keys()) == set(g.nodes.keys())
    assert set(restored.edges.keys()) == set(g.edges.keys())
    for node_id, node in g.nodes.items():
        assert restored.nodes[node_id].label == node.label
        assert restored.nodes[node_id].type == node.type
    for key, edge in g.edges.items():
        assert restored.edges[key].relation == edge.relation
        assert restored.edges[key].weight == edge.weight


def test_is_empty():
    g = TypedKnowledgeGraph()
    assert g.is_empty()
    g.add_or_merge_node(type_="Triggers", label="x", quote="", session_idx=0)
    assert not g.is_empty()


# ── kg.py — coreference rendering ───────────────────────────────────────────────

def test_render_for_coreference_empty():
    g = TypedKnowledgeGraph()
    result = g.render_for_coreference()
    assert "no existing entities" in result


def test_render_for_coreference_lists_ids_and_labels():
    g = TypedKnowledgeGraph()
    node_id = g.add_or_merge_node(type_="Triggers", label="loud noises", quote="q1", session_idx=0)
    result = g.render_for_coreference()
    assert "Triggers:" in result
    assert node_id in result
    assert "loud noises" in result


# ── render.py — interviewer-facing rendering ────────────────────────────────────

def test_build_interviewer_memory_context_empty_graph():
    result = build_interviewer_memory_context(None)
    assert "first time meeting" in result
    result_empty = build_interviewer_memory_context(TypedKnowledgeGraph())
    assert "first time meeting" in result_empty


def test_build_interviewer_memory_context_groups_by_type_and_shows_edges():
    g = TypedKnowledgeGraph()
    g.ingest({
        "entities": [
            {"temp_id": "e1", "existing_id": None, "type": "Triggers", "label": "loud noises", "quote": "q1"},
            {"temp_id": "e2", "existing_id": None, "type": "Memory", "label": "car accident", "quote": "q2"},
        ],
        "relations": [
            {"source_temp_id": "e1", "target_temp_id": "e2", "relation": "reactivates", "weight": 8, "quote": "q3"},
        ],
    }, session_idx=2)

    result = build_interviewer_memory_context(g)
    assert "Triggers:" in result
    assert "Memory:" in result
    assert "loud noises" in result
    assert "car accident" in result
    assert "sessions: 2" in result
    assert "loud noises --reactivates--> car accident" in result
    # passive rendering — no directive/suggestion language leaking into the text
    assert "ask about" not in result.lower()
    assert "consider" not in result.lower()


# ── patient_store.py — the patient's cross-session memory ─────────────────────
# Topics, not transcripts. Verbatim past conversations contain the patient's
# symptoms in plain language, so feeding them back killed tool use outright:
# zero calls in every session after the first.

def test_session_memory_empty():
    assert "No earlier conversations" in format_session_memory([])


def test_session_memory_caps_to_last_n_sessions():
    topics = [{"session_idx": i, "seed_id": f"seed_{i}", "gist": f"thing {i}"}
              for i in range(4)]
    result = format_session_memory(topics, max_sessions=2)
    assert "thing 0" not in result
    assert "thing 1" not in result
    assert "thing 2" in result
    assert "thing 3" in result


def test_session_memory_names_the_topic():
    topics = [{"session_idx": 0, "seed_id": "called_in",
               "gist": "someone with authority over you asked you to come and speak"}]
    result = format_session_memory(topics)
    assert "asked you to come and speak" in result


def test_session_memory_carries_no_transcript():
    """What was talked about, never what was said."""
    topics = [{"session_idx": 0, "seed_id": "called_in", "gist": "a meeting"}]
    result = format_session_memory(topics)
    assert "CLIENT:" not in result
    assert "INTERVIEWER:" not in result


def test_session_memory_survives_a_missing_gist():
    result = format_session_memory([{"session_idx": 0, "seed_id": "x", "gist": None}])
    assert "something that had happened" in result
