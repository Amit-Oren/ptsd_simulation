"""
Tests for the FORMA persona/vignette loader.

Skipped unless data/forma/ has been populated:
    python scripts/fetch_forma_data.py
"""
import pytest

from config.formulation_config import DIRECT_EDGES
from simulation.forma import FORMA_ROOT, available_personas, load_persona
from simulation.graph import iter_edges
from simulation.activation import from_self_report
from agents.patient_toolbox import build_toolbox
from simulation.activation import ActivationParams

pytestmark = pytest.mark.skipif(
    not (FORMA_ROOT / "personas").exists(),
    reason="FORMA data not fetched — run scripts/fetch_forma_data.py",
)


def test_all_500_personas_present():
    ids = available_personas()
    assert len(ids) == 500
    assert ids[0] == 1 and ids[-1] == 500


def test_load_persona_has_expected_keys():
    p = load_persona(1)
    assert set(p) == {"persona_id", "demographics", "self_report", "active_nodes",
                      "agg_edges", "vignette", "vignette_condition"}


def test_edges_cover_the_full_schema():
    g = load_persona(1)["agg_edges"]
    pairs = {(p, c) for p, c, _ in iter_edges(g)}
    assert pairs == set(DIRECT_EDGES)


def test_edge_weights_are_in_range():
    for pid in (1, 7, 250, 500):
        for _, _, w in iter_edges(load_persona(pid)["agg_edges"]):
            assert 0.0 <= w <= 1.0


def test_self_report_covers_exactly_the_active_nodes():
    for pid in (1, 7, 123, 500):
        p = load_persona(pid)
        assert set(p["self_report"]) == set(p["active_nodes"])


def test_inactive_nodes_have_no_edges():
    """An edge may only be non-zero when both endpoints are active."""
    for pid in (1, 7, 123, 500):
        p = load_persona(pid)
        active = set(p["active_nodes"])
        for parent, child, w in iter_edges(p["agg_edges"]):
            if w > 0:
                assert parent in active and child in active, f"persona {pid}: {parent}->{child}"


def test_load_is_deterministic():
    assert load_persona(42) == load_persona(42)


def test_vignette_condition_is_recorded():
    assert load_persona(1, "no_formulation")["vignette_condition"] == "no_formulation"
    assert load_persona(1, "full")["vignette_condition"] == "full"


def test_conditions_give_different_vignettes():
    a = load_persona(1, "no_formulation")
    b = load_persona(1, "full")
    assert a["vignette"] != b["vignette"]
    assert a["agg_edges"] == b["agg_edges"]       # same patient, different telling


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        load_persona(1, "made_up")


def test_missing_persona_raises():
    with pytest.raises(FileNotFoundError):
        load_persona(99999)


def test_persona_feeds_the_toolbox():
    """The loader's output is directly usable by the activation machinery."""
    p = load_persona(7)
    activation = from_self_report(p["self_report"])
    obs_nodes = {n: True for n in p["active_nodes"]}
    tools = build_toolbox(activation, p["self_report"], obs_nodes,
                          p["agg_edges"], ActivationParams())
    assert tools
    # Memory is inactive for persona 7, so that tool must not exist
    assert "recall_memory" not in {t.name for t in tools}
