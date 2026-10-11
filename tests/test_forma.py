"""
Tests for the FORMA persona loader and the generated biographies.

Skipped unless data/forma/ has been populated:
    python scripts/fetch_forma_data.py
    python scripts/generate_biographies.py --personas 7,42,113
"""
import pytest

from config.formulation_config import DIRECT_EDGES
from simulation.activation import ActivationParams, from_self_report
from simulation.forma import (
    FORMA_ROOT, PATIENT_CONTEXTS, available_biographies, available_personas,
    load_biography, load_persona,
)
from simulation.graph import iter_edges
from agents.patient_toolbox import build_toolbox

pytestmark = pytest.mark.skipif(
    not (FORMA_ROOT / "personas").exists(),
    reason="FORMA data not fetched — run scripts/fetch_forma_data.py",
)

needs_biography = pytest.mark.skipif(
    not available_biographies(),
    reason="no biographies — run scripts/generate_biographies.py",
)

# Vignette-condition checks work for any persona; biography checks need a
# generated one, and only a handful exist while the pipeline is being brought up.
SOME_PERSONAS = [1, 7, 123, 500]


def _with_biography() -> list[int]:
    return available_biographies()[:3]


# ── the persona data ──────────────────────────────────────────────────────────

def test_all_500_personas_present():
    ids = available_personas()
    assert len(ids) == 500
    assert ids[0] == 1 and ids[-1] == 500


def test_load_persona_has_expected_keys():
    p = load_persona(SOME_PERSONAS[0], "no_formulation")
    assert set(p) == {"persona_id", "demographics", "self_report", "active_nodes",
                      "agg_edges", "biography", "patient_context"}


def test_edges_cover_the_full_schema():
    g = load_persona(SOME_PERSONAS[0], "no_formulation")["agg_edges"]
    assert {(p, c) for p, c, _ in iter_edges(g)} == set(DIRECT_EDGES)


def test_edge_weights_are_in_range():
    for pid in SOME_PERSONAS:
        for _, _, w in iter_edges(load_persona(pid, "no_formulation")["agg_edges"]):
            assert 0.0 <= w <= 1.0


def test_self_report_covers_exactly_the_active_nodes():
    for pid in SOME_PERSONAS:
        p = load_persona(pid, "no_formulation")
        assert set(p["self_report"]) == set(p["active_nodes"])


def test_inactive_nodes_have_no_edges():
    """An edge may only be non-zero when both endpoints are active."""
    for pid in SOME_PERSONAS:
        p = load_persona(pid, "no_formulation")
        active = set(p["active_nodes"])
        for parent, child, w in iter_edges(p["agg_edges"]):
            if w > 0:
                assert parent in active and child in active, \
                    f"persona {pid}: {parent}->{child}"


def test_load_is_deterministic():
    pid = SOME_PERSONAS[0]
    assert load_persona(pid, "no_formulation") == load_persona(pid, "no_formulation")


# ── patient context ───────────────────────────────────────────────────────────

def test_context_is_recorded():
    pid = SOME_PERSONAS[0]
    for condition in ("no_formulation", "full"):
        assert load_persona(pid, condition)["patient_context"] == condition


def test_vignette_conditions_differ_but_describe_one_patient():
    pid = SOME_PERSONAS[0]
    a = load_persona(pid, "no_formulation")
    b = load_persona(pid, "full")
    assert a["biography"] != b["biography"]
    assert a["agg_edges"] == b["agg_edges"]       # same patient, different telling


def test_unknown_context_rejected():
    with pytest.raises(ValueError):
        load_persona(SOME_PERSONAS[0], "made_up")


def test_missing_persona_raises():
    with pytest.raises(FileNotFoundError):
        load_persona(99999, "no_formulation")


# ── biographies ───────────────────────────────────────────────────────────────

@needs_biography
def test_biography_is_the_default_context():
    pid = _with_biography()[0]
    assert load_persona(pid)["patient_context"] == "biography"


@needs_biography
def test_biography_is_short():
    """A biography is identity and event — a vignette is ten times longer."""
    for pid in _with_biography():
        assert 40 <= len(load_biography(pid).split()) <= 200


@needs_biography
def test_biography_names_no_symptom():
    """The point of the biography: the patient does not arrive already
    narrating its own symptoms, so the mechanism has to supply them."""
    banned = [
        "nightmare", "flashback", "intrusive", "intrusion", "avoid", "avoided",
        "trigger", "startle", "hypervigil", "numb", "panic", "anxious",
        "anxiety", "insomnia", "sleep", "since then", "ever since",
        "no longer", "stopped going", "withdraw",
    ]
    for pid in _with_biography():
        low = load_biography(pid).lower()
        for word in banned:
            assert word not in low, f"persona {pid} biography leaks: {word!r}"


@needs_biography
def test_biography_is_far_shorter_than_the_vignette():
    pid = _with_biography()[0]
    bio = len(load_persona(pid, "biography")["biography"].split())
    vig = len(load_persona(pid, "no_formulation")["biography"].split())
    assert bio < vig / 2


@needs_biography
def test_missing_biography_raises_with_a_usable_message():
    missing = next(p for p in available_personas() if p not in available_biographies())
    with pytest.raises(KeyError, match="generate_biographies"):
        load_persona(missing, "biography")


# ── the loader feeds the mechanism ────────────────────────────────────────────

def test_persona_feeds_the_toolbox():
    p = load_persona(7, "no_formulation")
    activation = from_self_report(p["self_report"])
    tools = build_toolbox(activation, p["self_report"],
                          {n: True for n in p["active_nodes"]},
                          p["agg_edges"], ActivationParams())
    assert tools
    # Memory is inactive for persona 7, so that tool must not exist
    assert "recall_memory" not in {t.name for t in tools}


def test_every_context_produces_a_usable_patient():
    pid = _with_biography()[0] if available_biographies() else 7
    for condition in PATIENT_CONTEXTS:
        if condition == "biography" and pid not in available_biographies():
            continue
        p = load_persona(pid, condition)
        assert p["biography"].strip()
        assert p["self_report"]
