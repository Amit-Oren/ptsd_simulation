"""
Load personas and vignettes from the FORMA dataset.

FORMA (github.com/Amit-Oren/FORMA) generated and clinician-validated the 500
personas this simulation uses, so building a patient costs no LLM calls and
`persona_id` is the seed — the same id always yields the same patient.

Two vignette conditions ship with the data:

  no_formulation  written without the cognitive graph, so no causal link is
                  ever narrated. This is what the patient agent reads: the
                  edges are what the analyst has to recover, and a vignette
                  that states them hands over the answer.
  full            written with the graph. An ablation arm only.

Run scripts/fetch_forma_data.py once to populate data/forma/.
"""
import json
from pathlib import Path

from config.formulation_config import DIRECT_EDGES
from simulation.graph import EdgeGraph, from_schema, set_weight

FORMA_ROOT = Path("data/forma")
CONDITIONS = ("no_formulation", "full")

# What the patient agent is given as its sense of itself. A vignette is a
# clinical document and narrates the person's symptoms, so a patient handed one
# has no reason to call a tool - it already knows everything a tool would tell
# it. The biography says only who they are and what happened, which is all a
# real person has; the symptoms then have to surface through the mechanism.
# The vignette conditions are kept as comparison arms.
PATIENT_CONTEXTS = ("biography", "no_formulation", "full")


def _edges_to_graph(agg_edges: dict) -> EdgeGraph:
    """FORMA stores edges as {"Triggers -- Memory": 0.51}; we use nested dicts."""
    graph = from_schema(DIRECT_EDGES)
    for key, weight in agg_edges.items():
        parent, _, child = key.partition(" -- ")
        if (parent, child) in DIRECT_EDGES:
            set_weight(graph, parent, child, float(weight))
    return graph


def load_biography(persona_id: int, root: Path = FORMA_ROOT) -> str:
    """Identity and event only - see scripts/generate_biographies.py."""
    path = root / "biographies.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/generate_biographies.py first")
    book = json.loads(path.read_text(encoding="utf-8"))
    record = book.get(str(persona_id))
    if record is None:
        raise KeyError(
            f"no biography for persona {persona_id} - generate it with "
            f"scripts/generate_biographies.py --personas {persona_id}")
    return record["text"]


def available_biographies(root: Path = FORMA_ROOT) -> list[int]:
    path = root / "biographies.json"
    if not path.exists():
        return []
    return sorted(int(k) for k in json.loads(path.read_text(encoding="utf-8")))


def available_personas(root: Path = FORMA_ROOT) -> list[int]:
    """Sorted persona ids present on disk."""
    return sorted(int(p.stem.split("_")[1])
                  for p in (root / "personas").glob("persona_*.json"))


def load_persona(persona_id: int, patient_context: str = "biography",
                 root: Path = FORMA_ROOT) -> dict:
    """One patient, in the shape the session pipeline expects.

    `patient_context` decides what the patient agent reads about itself:
    "biography" gives identity and event only; the two vignette conditions are
    ablation arms that progressively leak more (no_formulation leaks the
    sampled self-report items, full leaks those and the causal links).

    Returns demographics, self_report, active_nodes, agg_edges and the chosen
    context text, so nothing downstream needs to know where the patient came
    from.
    """
    if patient_context not in PATIENT_CONTEXTS:
        raise ValueError(
            f"patient_context must be one of {PATIENT_CONTEXTS}, got {patient_context!r}")

    persona_path = root / "personas" / f"persona_{persona_id}.json"
    if not persona_path.exists():
        raise FileNotFoundError(
            f"{persona_path} not found - run scripts/fetch_forma_data.py first")
    persona = json.loads(persona_path.read_text(encoding="utf-8"))

    if patient_context == "biography":
        context = load_biography(persona_id, root)
    else:
        vignette_path = (root / "vignettes" / patient_context /
                         f"experiment_{persona_id}.json")
        if not vignette_path.exists():
            raise FileNotFoundError(
                f"{vignette_path} not found - fetch the {patient_context!r} condition")
        context = json.loads(vignette_path.read_text(encoding="utf-8"))["vignette"]

    return {
        "persona_id":      persona_id,
        "demographics":    persona["demographics"],
        "self_report":     persona["self_report"],
        "active_nodes":    persona["active_nodes"],
        "agg_edges":       _edges_to_graph(persona["agg_edges"]),
        "biography":       context,
        "patient_context": patient_context,
    }
