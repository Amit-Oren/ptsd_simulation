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


def _edges_to_graph(agg_edges: dict) -> EdgeGraph:
    """FORMA stores edges as {"Triggers -- Memory": 0.51}; we use nested dicts."""
    graph = from_schema(DIRECT_EDGES)
    for key, weight in agg_edges.items():
        parent, _, child = key.partition(" -- ")
        if (parent, child) in DIRECT_EDGES:
            set_weight(graph, parent, child, float(weight))
    return graph


def available_personas(root: Path = FORMA_ROOT) -> list[int]:
    """Sorted persona ids present on disk."""
    return sorted(int(p.stem.split("_")[1])
                  for p in (root / "personas").glob("persona_*.json"))


def load_persona(persona_id: int, condition: str = "no_formulation",
                 root: Path = FORMA_ROOT) -> dict:
    """One patient, in the shape the session pipeline expects.

    Returns demographics, self_report, active_nodes, agg_edges and vignette —
    the same keys the old sampling-plus-crafting phase produced, so nothing
    downstream needs to know where the patient came from.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")

    persona_path = root / "personas" / f"persona_{persona_id}.json"
    if not persona_path.exists():
        raise FileNotFoundError(
            f"{persona_path} not found — run scripts/fetch_forma_data.py first")
    persona = json.loads(persona_path.read_text(encoding="utf-8"))

    vignette_path = root / "vignettes" / condition / f"experiment_{persona_id}.json"
    if not vignette_path.exists():
        raise FileNotFoundError(
            f"{vignette_path} not found — fetch the {condition!r} condition")
    vignette = json.loads(vignette_path.read_text(encoding="utf-8"))

    return {
        "persona_id":        persona_id,
        "demographics":      persona["demographics"],
        "self_report":       persona["self_report"],
        "active_nodes":      persona["active_nodes"],
        "agg_edges":         _edges_to_graph(persona["agg_edges"]),
        "vignette":          vignette["vignette"],
        "vignette_condition": condition,
    }
