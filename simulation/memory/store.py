"""Persistence for the memory engine.

Memory is genuinely persistent state, not a per-run output artifact — it
lives in data/memory/, separate from data/output/<experiment>/, and
survives across separate `main.py` invocations. Each run additionally
copies the memory state it used into its own output folder, purely as a
reproducibility record.

Keyed by memory_id, a random id generated once per patient (see
simulation/patient_store.py::load_patient_state) — not by patient_idx.
patient_idx is only a position within one run (0, 1, 2, ...); two separate
`main.py` invocations both produce a "patient 0", so using it directly as
the persistent-memory key would make unrelated patients from different
runs silently collide and overwrite each other's memory.
"""
import json
import logging
from pathlib import Path

from simulation.memory.kg import TypedKnowledgeGraph

logger = logging.getLogger(__name__)

MEMORY_ROOT = Path("data/memory")


def _patient_dir(memory_id: str) -> Path:
    d = MEMORY_ROOT / "patients" / f"patient_{memory_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_memory_graph(memory_id: str) -> TypedKnowledgeGraph:
    path = _patient_dir(memory_id) / "graph.json"
    if not path.exists():
        return TypedKnowledgeGraph()
    with open(path, "r", encoding="utf-8") as f:
        return TypedKnowledgeGraph.from_dict(json.load(f))


def save_memory_graph(memory_id: str, graph: TypedKnowledgeGraph, session_idx: int) -> None:
    patient_dir = _patient_dir(memory_id)
    data = graph.to_dict()

    with open(patient_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    snapshots_dir = patient_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    with open(snapshots_dir / f"session_{session_idx}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("[memory_%s] memory graph saved — %d nodes, %d edges (session %d)",
                memory_id, len(graph.nodes), len(graph.edges), session_idx)


def copy_memory_snapshot_to_output(patient_idx: int, graph: TypedKnowledgeGraph, experiment_dir: Path) -> None:
    """Copy the memory state used into this specific run's output, for a reproducibility
    record — this one is fine keyed by patient_idx, since it's scoped to one experiment_dir."""
    path = experiment_dir / "patients" / f"patient_{patient_idx}" / "memory_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
