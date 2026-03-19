import pandas as pd
from pathlib import Path
from config.formulation_config import NODES, DIRECT_EDGES, NODE_MAP, EDGE_MAP

DATA_DIR = Path(__file__).parent

# Map CSV short names → config full names
_NAME_MAP = {
    "Appraisals": "Negative Appraisals",
    "Strategies": "Maladaptive Strategies",
    "Memory":     "Memory",
    "Threat":     "Threat",
    "Triggers":   "Triggers",
}

_df = pd.read_csv(DATA_DIR / "aggregated_ema_data.csv")

_ema_df = pd.read_csv(DATA_DIR / "ema_data.csv", parse_dates=["datetime"])


def get_available_ids():
    """Return list of all patient IDs in the CSV."""
    return _df["id"].unique().tolist()


def prepare_patient_components(patient_id):
    """Extract nodes and edges for the given patient ID.

    Returns:
        (patient_id, nodes, edges)
        - nodes: dict with default 0 values (no node-level data in CSV)
        - edges: DIRECT_EDGES dict populated with estimates from CSV
    """
    rows = _df[_df["id"] == patient_id]
    if rows.empty:
        available = get_available_ids()
        raise ValueError(f"ID {patient_id!r} not found. Available IDs: {available}")

    edges = DIRECT_EDGES.copy()
    for _, row in rows.iterrows():
        parent = _NAME_MAP[row["Parent"]]
        child = _NAME_MAP[row["Child"]]
        key = (parent, child)
        if key in edges:
            edges[key] = row["estimate"]

    nodes = NODES.copy()

    return nodes, edges


# ── EMA data (per-observation) ────────────────────────────────────────────────

def get_observation_components(patient_id, index: int):
    """Extract node states and edge activations for one observation by index.

    Returns:
        (datetime, nodes, edges)
    """
    patient_rows = _ema_df[_ema_df["id"] == patient_id].reset_index(drop=True)
    if patient_rows.empty:
        raise ValueError(f"ID {patient_id!r} not found in EMA data.")
    if index >= len(patient_rows):
        raise IndexError(f"Index {index} out of range — patient {patient_id!r} has {len(patient_rows)} observations.")

    row = patient_rows.iloc[index]

    nodes = NODES.copy()
    for col, node in NODE_MAP.items():
        if col in row.index:
            nodes[node] = row[col]

    edges = DIRECT_EDGES.copy()
    for col, edge in EDGE_MAP.items():
        if col in row.index:
            edges[edge] = row[col]

    return row["datetime"], nodes, edges
