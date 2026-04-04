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

_demographics_df = pd.read_csv(DATA_DIR / "demographics_labeled.csv").rename(columns={"Subnum": "id"})

def get_demographics(patient_id):
    """Return demographics dict for a patient ID."""
    row = _demographics_df[_demographics_df["id"] == patient_id]
    if row.empty:
        available = get_available_ids()
        raise ValueError(f"ID {patient_id!r} not found in demographics data. Available IDs: {available}")
    return {
        "age": int(row["Age"].iloc[0]),
        "gender": row["Gender"].iloc[0],
        "nationality": row["Nationality"].iloc[0],
        "profession": row["Profession"].iloc[0],
        "pcl5": int(row["PCL score_highest"].iloc[0]),
        "trauma_type": row["Trauma type"].iloc[0]
    }

def get_self_report(patient_id):
    """Return self-report dict for a patient ID."""
    row = _demographics_df[_demographics_df["id"] == patient_id]
    if row.empty:
        available = get_available_ids()
        raise ValueError(f"ID {patient_id!r} not found in demographics data. Available IDs: {available}")
    return {
        "trigger_items":                    row["Triggers_items"].iloc[0],
        "trigger_category":                 row["Triggers_category"].iloc[0],
        "trigger_subcategory":              row["Triggers_subcategory"].iloc[0],
        "appraisals_items_label":           row["Appraisals_items_label"].iloc[0],
        "appraisals_items_original":        row["Appraisals_items_original"].iloc[0],
        "memory_items_label":               row["Memory_items_label"].iloc[0],
        "memory_items_original":            row["Memory_items_original"].iloc[0],
        "threat_items_label":               row["Threat_items_label"].iloc[0],
        "threat_items_original":            row["Threat_items_original"].iloc[0],
        "threat_items_original_label":      row["Threat_items_original_label"].iloc[0],
        "threat_items_original_original":   row["Threat_items_original_original"].iloc[0],
        "strategies_items_label":           row["Strategies_items_label"].iloc[0],
        "strategies_items_original":        row["Strategies_items_original"].iloc[0],
        "avoidance_items_label":            row["Avoidance_items_label"].iloc[0],
        "avoidance_items_original":         row["Avoidance_items_original"].iloc[0],
    }

def get_available_ids():
    """Return list of all patient IDs in the CSV."""
    return _df["id"].unique().tolist()


def extract_aggregated_edges(patient_id):
    """Extract aggregated edges for the given patient ID.

    Returns:
        edges: DIRECT_EDGES dict populated with estimates from CSV
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

    return edges


def _split_items(val):
    if not val or isinstance(val, float):
        return []
    return [v.strip() for v in str(val).split('|') if v.strip()]


def _node_items(sr: dict) -> dict:
    return {
        "Triggers":               _split_items(sr["trigger_items"]),
        "Negative Appraisals":    _split_items(sr["appraisals_items_original"]),
        "Memory":                 _split_items(sr["memory_items_original"]),
        "Threat":                 _split_items(sr["threat_items_original"]),
        "Maladaptive Strategies": _split_items(sr["strategies_items_original"]) + _split_items(sr["avoidance_items_original"]),
    }


def get_aggregated_context(patient_id) -> dict:
    """Aggregated context: self-report items per node + aggregated edge strengths."""
    sr        = get_self_report(patient_id)
    agg       = extract_aggregated_edges(patient_id)
    ni        = _node_items(sr)

    return {
        "nodes": {node: {"items": items} for node, items in ni.items()},
        "edges": {
            edge: {
                "strength":     round(weight, 4),
                "parent_items": ni.get(edge[0], []),
                "child_items":  ni.get(edge[1], []),
            }
            for edge, weight in agg.items()
        },
    }


def get_session_context(patient_id, obs_index: int) -> dict:
    """Session context: self-report items per node + session active states."""
    sr                              = get_self_report(patient_id)
    obs_datetime, obs_nodes, obs_edges = get_session_components(patient_id, obs_index)
    ni                              = _node_items(sr)

    return {
        "obs_datetime": obs_datetime,
        "nodes": {
            node: {"items": items, "active": obs_nodes[node]}
            for node, items in ni.items()
        },
        "edges": {
            edge: {
                "active":       obs_edges[edge],
                "parent_items": ni.get(edge[0], []),
                "child_items":  ni.get(edge[1], []),
            }
            for edge in obs_edges
        },
    }


# ── EMA data (per-observation) ────────────────────────────────────────────────

def get_session_components(patient_id, index: int):
    """Extract node states and edge activations for one session by index.

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


