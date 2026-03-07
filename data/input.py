import pandas as pd
from pathlib import Path
from config.formulation_config import NODES, DIRECT_EDGES, NODE_MAP, EDGE_MAP

DATA_DIR = Path(__file__).parent

df = pd.read_excel(DATA_DIR / "MSN data processed.xlsx")

def get_available_subnums():
    return df["Subnum"].unique().tolist()

def get_patient_rows(subnum):
    """Return available rows for a subnum as a list of dicts with index, Day, and Time."""
    patient_data = df[df["Subnum"].astype(str) == str(subnum)].reset_index(drop=True)
    return [{"row_index": i, "Day": row["Day"], "Time": row["Time"]}
            for i, row in patient_data.iterrows()]

def sample_patients(num_patients):
    patient_subject = df.sample(n=num_patients)["Subnum"]
    return patient_subject

def prepare_patient_components(subnum, row_index=None):
    """Extract id, schedule, nodes, and edges for the given subnum.
    If row_index is None, a random row is sampled."""
    patient_data = df[df["Subnum"].astype(str) == str(subnum)].reset_index(drop=True)

    if patient_data.empty:
        available = df["Subnum"].unique().tolist()
        raise ValueError(f"Subnum {subnum!r} not found. Available subnums: {available}")

    row = patient_data.iloc[row_index] if row_index is not None else patient_data.sample(1).iloc[0]

    nodes = NODES.copy()
    for col, node in NODE_MAP.items():
        nodes[node] = row[col]

    edges = DIRECT_EDGES.copy()
    for col, edge in EDGE_MAP.items():
        edges[edge] = row[col]

    schedule = {row["Day"]: [row["Time"]]}

    return subnum, schedule, nodes, edges

