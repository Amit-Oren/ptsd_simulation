
""" Elhers and clark PTSD formulation """

from typing import List, Dict

# ============================================================
# UndirectedGraph - Nodes and Edges
# ============================================================

UNDIRECTED_EDGES: dict = {
    ("Triggers", "Strategies"): 0,
    ("Triggers", "Threat"): 0,
    ("Triggers", "Memory"): 0,
    ("Triggers", "Negative Appraisals"): 0,
    ("Negative Appraisals", "Strategies"): 0,
    ("Negative Appraisals", "Threat"): 0,
    ("Negative Appraisals", "Memory"): 0,
    ("Memory", "Strategies"): 0,
    ("Memory", "Threat"): 0,
    ("Threat", "Strategies"): 0,
}

# ============================================================
# Nodes
# ============================================================

NODES: dict = {
    "Triggers": 0,
    "Negative Appraisals": 0,
    "Memory": 0,
    "Threat": 0,
    "Maladaptive Strategies": 0,
}

# ============================================================
# DirectedGraph - Nodes and Edges
# ============================================================

DIRECT_EDGES: dict = {
    ("Triggers", "Maladaptive Strategies"): 0,
    ("Triggers", "Threat"): 0,
    ("Triggers", "Memory"): 0,
    ("Triggers", "Negative Appraisals"): 0,
    ("Negative Appraisals", "Maladaptive Strategies"): 0,
    ("Negative Appraisals", "Threat"): 0,
    ("Negative Appraisals", "Memory"): 0,
    ("Negative Appraisals", "Triggers"): 0,
    ("Memory", "Maladaptive Strategies"): 0,
    ("Memory", "Threat"): 0,
    ("Memory", "Negative Appraisals"): 0,
    ("Memory", "Triggers"): 0,
    ("Threat", "Maladaptive Strategies"): 0,
    ("Threat", "Memory"): 0,
    ("Threat", "Negative Appraisals"): 0,
    ("Threat", "Triggers"): 0,
    ("Maladaptive Strategies", "Memory"): 0,
    ("Maladaptive Strategies", "Negative Appraisals"): 0,
}

# ============================================================
# Mappings from Excel column names to formulation keys
# ============================================================

NODE_MAP: dict = {
    "n_Triggers":   "Triggers",
    "n_Threat":     "Threat",
    "n_Memory":     "Memory",
    "n_Appraisals": "Negative Appraisals",
    "n_Strategies": "Maladaptive Strategies",
}

EDGE_MAP: dict = {
    "e_Tr_to_Str":  ("Triggers", "Maladaptive Strategies"),
    "e_Tr_to_Thr":  ("Triggers", "Threat"),
    "e_Tr_to_Mem":  ("Triggers", "Memory"),
    "e_Tr_to_App":  ("Triggers", "Negative Appraisals"),
    "e_App_to_Str": ("Negative Appraisals", "Maladaptive Strategies"),
    "e_App_to_Thr": ("Negative Appraisals", "Threat"),
    "e_App_to_Mem": ("Negative Appraisals", "Memory"),
    "e_App_to_Tr":  ("Negative Appraisals", "Triggers"),
    "e_Mem_to_Str": ("Memory", "Maladaptive Strategies"),
    "e_Mem_to_Thr": ("Memory", "Threat"),
    "e_Mem_to_App": ("Memory", "Negative Appraisals"),
    "e_Mem_to_Tr":  ("Memory", "Triggers"),
    "e_Thr_to_Str": ("Threat", "Maladaptive Strategies"),
    "e_Thr_to_Mem": ("Threat", "Memory"),
    "e_Thr_to_App": ("Threat", "Negative Appraisals"),
    "e_Thr_to_Tr":  ("Threat", "Triggers"),
}