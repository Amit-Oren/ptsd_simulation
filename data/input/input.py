import random
from config.formulation_config import DIRECT_EDGES, NODES
from simulation.graph import EdgeGraph, from_schema, set_weight
from config.demographics import (AGE, GENDER, ETHNICITY, OCCUPATION, TRAUMA_TYPE, PCL5,
                                  RELATIONSHIP_WEIGHTS_BY_AGE)
from config.self_report import (TRIGGERS, NEGATIVE_APPRAISALS, MEMORY,
                                 THREAT, MALADAPTIVE_STRATEGIES,
                                 DISTRACTION_STRATEGIES, AVOIDANCE_STRATEGIES)

_STRATEGIES_POOL = {**MALADAPTIVE_STRATEGIES, **DISTRACTION_STRATEGIES, **AVOIDANCE_STRATEGIES}

_NODE_POOLS: dict[str, dict] = {
    "Triggers":               TRIGGERS,
    "Negative Appraisals":    NEGATIVE_APPRAISALS,
    "Memory":                 MEMORY,
    "Threat":                 THREAT,
    "Maladaptive Strategies": _STRATEGIES_POOL,
}

_ALL_NODES = list(dict.fromkeys(n for edge in DIRECT_EDGES for n in edge))


def _weights_for_age(age: int, table: list[tuple]) -> dict:
    for (lo, hi), weights in table:
        if lo <= age <= hi:
            return weights
    return table[-1][1]


def _sample_relationship(age: int) -> str:
    weights = _weights_for_age(age, RELATIONSHIP_WEIGHTS_BY_AGE)
    return random.choices(list(weights.keys()), weights=list(weights.values()))[0]


def sample_demographics() -> dict:
    age = random.randint(AGE["min"], AGE["max"])
    return {
        "age":                 age,
        "gender":              random.choices(list(GENDER), weights=list(GENDER.values()))[0],
        "ethnicity":           random.choice(ETHNICITY),
        "relationship_status": _sample_relationship(age),
        "occupation":          random.choice(OCCUPATION),
        "trauma_type":         random.choice(TRAUMA_TYPE),
        "pcl5":                random.randint(PCL5["min"], PCL5["max"]),
    }


def resample_demographics_fields(demographics: dict, fields: list[str]) -> dict:
    _samplers = {
        "age":                 lambda: random.randint(AGE["min"], AGE["max"]),
        "gender":              lambda: random.choices(list(GENDER), weights=list(GENDER.values()))[0],
        "ethnicity":           lambda: random.choice(ETHNICITY),
        "relationship_status": lambda: _sample_relationship(demographics.get("age", 40)),
        "occupation":          lambda: random.choice(OCCUPATION),
        "trauma_type":         lambda: random.choice(TRAUMA_TYPE),
        "pcl5":                lambda: random.randint(PCL5["min"], PCL5["max"]),
    }
    updated = dict(demographics)
    for field in fields:
        if field in _samplers:
            updated[field] = _samplers[field]()
    return updated


def sample_cognitive_model(node_prob: float = 0.7, edge_prob: float = 0.5) -> dict:
    """Sample a random Ehlers & Clark cognitive model.

    Returns {"edges": EdgeGraph, "active_nodes": [str]}.
    """
    active_set = {n for n in _ALL_NODES if random.random() < node_prob}
    graph: EdgeGraph = from_schema(DIRECT_EDGES)
    for (p, c) in DIRECT_EDGES:
        if p in active_set and c in active_set and random.random() < edge_prob:
            set_weight(graph, p, c, round(random.uniform(0.01, 1.0), 4))
    active_nodes = [n for n in _ALL_NODES if n in active_set]
    return {"edges": graph, "active_nodes": active_nodes}


def sample_self_report(cognitive_model: dict, n_items: int = 3) -> dict:
    """Sample self-report items for each active node.

    Returns {node_name: [{"key": str, "value": str}]}.
    """
    result = {}
    for node in cognitive_model["active_nodes"]:
        pool = _NODE_POOLS.get(node, {})
        sampled = random.sample(list(pool.items()), min(n_items, len(pool)))
        result[node] = [{"key": k, "value": v} for k, v in sampled]
    return result


def sample_formulation(n_items: int = 3, node_prob: float = 0.7, edge_prob: float = 0.5) -> dict:
    """Sample a full formulation dict — {"nodes": ..., "edges": ...} — for VignetteCrafterAgent."""
    model = sample_cognitive_model(node_prob, edge_prob)
    ni    = sample_self_report(model, n_items)
    edges: dict = {}
    for p, children in model["edges"].items():
        for c, weight in children.items():
            edges.setdefault(p, {})[c] = {
                "strength":     weight,
                "parent_items": ni.get(p, []),
                "child_items":  ni.get(c, []),
            }
    return {"nodes": ni, "edges": edges}


def sample_session_state(agg_edges: EdgeGraph, active_nodes: list) -> tuple[dict, EdgeGraph]:
    """Draw a random binary session state from the patient's cognitive model.

    Each active node activates independently with p=0.7.
    Each edge activates if both endpoints active, weight > 0, and Bernoulli(weight) fires.
    Sessions are independent samples from the true graph — strong edges appear consistently.
    """
    nodes = {n: bool(n in active_nodes and random.random() < 0.7) for n in NODES}
    obs: EdgeGraph = {}
    for p, children in agg_edges.items():
        for c, weight in children.items():
            active = nodes.get(p) and nodes.get(c) and weight > 0 and random.random() < weight
            obs.setdefault(p, {})[c] = weight if active else 0.0
    return nodes, obs
