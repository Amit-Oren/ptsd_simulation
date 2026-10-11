"""
Spreading activation over the patient's cognitive graph.

ActivationMap is the live state of one patient's mind: a number per item,
changing every turn. The wiring it spreads along is the already-sampled
obs_edges — this module never invents structure, only moves energy through it.

"""
from dataclasses import dataclass

# {node: {item_key: activation}}  - activation is [0-1]
ActivationMap = dict


@dataclass
class ActivationParams:
    """Per-patient dynamics. Sampled once, constant within a session."""
    strength:          float = 0.60   # activation injected by one tool call
    max_activation:    float = 1.00   # no item may go above this
    surface_threshold: float = 0.50   # an item this live won't be surfaced again
    spread_rate:       float = 0.40   # fraction of activation surviving one hop
    spread_floor:      float = 0.05   # a node below this does not pass anything on
    decay_rate:        float = 0.70   # fraction of activation surviving one turn
    quiet_level:       float = 0.02   # below this an item is treated as cold
    hops:              int   = 2      # how far a cascade reaches from its origin


def from_self_report(self_report: dict) -> ActivationMap:
    """Zeroed map covering every item this patient has."""
    result: ActivationMap = {}
    for node, items in self_report.items():
        result[node] = {
            (item["key"] if isinstance(item, dict) else item): 0.0
            for item in items
        }
    return result


def activate(activation: ActivationMap, node: str, item: str,
             amount: float, max_activation: float = 1.0) -> None:
    """Raise one item's activation, never above max_activation."""
    current = activation.setdefault(node, {}).get(item, 0.0)
    activation[node][item] = min(current + amount, max_activation)


def next_unused(activation: ActivationMap, node: str,
                surface_threshold: float = 0.5) -> str | None:
    """The first item in a node that is not already live, or None if all are.

    Scanned in sampled order rather than sorted by level, so the choice is not
    biased by item naming when several are equally cold (which is every item on
    turn 1). The order is whatever the persona's self_report lists, which is
    already randomised per patient upstream.
    """
    items = activation.get(node) or {}
    for key, level in items.items():
        if level < surface_threshold:
            return key
    return None


def hottest_item(activation: ActivationMap, node: str) -> str | None:
    """The most active item in a node - who spoke for it when it broadcast."""
    items = activation.get(node) or {}
    return max(items, key=items.get) if items else None


def cluster_levels(activation: ActivationMap) -> dict[str, float]:
    """How live each node is, combining its items by noisy-OR.

    1 - prod(1 - a_i): bounded at 1, monotone in every item, and reads as the
    chance that at least one item carries activation through. Two live triggers
    push harder than one (max would ignore the second) without summing, which
    would be unbounded and would make a node look hotter merely for having been
    sampled more items.
    """
    levels = {}
    for node, items in activation.items():
        survive = 1.0
        for level in items.values():
            survive *= (1.0 - level)
        levels[node] = 1.0 - survive
    return levels


def spread(activation: ActivationMap, obs_edges: dict, params: ActivationParams,
           wave: dict) -> dict:
    """One hop of spreading, item to item.

    `wave` is {(node, item): energy that item is carrying}. A specific item
    broadcasts its own activation - the cue that actually fired is what pulls in
    the memory, not an aggregate over every trigger the patient has. Weights stay
    node-level, because that is where the sampled ground truth lives.

    Energy carried is what just arrived, not the item's total level, so an item
    already hot from earlier in the conversation does not re-broadcast at full
    strength every time the wave touches it - otherwise a two-way edge
    (Triggers->Memory alongside Memory->Triggers) pumps instead of attenuating.

    Returns {(node, item): {"amount", "from_node", "from_item"}}.
    """
    incoming: dict[tuple[str, str], dict] = {}
    for (parent, parent_item), carried in wave.items():
        if carried < params.spread_floor:
            continue
        for child, weight in obs_edges.get(parent, {}).items():
            if weight <= 0:
                continue
            target = next_unused(activation, child, params.surface_threshold)
            if target is None:
                continue
            amount = carried * weight * params.spread_rate
            entry = incoming.setdefault((child, target),
                                        {"amount": 0.0, "from_node": parent,
                                         "from_item": parent_item})
            entry["amount"] += amount

    for (node, item), entry in incoming.items():
        activate(activation, node, item, entry["amount"], params.max_activation)
    return incoming


def decay(activation: ActivationMap, params: ActivationParams) -> None:
    """Fade every item by one turn.

    Applied once per turn, after tools and propagation — not inside each hop,
    which would conflate reach with time and make `hops` a second decay dial.
    Items falling under quiet_level are zeroed so they can be surfaced again.
    """
    for node, items in activation.items():
        for item, level in items.items():
            faded = level * params.decay_rate
            items[item] = 0.0 if faded < params.quiet_level else faded


def propagate(activation: ActivationMap, obs_edges: dict, params: ActivationParams,
              origin_node: str, origin_item: str, hops: int = 2) -> list[dict]:
    """Run the cascade outward from the item a tool just activated.

    The wave starts at that one item carrying its own activation, and each hop
    carries only what just arrived - so the cascade attenuates with distance and
    a cycle loses energy on every pass. Stops early when nothing arrives.

    Returns one hop dict per hop, for tracing.
    """
    start = activation.get(origin_node, {}).get(origin_item, 0.0)
    wave = {(origin_node, origin_item): start}
    history = []
    for _ in range(hops):
        arrived = spread(activation, obs_edges, params, wave=wave)
        if not arrived:
            break
        history.append(arrived)
        wave = {key: entry["amount"] for key, entry in arrived.items()}
    return history
