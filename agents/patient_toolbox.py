"""
patient_toolbox.py - builds the cognitive-component tools for one patient.

Built per-patient rather than at module level: each tool closes over this
patient's activation map, and obs_nodes decides which tools exist at all, so an
inactive node cannot be expressed instead of merely being discouraged.

Every tool runs the identical path - activate an item, let it cascade along the
sampled edges. Only the wording of what comes back differs. The clinical theory
lives in the edge weights, not in special cases here.
"""
import logging
from langchain_core.tools import tool

from simulation.activation import activate, next_unused, propagate

logger = logging.getLogger(__name__)

# How each node announces itself back to the patient.
_HEADERS = {
    "Triggers":               "SOMETHING CAUGHT YOU",
    "Memory":                 "A MEMORY SURFACED",
    "Threat":                 "DANGER RISES IN YOU",
    "Negative Appraisals":    "A BELIEF HARDENS",
    "Maladaptive Strategies": "YOU PULL AWAY",
}


def _item_values(self_report: dict) -> dict:
    """{node: {item_key: human-readable text}}"""
    out = {}
    for node, items in self_report.items():
        out[node] = {
            (i["key"] if isinstance(i, dict) else i):
            (i.get("value", i["key"]) if isinstance(i, dict) else i)
            for i in items
        }
    return out


def _word(level: float) -> str:
    if level >= 0.75:
        return "overwhelming"
    if level >= 0.50:
        return "strong"
    if level >= 0.25:
        return "present"
    return "faint"


def build_toolbox(activation: dict, self_report: dict, obs_nodes: dict,
                  obs_edges: dict, params, trace: list | None = None) -> list:
    """Build this patient's toolset, ready for BaseAgent(tools=...)."""
    values = _item_values(self_report)
    trace = trace if trace is not None else []

    def _fire(node: str, matched: str | None = None) -> str:
        item = next_unused(activation, node, params.surface_threshold)
        if item is None:
            return f"(nothing new surfaces there right now)"

        activate(activation, node, item, params.strength, params.max_activation)
        # The cascade is part of the same event - a cue does not fire and then
        # later pull the memory in, it arrives with it. Running it here means the
        # model reads the whole chain before it speaks, rather than a turn late.
        history = propagate(activation, obs_edges, params,
                            origin_node=node, origin_item=item, hops=params.hops)

        level = activation[node][item]
        trace.append({"node": node, "item": item, "matched": matched,
                      "level": level, "cascade": history})
        logger.debug("%s -> %s (%.2f), %d hops", node, item, level, len(history))

        lines = [f"{_HEADERS.get(node, node.upper())}: {values.get(node, {}).get(item, item)}",
                 f"  It is {_word(level)} ({level:.2f})."]

        followed = [(n, i, e) for hop in history for (n, i), e in hop.items()]
        if followed:
            lines.append("  It brings with it:")
            for n, i, entry in followed:
                lines.append(f"    - {values.get(n, {}).get(i, i)} ({_word(entry['amount'])})")
        return "\n".join(lines)

    tools = []

    if obs_nodes.get("Triggers"):
        @tool
        def notice_trigger(what_matched: str) -> str:
            """Something in this conversation resembles a cue that sets you off.
            Call this when a word, image or topic just snagged your attention the
            way a reminder does. Pass what specifically matched, in your own words."""
            return _fire("Triggers", matched=what_matched)
        tools.append(notice_trigger)

    if obs_nodes.get("Memory"):
        @tool
        def recall_memory() -> str:
            """A specific memory of what happened has surfaced. Call this when you
            have just been pulled back to the event itself, not when you are merely
            discussing it."""
            return _fire("Memory")
        tools.append(recall_memory)

    if obs_nodes.get("Threat"):
        @tool
        def feel_threatened() -> str:
            """A sense of present danger has risen in you. Call this when something
            feels unsafe right now, not when you are remembering having felt unsafe."""
            return _fire("Threat")
        tools.append(feel_threatened)

    if obs_nodes.get("Negative Appraisals"):
        @tool
        def express_belief() -> str:
            """A harsh conclusion about yourself, other people or the world has just
            become active in you, with a feeling of being simply true."""
            return _fire("Negative Appraisals")
        tools.append(express_belief)

    if obs_nodes.get("Maladaptive Strategies"):
        @tool
        def use_coping_strategy() -> str:
            """You have reached for something you do to manage this - avoiding,
            distracting, shutting down, changing the subject. Call this when you are
            pulling away from the feeling rather than staying with it."""
            return _fire("Maladaptive Strategies")
        tools.append(use_coping_strategy)

    logger.info("built %d patient tools: %s", len(tools), [t.name for t in tools])
    return tools
