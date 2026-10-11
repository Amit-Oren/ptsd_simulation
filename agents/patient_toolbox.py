"""
patient_toolbox.py - builds the cognitive-component tools for one patient.

Built per-patient rather than at module level: each tool closes over this
patient's activation map, and obs_nodes decides which tools exist at all, so an
inactive node cannot be expressed instead of merely being discouraged.

Every tool runs the identical path - activate an item, let it cascade along the
sampled edges. Only the wording of what comes back differs.
"""
import logging
from langchain_core.tools import tool

from simulation.activation import activate, next_unused, propagate

logger = logging.getLogger(__name__)


_TAGS = {
    "Triggers":               "[trigger]",
    "Memory":                 "[memory]",
    "Threat":                 "[threat]",
    "Negative Appraisals":    "[belief]",
    "Maladaptive Strategies": "[coping]",
}


_FOOTER = ("  Those are clinical labels, not your words. Say what actually "
           "happened to you just now - one specific moment, the way you would "
           "text it to someone.")


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
        history = propagate(activation, obs_edges, params,
                            origin_node=node, origin_item=item, hops=params.hops)

        level = activation[node][item]
        cascade = [
            {"hop": hop_i, "from_node": entry["from_node"],
             "from_item": entry["from_item"], "to_node": to_node,
             "to_item": to_item, "amount": entry["amount"]}
            for hop_i, hop in enumerate(history, 1)
            for (to_node, to_item), entry in hop.items()
        ]
        trace.append({"node": node, "item": item, "matched": matched,
                      "level": level, "cascade": cascade})
        logger.debug("%s -> %s (%.2f), %d hops", node, item, level, len(history))

        lines = [f"{_TAGS.get(node, node.lower())}",
                 f"  what: {values.get(node, {}).get(item, item)}",
                 f"  how strongly: {_word(level)} ({level:.2f})"]


        strongest: dict[tuple[str, str], float] = {}
        for step in cascade:
            key = (step["to_node"], step["to_item"])
            strongest[key] = max(strongest.get(key, 0.0), step["amount"])
        arrived = sorted(strongest.items(), key=lambda kv: -kv[1])[:2]
        arrived = [(k, a) for k, a in arrived if a >= params.spread_floor]
        if arrived:
            lines.append("  it brought with it:")
            for (to_node, to_item), amount in arrived:
                label = values.get(to_node, {}).get(to_item, to_item)
                lines.append(f"    - {label} ({_word(amount)})")
        lines.append(_FOOTER)
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
