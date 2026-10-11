"""
One patient, one session, no LLM.

A worked example of the activation mechanism: a scenario, a sampled graph with
items inside each node, and a scripted sequence of tool calls standing in for
what the model would decide. Run it to watch activation move.

    python examples/simple_patient.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.activation import from_self_report, ActivationParams, decay
from agents.patient_toolbox import build_toolbox


# ── the patient ───────────────────────────────────────────────────────────────

SCENARIO = (
    "On Saturday you went to a friend's house for dinner. It was a part of town "
    "you had not been to before. You left after about an hour, before the food "
    "was served, and took a taxi home."
)

SELF_REPORT = {
    "Triggers": [
        {"key": "cold_temperatures", "value": "being cold"},
        {"key": "animals",           "value": "dogs, especially large ones"},
        {"key": "unfamiliar_places", "value": "places you have not been before"},
    ],
    "Memory": [
        {"key": "vivid_memory", "value": "the moment itself, very sharp"},
        {"key": "memory_gaps",  "value": "the parts you cannot account for"},
    ],
    "Maladaptive Strategies": [
        {"key": "avoid_location", "value": "not going near where it happened"},
        {"key": "scanning_exits", "value": "finding the way out of a room"},
        {"key": "over_work",      "value": "filling every hour with work"},
    ],
    "Negative Appraisals": [
        {"key": "self_blame", "value": "it happened because of how you acted"},
        {"key": "isolation",  "value": "you are set apart from everyone else"},
    ],
}

# Node-level weights, sampled once for this session. Both directions of a pair
# are drawn independently, so Triggers->Strategies and Strategies->Triggers
# can differ — the mind is not symmetric.
OBS_EDGES = {
    "Memory":                 {"Triggers": 0.8, "Maladaptive Strategies": 0.5},
    "Triggers":               {"Maladaptive Strategies": 0.6},
    "Maladaptive Strategies": {"Triggers": 0.4, "Negative Appraisals": 0.7},
}

OBS_NODES = {node: True for node in SELF_REPORT}

# What the model would have decided, turn by turn. None means a quiet turn.
SCRIPT = [
    ("the friend mentions the new neighbourhood", "notice_trigger"),
    ("asked how the evening went",                None),
    ("asked why they left early",                 "use_coping_strategy"),
    ("asked what was going through their mind",   "recall_memory"),
    ("asked something neutral about work",        None),
    ("asked how they feel about it now",          None),
]


# ── display ───────────────────────────────────────────────────────────────────

def bar(level: float, width: int = 18) -> str:
    filled = round(level * width)
    return "#" * filled + "." * (width - filled)


def show(activation: dict, params: ActivationParams) -> None:
    for node, items in activation.items():
        live = {k: v for k, v in items.items() if v > 0}
        if not live:
            continue
        print(f"    {node}")
        for item, level in sorted(live.items(), key=lambda kv: -kv[1]):
            over = "<- would say it" if level >= params.surface_threshold else ""
            print(f"      {item:<20} {level:5.3f}  {bar(level)}  {over}")


def main() -> None:
    params = ActivationParams()
    activation = from_self_report(SELF_REPORT)
    trace: list = []
    tools = {t.name: t for t in build_toolbox(
        activation, SELF_REPORT, OBS_NODES, OBS_EDGES, params, trace)}

    print("=" * 74)
    print("SCENARIO")
    print("=" * 74)
    for line in [SCENARIO[i:i + 72] for i in range(0, len(SCENARIO), 72)]:
        print(f"  {line}")
    print()
    print(f"  tools available: {', '.join(tools)}")
    print(f"  params: strength={params.strength} spread_rate={params.spread_rate} "
          f"decay_rate={params.decay_rate} hops={params.hops}")

    for turn, (prompt, tool_name) in enumerate(SCRIPT, 1):
        print()
        print("=" * 74)
        print(f"TURN {turn} — {prompt}")
        print("=" * 74)

        decay(activation, params)          # time passed since the last turn

        if tool_name is None:
            print("  (no tool called — nothing in the question touched anything)")
        else:
            args = {"what_matched": prompt} if tool_name == "notice_trigger" else {}
            result = tools[tool_name].invoke(args)
            print(f"  {tool_name}() returned:")
            for line in result.splitlines():
                print(f"    | {line}")
            cascade = trace[-1]["cascade"]
            if cascade:
                print("  cascade:")
                for hop_i, hop in enumerate(cascade, 1):
                    for (node, item), entry in hop.items():
                        print(f"    hop {hop_i}  {entry['from_item']} -> {item}"
                              f"  (+{entry['amount']:.3f})")

        print("  state after this turn:")
        show(activation, params)

    print()
    print("=" * 74)
    print("SESSION TRACE")
    print("=" * 74)
    for t in trace:
        print(f"  {t['node']:<24} {t['item']:<20} {t['level']:.3f}")


if __name__ == "__main__":
    main()
