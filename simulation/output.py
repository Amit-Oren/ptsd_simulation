import json
import math
import textwrap
import numpy as np

WRAP_WIDTH = 100


def wrap(text: str) -> str:
    return "\n".join(
        textwrap.fill(line, width=WRAP_WIDTH) if line.strip() else ""
        for line in text.splitlines()
    )


def to_serializable(obj):
    if isinstance(obj, dict):
        def _key(k):
            if isinstance(k, tuple):
                return " -- ".join(k)
            if isinstance(k, np.integer):
                return int(k)
            return k
        return {_key(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def write_json(output: dict, f):
    """Write output JSON with blank lines between conversation turns."""
    conversation = output.pop("conversation")
    analysis     = output.pop("analysis")

    f.write(json.dumps(output, indent=2, ensure_ascii=False)[:-1])

    f.write(',\n\n  "conversation": [\n')
    for i, turn in enumerate(conversation):
        f.write("    " + json.dumps(turn, ensure_ascii=False))
        f.write(",\n\n" if i < len(conversation) - 1 else "\n")
    f.write("  ],\n\n")

    f.write('  "analysis": ' + json.dumps(analysis, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    f.write("\n}\n")


def write_summary_json(output: dict, path):
    """Write a compact summary JSON: config, demographics, self_report, edges, observation."""
    _SR_KEYS = [
        "trigger_items", "appraisals_items_original",
        "memory_items_original", "threat_items_original",
        "strategies_items_original", "avoidance_items_original",
    ]
    sr_full = output.get("self_report", {})
    sr_clean = {k: sr_full[k] for k in _SR_KEYS if k in sr_full}

    summary = {
        "patient_id":   output["patient_id"],
        "timestamp":    output["experiment_timestamp"],
        "config":       output.get("config", {}),
        "demographics": output.get("demographics", {}),
        "self_report":  sr_clean,
        "agg_edges":    output.get("agg_edges", {}),
        "observation": {
            "datetime": output.get("obs_datetime", ""),
            "index":    output.get("obs_index"),
            "nodes":    output.get("obs_nodes", {}),
            "edges":    output.get("obs_edges", {}),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_txt(output: dict, path):
    """Write a human-readable text report."""
    SEP  = "─" * WRAP_WIDTH
    SEP2 = "· " * (WRAP_WIDTH // 2)

    def section(title):
        return f"\n{title}\n{SEP}\n"

    with open(path, "w", encoding="utf-8") as f:

        # ── Header ──────────────────────────────────────────────────────────
        f.write(f"PATIENT {output['patient_id']}  |  session {output.get('obs_index', '?')}  |  {output['experiment_timestamp']}\n")
        f.write(SEP + "\n")

        # ── Config ──────────────────────────────────────────────────────────
        cfg = output.get("config", {})
        f.write(section("CONFIG"))
        f.write(f"  pipeline:    {cfg.get('pipeline')}\n")
        f.write(f"  temperature: {cfg.get('temperature')}\n")
        f.write(f"  num_turns:   {cfg.get('num_turns')}\n")
        f.write(f"  obs_index:   {cfg.get('obs_index')}\n")
        f.write(f"  max_retries: {cfg.get('max_retries')}\n")
        f.write(f"  seed:        {cfg.get('seed')}\n")
        f.write("  models:\n")
        for role, model in cfg.get("models", {}).items():
            f.write(f"    {role:<20} {model}\n")

        # ── Demographics ─────────────────────────────────────────────────────
        demo = output.get("demographics", {})
        f.write(section("DEMOGRAPHICS"))
        for k, v in demo.items():
            f.write(f"  {k:<15} {v}\n")

        # ── Self-report (key items only) ─────────────────────────────────────
        sr = output.get("self_report", {})
        _SR_LABELS = {
            "trigger_items":               "Triggers",
            "appraisals_items_original":   "Negative Appraisals",
            "memory_items_original":       "Memory",
            "threat_items_original":       "Threat",
            "strategies_items_original":   "Maladaptive Strategies",
            "avoidance_items_original":    "Avoidance",
        }
        f.write(section("SELF-REPORT"))
        for key, label in _SR_LABELS.items():
            val = sr.get(key, "")
            if val and not (isinstance(val, float) and math.isnan(val)):
                items = [i.strip() for i in str(val).split("|") if i.strip()]
                f.write(f"  {label}:\n")
                for item in items:
                    f.write(f"    • {item}\n")

        # ── Aggregated edges ─────────────────────────────────────────────────
        f.write(section("AGGREGATED EMA EDGES"))
        for edge, val in output.get("agg_edges", {}).items():
            arrow = edge.replace("--", "→")
            f.write(f"  {arrow:<50} {val:.4f}\n")

        # ── Observation ──────────────────────────────────────────────────────
        f.write(section(f"OBSERVATION  |  {output.get('obs_datetime', '')}  |  index {output.get('obs_index', '?')}"))
        f.write("  Nodes:\n")
        for node, val in output.get("obs_nodes", {}).items():
            active = "✓" if val else "·"
            f.write(f"    {active} {node}\n")
        f.write("  Edges:\n")
        for edge, val in output.get("obs_edges", {}).items():
            active = "✓" if val else "·"
            arrow = edge.replace("--", "→")
            f.write(f"    {active} {arrow}\n")

        # ── Validation summary ───────────────────────────────────────────────
        vs = output.get("validation_summary", {})
        f.write(section("VALIDATION SUMMARY"))
        for part, summary in vs.items():
            status = "PASS" if summary.get("ultimately_passed") else "FAIL"
            f.write(f"  {part:<20} {summary['attempts']} attempt(s)  →  {status}\n")

        # ── Persona validation attempts ──────────────────────────────────────
        f.write(section("PERSONA VALIDATION"))
        for i, attempt in enumerate(output.get("persona_attempts", []), 1):
            status = "PASS" if attempt["passed"] else "FAIL"
            f.write(f"\n  [{i}] {status}\n")
            f.write(wrap(attempt["output"]) + "\n")
            if attempt.get("feedback"):
                f.write(f"\n  Feedback: {wrap(attempt['feedback'])}\n")

        # ── Persona (final) ───────────────────────────────────────────────────
        f.write(section("PERSONA (final)"))
        f.write(wrap(output.get("persona", "")) + "\n")

        # ── Initial message validation ────────────────────────────────────────
        f.write(section("INITIAL MESSAGE VALIDATION"))
        for i, attempt in enumerate(output.get("message_attempts", []), 1):
            status = "PASS" if attempt["passed"] else "FAIL"
            f.write(f"\n  [{i}] {status}\n")
            f.write(wrap(attempt["output"]) + "\n")
            if attempt.get("feedback"):
                f.write(f"\n  Feedback: {wrap(attempt['feedback'])}\n")

        # ── Conversation ──────────────────────────────────────────────────────
        f.write(section("CONVERSATION"))
        mod_by_turn = {e["turn_index"]: e["raw"] for e in output.get("moderation_log", [])}
        for turn in output.get("conversation", []):
            idx     = turn["turn_index"]
            speaker = turn["speaker"].upper()
            f.write(f"\n[{idx}] {speaker}\n")
            f.write(wrap(turn["text"]) + "\n")
            if turn["speaker"] == "client" and idx in mod_by_turn:
                raw = mod_by_turn[idx]
                if raw != turn["text"]:
                    f.write(f"  (raw) {wrap(raw)}\n")

        # ── Analysis ──────────────────────────────────────────────────────────
        analysis = output.get("analysis", {})
        f.write(section("ANALYSIS — NODES"))
        for node, data in analysis.get("nodes", {}).items():
            score = data.get("score", 0)
            mark  = "✓" if score else "·"
            f.write(f"  {mark} {node:<28} {data.get('explanation', '')}\n")
            if data.get("quote"):
                f.write(f"      \"{data['quote']}\"\n")

        f.write(section("ANALYSIS — EDGES"))
        for edge, data in analysis.get("edges", {}).items():
            score = data.get("score", 0)
            mark  = "✓" if score else "·"
            arrow = edge.replace("--", " → ")
            f.write(f"  {mark} {arrow:<45} {data.get('explanation', '')}\n")
            if data.get("quote"):
                f.write(f"      \"{data['quote']}\"\n")
