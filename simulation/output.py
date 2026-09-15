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
        return {k: to_serializable(v) for k, v in obj.items()}
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
    conversation = output.get("conversation", [])
    analysis     = output.get("analysis", {})
    body = {k: v for k, v in output.items() if k not in ("conversation", "analysis")}

    f.write(json.dumps(body, indent=2, ensure_ascii=False)[:-1])

    f.write(',\n\n  "conversation": [\n')
    for i, turn in enumerate(conversation):
        f.write("    " + json.dumps(turn, ensure_ascii=False))
        f.write(",\n\n" if i < len(conversation) - 1 else "\n")
    f.write("  ],\n\n")

    f.write('  "analysis": ' + json.dumps(analysis, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    f.write("\n}\n")


def write_manifest(manifest: dict, path):
    """Write the experiment-level manifest once (pipeline, models, seed, etc.) —
    the single source for run config, instead of duplicating it into every session file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def write_summary_json(output: dict, path, manifest: dict | None = None):
    """Write a compact summary JSON: config, demographics, self_report, cognitive model."""
    analysis = output.get("analysis", {})
    summary = {
        "patient_idx":       output.get("patient_idx"),
        "session_idx":       output.get("session_idx"),
        "timestamp":         output["experiment_timestamp"],
        "config":            manifest or output.get("config", {}),
        "demographics":      output.get("demographics", {}),
        "self_report":       output.get("self_report", {}),
        "active_nodes":      output.get("active_nodes", []),
        "agg_edges":         output.get("agg_edges", {}),
        "vignette_analysis": output.get("vignette_analysis", {}),
        "session": {
            "obs_nodes":        output.get("obs_nodes", {}),
            "obs_edges":        output.get("obs_edges", {}),
            "analysis_summary": analysis.get("summary", ""),
            "analysis_graph":   analysis.get("edges", {}),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_txt(output: dict, path, manifest: dict | None = None):
    """Write a human-readable text report."""
    SEP  = "─" * WRAP_WIDTH
    SEP2 = "· " * (WRAP_WIDTH // 2)

    def section(title):
        return f"\n{title}\n{SEP}\n"

    with open(path, "w", encoding="utf-8") as f:

        # ── Header ──────────────────────────────────────────────────────────
        f.write(f"PATIENT {output.get('patient_idx', '?')}  |  session {output.get('session_idx', '?')}  |  {output['experiment_timestamp']}\n")
        f.write(SEP + "\n")

        # ── Config ──────────────────────────────────────────────────────────
        cfg = manifest or output.get("config", {})
        f.write(section("CONFIG"))
        f.write(f"  pipeline:    {cfg.get('pipeline')}\n")
        f.write(f"  temperature: {cfg.get('temperature')}\n")
        f.write(f"  num_turns:   {cfg.get('num_turns')}\n")
        f.write(f"  num_sessions:{cfg.get('num_sessions')}\n")
        f.write(f"  n_items:     {cfg.get('n_items')}\n")
        f.write(f"  node_prob:   {cfg.get('node_prob')}\n")
        f.write(f"  edge_prob:   {cfg.get('edge_prob')}\n")
        f.write(f"  max_retries: {cfg.get('max_retries')}\n")
        f.write(f"  seed:        {cfg.get('seed')}\n")
        f.write("  models:\n")
        for role, model in cfg.get("models", {}).items():
            f.write(f"    {role:<20} {model}\n")

        # ── Demographics ─────────────────────────────────────────────────────
        demo = output.get("demographics", {})
        f.write(section("DEMOGRAPHICS"))
        for k, v in demo.items():
            f.write(f"  {k:<20} {v}\n")

        # ── Self-report ──────────────────────────────────────────────────────
        sr = output.get("self_report", {})
        f.write(section("SELF-REPORT"))
        for node, items in sr.items():
            f.write(f"  {node}:\n")
            for item in items:
                if isinstance(item, dict):
                    f.write(f"    • {item['key']}: {item['value']}\n")
                else:
                    f.write(f"    • {item}\n")

        # ── Cognitive model edges ────────────────────────────────────────────
        f.write(section("COGNITIVE MODEL EDGES"))
        for p, children in output.get("agg_edges", {}).items():
            for c, val in children.items():
                f.write(f"  {p} → {c:<40} {val:.4f}\n")

        # ── Session state ────────────────────────────────────────────────────
        f.write(section(f"SESSION STATE  |  session {output.get('session_idx', '?')}"))
        f.write("  Nodes:\n")
        for node, val in output.get("obs_nodes", {}).items():
            active = "✓" if val else "·"
            f.write(f"    {active} {node}\n")
        f.write("  Edges:\n")
        for p, children in output.get("obs_edges", {}).items():
            for c, val in children.items():
                active = "✓" if val else "·"
                f.write(f"    {active} {p} → {c}\n")

        # ── Validation summary ───────────────────────────────────────────────
        vs = output.get("validation_summary", {})
        f.write(section("VALIDATION SUMMARY"))
        for part, summary in vs.items():
            status = "PASS" if summary.get("ultimately_passed") else "FAIL"
            f.write(f"  {part:<20} {summary['attempts']} attempt(s)  →  {status}\n")

        # ── Vignette validation attempts ─────────────────────────────────────
        f.write(section("VIGNETTE VALIDATION"))
        for i, attempt in enumerate(output.get("vignette_attempts", []), 1):
            status = "PASS" if attempt.get("passed") else "FAIL"
            f.write(f"\n  [{i}] {status}\n")
            f.write(wrap(attempt.get("vignette", attempt.get("output", ""))) + "\n")
            if attempt.get("feedback"):
                f.write(f"\n  Feedback: {wrap(attempt['feedback'])}\n")

        # ── Vignette (final) ─────────────────────────────────────────────────
        f.write(section("VIGNETTE (final)"))
        f.write(wrap(output.get("vignette", "")) + "\n")

        # ── Initial message validation ────────────────────────────────────────
        f.write(section("INITIAL MESSAGE VALIDATION"))
        for i, attempt in enumerate(output.get("message_attempts", []), 1):
            status = "PASS" if attempt.get("passed") else "FAIL"
            f.write(f"\n  [{i}] {status}\n")
            f.write(wrap(attempt.get("output", "")) + "\n")
            if attempt.get("feedback"):
                f.write(f"\n  Feedback: {wrap(attempt['feedback'])}\n")

        # ── Tokens ────────────────────────────────────────────────────────────
        tokens = output.get("tokens", {})
        if tokens:
            f.write(section("TOKEN USAGE (session)"))
            f.write(f"  input:  {tokens.get('input', 0)}\n")
            f.write(f"  output: {tokens.get('output', 0)}\n")
            f.write(f"  total:  {tokens.get('total', 0)}\n")

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
        if analysis.get("summary"):
            f.write(section("ANALYSIS — SUMMARY"))
            f.write(wrap(analysis["summary"]) + "\n")

        f.write(section("ANALYSIS — NODES"))
        for node, data in analysis.get("nodes", {}).items():
            score = data.get("score", 0)
            mark  = "✓" if score else "·"
            f.write(f"  {mark} {node:<28} {data.get('explanation', '')}\n")
            if data.get("quote"):
                f.write(f"      \"{data['quote']}\"\n")

        f.write(section("ANALYSIS — EDGES"))
        for p, children in analysis.get("edges", {}).items():
            for c, data in children.items():
                score = data.get("score", 0)
                mark  = "✓" if score else "·"
                arrow = f"{p} → {c}"
                f.write(f"  {mark} {arrow:<45} {data.get('explanation', '')}\n")
                if data.get("quote"):
                    f.write(f"      \"{data['quote']}\"\n")
