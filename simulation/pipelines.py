# Personas come from the FORMA dataset, so there is no sampling or vignette
# phase to toggle any more — the only variation left is whether memory and
# analysis run.
PIPELINES: dict[str, list[str]] = {
    "full":        ["load_persona", "session_state", "dialogue_state",
                    "validate_message", "conversation", "analysis"],
    "full_memory": ["load_persona", "session_state", "render_memory",
                    "dialogue_state", "validate_message", "conversation",
                    "analysis", "update_memory"],
    "no_analysis": ["load_persona", "session_state", "dialogue_state",
                    "validate_message", "conversation"],
    "no_validation": ["load_persona", "session_state", "dialogue_state",
                      "conversation", "analysis"],
}
