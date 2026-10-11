# Personas come from FORMA and scenarios from the generated pool, so there is
# no sampling or vignette phase left to toggle. What varies is whether the
# interviewer carries memory across sessions, and whether the analyst runs.
PIPELINES: dict[str, list[str]] = {
    "full":        ["load_persona", "scenario", "session_state", "activation",
                    "conversation", "analysis"],
    "full_memory": ["load_persona", "scenario", "session_state", "render_memory",
                    "activation", "conversation", "analysis", "update_memory"],
    "no_analysis": ["load_persona", "scenario", "session_state", "activation",
                    "conversation"],
}
