PIPELINES: dict[str, list[str]] = {
    "full":                      ["craft_persona", "vignette", "validate_vignette", "analyse_vignette",
                                  "session_state", "dialogue_state", "validate_message", "conversation", "analysis"],
    "no_validation":             ["craft_persona", "vignette", "analyse_vignette",
                                  "session_state", "dialogue_state", "conversation", "analysis"],
    "vignette_only":             ["craft_persona", "vignette", "validate_vignette", "analyse_vignette"],
    "no_analysis":               ["craft_persona", "vignette", "validate_vignette", "analyse_vignette",
                                  "session_state", "dialogue_state", "validate_message", "conversation"],
    "no_validation_no_analysis": ["craft_persona", "vignette",
                                  "session_state", "dialogue_state", "conversation"],
}
