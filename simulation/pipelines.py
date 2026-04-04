PIPELINES: dict[str, list[str]] = {
    "full":                      ["persona", "validate_persona", "analyse_vignette", "dialogue_state", "validate_message", "conversation", "analysis"],
    "full_moderated":            ["persona", "validate_persona", "analyse_vignette", "dialogue_state", "validate_message", "conversation_moderated", "analysis"],
    "no_validation":             ["persona", "analyse_vignette", "dialogue_state", "conversation", "analysis"],
    "no_validation_moderated":   ["persona", "analyse_vignette", "dialogue_state", "conversation_moderated", "analysis"],
    "vignette_validation":       ["persona", "validate_persona", "analyse_vignette"],
    "no_analysis":               ["persona", "validate_persona", "analyse_vignette", "dialogue_state", "validate_message", "conversation"],
    "no_validation_no_analysis": ["persona", "dialogue_state", "conversation"],
}
