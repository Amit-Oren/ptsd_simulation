import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import cast
from agents.base_agent import set_context_subdir
from agents.validator_agent import ValidatorAgent
from data.input.input import get_available_ids
from simulation.factory import build_llm
from simulation.output import to_serializable, write_json, write_txt, write_summary_json
from simulation.pipelines import PIPELINES
from simulation.steps import (
    step_persona, step_validate_persona, step_analyse_vignette,
    step_dialogue_state, step_validate_message,
    step_conversation, step_conversation_moderated, step_analysis,
)

logger = logging.getLogger(__name__)


class SimulationRunner:
    def __init__(self, num_patients: int, num_turns: int, seed: int,
                 models: dict, agent_classes: dict,
                 experiment_dir: Path,
                 patient_ids: list = None, obs_index: int | list = 0,
                 max_retries: int = 2, temperature: float = 0.7,
                 pipeline: str = "full", persona_context: bool = False):
        self.num_patients  = num_patients
        self.num_turns     = num_turns
        self.seed          = seed
        self.patient_ids   = patient_ids
        self.obs_index     = obs_index if isinstance(obs_index, list) else [obs_index]
        self.max_retries   = max_retries
        self.agent_classes = agent_classes
        self.experiment_dir = experiment_dir
        self.pipeline_name  = pipeline
        self.pipeline       = PIPELINES[pipeline]
        self.persona_context = persona_context
        self.models        = models
        self.temperature   = temperature
        np.random.seed(seed)
        self._llms = {role: build_llm(model_name, temperature) for role, model_name in models.items()}

    def _sample_patients(self) -> list:
        available = get_available_ids()
        return list(np.random.choice(available, size=self.num_patients, replace=False))

    # Steps that run once per patient (not per session)
    _PATIENT_STEPS = {"persona", "validate_persona", "analyse_vignette"}

    def run(self):
        patient_ids = self.patient_ids if self.patient_ids is not None else self._sample_patients()

        for patient_id in patient_ids:
            label = f"Client_{patient_id}"
            logger.info("=== Patient %s starting ===", patient_id)

            patient_state: dict = {}
            validator = ValidatorAgent(
                name=f"{label}_Validator", role="Validator", llm=self._llms["validator"],
            )

            # Run patient-level steps once (persona, validate_persona)
            set_context_subdir(f"patient_{patient_id}")
            for step in self.pipeline:
                if step not in self._PATIENT_STEPS:
                    continue
                logger.info("[%s] step_%s: starting", label, step)
                if step == "persona":
                    patient_state.update(step_persona(label, patient_id, self._llms, self.persona_context))
                elif step == "validate_persona":
                    patient_state.update(step_validate_persona(label, patient_state, validator, self.max_retries))
                elif step == "analyse_vignette":
                    patient_state.update(step_analyse_vignette(label, patient_state, self._llms, self.agent_classes))
                logger.info("[%s] step_%s: done", label, step)

            # Run session-level steps once per obs_index
            for obs_index in self.obs_index:
                set_context_subdir(f"session{obs_index}")
                logger.info("[%s] === Session obs_index=%d starting ===", label, obs_index)
                state = dict(patient_state)

                for step in self.pipeline:
                    if step in self._PATIENT_STEPS:
                        continue
                    logger.info("[%s] step_%s (obs=%d): starting", label, step, obs_index)
                    if step == "dialogue_state":
                        state.update(step_dialogue_state(label, patient_id, state, self._llms, obs_index))
                    elif step == "validate_message":
                        state.update(step_validate_message(label, state, validator, self.max_retries))
                    elif step == "conversation":
                        state.update(step_conversation(label, state, self._llms, self.agent_classes, self.num_turns))
                    elif step == "conversation_moderated":
                        state.update(step_conversation_moderated(label, state, self._llms, self.agent_classes, self.num_turns))
                    elif step == "analysis":
                        state.update(step_analysis(label, state, self._llms, self.agent_classes))
                    logger.info("[%s] step_%s (obs=%d): done", label, step, obs_index)

                self._save(patient_id, obs_index, state)

    def _validation_summary(self, attempts: list) -> dict:
        passed    = sum(1 for a in attempts if a["passed"])
        failed    = sum(1 for a in attempts if not a["passed"])
        ultimately_passed = attempts[-1]["passed"] if attempts else None
        return {"attempts": len(attempts), "passed": passed, "failed": failed,
                "ultimately_passed": ultimately_passed}

    def _save(self, patient_id, obs_index: int, state: dict):
        persona_attempts  = state.get("persona_attempts", [])
        message_attempts  = state.get("message_attempts", [])

        output = cast(dict, to_serializable({
            "patient_id":          patient_id,
            "experiment_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "pipeline":     self.pipeline_name,
                "models":       self.models,
                "temperature":  self.temperature,
                "num_turns":    self.num_turns,
                "obs_index":    obs_index,
                "max_retries":  self.max_retries,
                "seed":         self.seed,
            },
            "demographics":    state.get("demographics", {}),
            "self_report":     state.get("self_report", {}),
            "obs_datetime":    str(state.get("obs_datetime", "")),
            "obs_index":       obs_index,
            "obs_nodes":       state.get("obs_nodes", {}),
            "obs_edges":       state.get("obs_edges", {}),
            "agg_edges":       state.get("agg_edges", {}),
            "validation_summary": {
                "persona":  self._validation_summary(persona_attempts),
                "message":  self._validation_summary(message_attempts),
            },
            "persona":           state.get("persona", ""),
            "persona_attempts":  persona_attempts,
            "vignette_analysis": state.get("vignette_analysis", {}),
            "initial_message":  state.get("initial_message", ""),
            "message_attempts": message_attempts,
            "conversation":     state.get("conversation", []),
            "moderation_log":   state.get("moderation_log", []),
            "analysis":         state.get("analysis", ""),
        }))

        base    = self.experiment_dir / f"experiment_{patient_id}_session{obs_index}"
        summary = self.experiment_dir / f"patient_{patient_id}_session{obs_index}_summary"

        with open(f"{base}.json", "w", encoding="utf-8") as f:
            write_json(output.copy(), f)

        write_txt(output, f"{base}.txt")

        write_summary_json(output, f"{summary}.json")

        label = f"Client_{patient_id}"
        logger.info("[%s] saved → %s.json / .txt / _summary.json", label, base.name)
