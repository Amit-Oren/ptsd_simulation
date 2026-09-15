import logging
import random
from datetime import datetime
from pathlib import Path
from typing import cast
from simulation.run_context import set_context_subdir, reset_run_tokens, get_run_tokens
from agents.persona_crafter_agent import PersonaCrafterAgent
from agents.persona_validator_agent import PersonaValidatorAgent
from agents.vignette_validator_agent import VignetteValidatorAgent
from agents.validator_agent import ValidatorAgent
from simulation.factory import build_llm
from simulation.output import to_serializable, write_json, write_txt, write_summary_json, write_manifest
from simulation.patient_store import load_patient_state, update_and_save
from simulation.memory.store import save_memory_graph, copy_memory_snapshot_to_output
from simulation.pipelines import PIPELINES
from simulation.steps import (
    step_craft_persona, step_vignette, step_validate_vignette, step_analyse_vignette,
    step_session_state, step_render_memory, step_dialogue_state, step_validate_message,
    step_conversation, step_analysis, step_update_memory,
)

logger = logging.getLogger(__name__)


class SimulationRunner:
    def __init__(self, num_patients: int, num_sessions: int, num_turns: int, seed: int,
                 models: dict, agent_classes: dict,
                 experiment_dir: Path,
                 max_retries: int = 2, temperature: float = 0.7,
                 pipeline: str = "full",
                 n_items: int = 3, node_prob: float = 0.7, edge_prob: float = 0.5):
        self.num_patients   = num_patients
        self.num_sessions   = num_sessions
        self.num_turns      = num_turns
        self.seed           = seed
        self.max_retries    = max_retries
        self.agent_classes  = agent_classes
        self.experiment_dir = experiment_dir
        self.pipeline_name  = pipeline
        self.pipeline       = PIPELINES[pipeline]
        self.models         = models
        self.temperature    = temperature
        self.n_items        = n_items
        self.node_prob      = node_prob
        self.edge_prob      = edge_prob
        random.seed(seed)
        self._llms = {role: build_llm(model_name, temperature) for role, model_name in models.items()}
        self._manifest = {
            "pipeline":     self.pipeline_name,
            "models":       self.models,
            "temperature":  self.temperature,
            "num_turns":    self.num_turns,
            "num_sessions": self.num_sessions,
            "num_patients": self.num_patients,
            "max_retries":  self.max_retries,
            "seed":         self.seed,
            "n_items":      self.n_items,
            "node_prob":    self.node_prob,
            "edge_prob":    self.edge_prob,
        }

    _PATIENT_STEPS = {"craft_persona", "vignette", "validate_vignette", "analyse_vignette"}

    def _patient_dir(self, patient_idx: int) -> Path:
        return self.experiment_dir / "patients" / f"patient_{patient_idx}"

    def _session_dir(self, patient_idx: int, session_idx: int) -> Path:
        return self._patient_dir(patient_idx) / "sessions" / f"session_{session_idx}"

    def run(self):
        write_manifest(self._manifest, self.experiment_dir / "manifest.json")

        for patient_idx in range(self.num_patients):
            label = f"Client_{patient_idx}"
            logger.info("=== Patient %d starting ===", patient_idx)

            patient_state:     dict = {}
            persona_validator  = PersonaValidatorAgent(name=f"{label}_PersonaValidator",  role="PersonaValidator",  llm=self._llms["validator"])
            persona_crafter    = PersonaCrafterAgent(  name=f"{label}_PersonaCrafter",    role="PersonaCrafter",    llm=self._llms["persona_crafter"], n_items=self.n_items)
            vignette_validator = VignetteValidatorAgent(name=f"{label}_VignetteValidator", role="VignetteValidator", llm=self._llms["validator"])
            validator          = ValidatorAgent(        name=f"{label}_Validator",         role="Validator",         llm=self._llms["validator"])

            set_context_subdir(f"patients/patient_{patient_idx}/context/persona")

            for step in self.pipeline:
                if step not in self._PATIENT_STEPS:
                    continue
                logger.info("[%s] step_%s: starting", label, step)
                if step == "craft_persona":
                    patient_state.update(step_craft_persona(
                        label, self._llms, persona_validator, persona_crafter, self.max_retries,
                        self.n_items, self.node_prob, self.edge_prob,
                    ))
                elif step == "vignette":
                    patient_state.update(step_vignette(label, patient_state, self._llms))
                elif step == "validate_vignette":
                    patient_state.update(step_validate_vignette(label, patient_state, vignette_validator, self.max_retries))
                elif step == "analyse_vignette":
                    patient_state.update(step_analyse_vignette(label, patient_state, self._llms, self.agent_classes))
                logger.info("[%s] step_%s: done", label, step)

            patient_store_state = load_patient_state(self.experiment_dir, patient_idx, patient_state)

            reset_run_tokens()
            for session_idx in range(self.num_sessions):
                set_context_subdir(f"patients/patient_{patient_idx}/context/session_{session_idx}")
                logger.info("[%s] === Session %d starting ===", label, session_idx)
                state = dict(patient_store_state)

                for step in self.pipeline:
                    if step in self._PATIENT_STEPS:
                        continue
                    logger.info("[%s] step_%s (session=%d): starting", label, step, session_idx)
                    if step == "session_state":
                        state.update(step_session_state(label, state))
                    elif step == "render_memory":
                        state.update(step_render_memory(label, state))
                    elif step == "dialogue_state":
                        state.update(step_dialogue_state(label, state, self._llms))
                    elif step == "validate_message":
                        state.update(step_validate_message(label, state, validator, self.max_retries))
                    elif step == "conversation":
                        state.update(step_conversation(label, state, self._llms, self.agent_classes, self.num_turns))
                    elif step == "analysis":
                        state.update(step_analysis(label, state, self._llms, self.agent_classes))
                    elif step == "update_memory":
                        state.update(step_update_memory(label, state, session_idx))
                    logger.info("[%s] step_%s (session=%d): done", label, step, session_idx)

                if "update_memory" in self.pipeline:
                    save_memory_graph(state["memory_id"], state["memory_graph"], session_idx)
                    copy_memory_snapshot_to_output(patient_idx, state["memory_graph"], self.experiment_dir)

                tokens = get_run_tokens()
                reset_run_tokens()
                self._save(patient_idx, session_idx, state, tokens)
                patient_store_state = update_and_save(
                    self.experiment_dir, patient_idx, patient_store_state, state, session_idx
                )

    def _validation_summary(self, attempts: list) -> dict:
        passed = sum(1 for a in attempts if a.get("passed"))
        failed = len(attempts) - passed
        ultimately_passed = attempts[-1].get("passed") if attempts else None
        return {"attempts": len(attempts), "passed": passed, "failed": failed,
                "ultimately_passed": ultimately_passed}

    def _save(self, patient_idx: int, session_idx: int, state: dict, tokens: dict = None):
        vignette_attempts = state.get("vignette_attempts", [])
        message_attempts  = state.get("message_attempts", [])

        output = cast(dict, to_serializable({
            "patient_idx":          patient_idx,
            "session_idx":          session_idx,
            "experiment_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "demographics":    state.get("demographics", {}),
            "self_report":     state.get("self_report", {}),
            "active_nodes":    state.get("active_nodes", []),
            "agg_edges":       state.get("agg_edges", {}),
            "obs_nodes":       state.get("obs_nodes", {}),
            "obs_edges":       state.get("obs_edges", {}),
            "validation_summary": {
                "vignette": self._validation_summary(vignette_attempts),
                "message":  self._validation_summary(message_attempts),
            },
            "vignette":          state.get("vignette", ""),
            "vignette_attempts": vignette_attempts,
            "vignette_analysis": state.get("vignette_analysis", {}),
            "initial_message":   state.get("initial_message", ""),
            "message_attempts":  message_attempts,
            "conversation":      state.get("conversation", []),
            "moderation_log":    state.get("moderation_log", []),
            "analysis":          state.get("analysis", ""),
            "tokens":            tokens or {},
        }))

        session_dir = self._session_dir(patient_idx, session_idx)
        session_dir.mkdir(parents=True, exist_ok=True)

        with open(session_dir / "full.json", "w", encoding="utf-8") as f:
            write_json(output.copy(), f)

        write_txt(output, session_dir / "report.txt", manifest=self._manifest)
        write_summary_json(output, session_dir / "summary.json", manifest=self._manifest)

        logger.info("[Client_%d] saved → %s/{full.json,report.txt,summary.json}",
                    patient_idx, session_dir.relative_to(self.experiment_dir))
