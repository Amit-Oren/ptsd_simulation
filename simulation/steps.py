"""
Pipeline step functions. Each step receives the current state dict,
updates it, and returns the updated dict.
"""
import logging
from agents.persona_crafter_agent import PersonaCrafterAgent
from agents.persona_validator_agent import PersonaValidatorAgent
from agents.vignette_crafter_agent import VignetteCrafterAgent
from agents.vignette_validator_agent import VignetteValidatorAgent
from agents.dialogue_state_agent import DialogueStateAgent
from agents.validator_agent import ValidatorAgent
from data.input.input import (
    sample_demographics, sample_cognitive_model, sample_self_report,
    sample_session_state, resample_demographics_fields,
)
from simulation.factory import load_agent_class
from simulation.graph import count_active, count_by_strength, iter_edges
from simulation.patient_store import format_session_history

logger = logging.getLogger(__name__)


def step_craft_persona(label: str, llms: dict,
                       persona_validator: PersonaValidatorAgent,
                       persona_crafter: PersonaCrafterAgent,
                       max_retries: int,
                       n_items: int = 3,
                       node_prob: float = 0.7,
                       edge_prob: float = 0.5) -> dict:
    # Stage 1 — demographics
    demographics = sample_demographics()
    for _ in range(max_retries + 1):
        result = persona_validator.validate_demographics(demographics)
        if result["passed"]:
            break
        demographics = resample_demographics_fields(demographics, result["problematic_fields"])

    # Stage 2 — cognitive model
    cognitive_model = sample_cognitive_model(node_prob, edge_prob)
    agg_edges    = cognitive_model["edges"]
    active_nodes = cognitive_model["active_nodes"]

    # Stage 3 — self-report
    self_report = sample_self_report(cognitive_model, n_items)
    cumulative_bad_keys: dict[str, set] = {}
    for _ in range(max_retries + 1):
        result = persona_validator.validate_self_report(demographics, agg_edges, self_report)
        if result["passed"]:
            break
        self_report = persona_crafter.fix_self_report(
            demographics, self_report, result["issues"],
            result["problematic_items"], cumulative_bad_keys,
        )
        for node, keys in result["problematic_items"].items():
            cumulative_bad_keys.setdefault(node, set()).update(keys)

    logger.info("[%s] craft_persona — %d active nodes, %d active edges",
                label, len(active_nodes), count_active(agg_edges))
    return {
        "demographics": demographics,
        "self_report":  self_report,
        "agg_edges":    agg_edges,
        "active_nodes": active_nodes,
    }


def step_vignette(label: str, state: dict, llms: dict) -> dict:
    agg_edges    = state["agg_edges"]
    active_nodes = state["active_nodes"]
    self_report  = state["self_report"]

    formulation_edges: dict = {}
    for p, c, weight in iter_edges(agg_edges):
        formulation_edges.setdefault(p, {})[c] = {
            "strength":     weight,
            "parent_items": self_report.get(p, []),
            "child_items":  self_report.get(c, []),
        }
    formulation = {"nodes": active_nodes, "edges": formulation_edges}
    crafter = VignetteCrafterAgent(
        name=f"{label}_VignetteCrafter", role="VignetteCrafter",
        llm=llms["persona_crafter"],
        demographics=state["demographics"],
        self_report=self_report,
        formulation=formulation,
    )
    vignette = crafter.create_vignette()
    logger.info("[%s] vignette written", label)
    return {
        "vignette": vignette,
        "vignette_attempts": [{"vignette": vignette, "passed": True,
                               "violations": [], "violation_count": 0, "feedback": None}],
        "_vignette_crafter": crafter,
    }


def step_validate_vignette(label: str, state: dict,
                            vignette_validator: VignetteValidatorAgent,
                            max_retries: int) -> dict:
    agg_edges = state["agg_edges"]
    strong, moderate = count_by_strength(agg_edges)
    logger.info("[%s] validate_vignette — %d strong, %d moderate edges", label, strong, moderate)

    vignette, attempts = vignette_validator.validate_with_retry(
        initial_vignette=state["vignette"],
        context={"edges": agg_edges},
        retry_fn=state["_vignette_crafter"].create_vignette_with_feedback,
        max_retries=max_retries,
        label=label,
    )
    return {"vignette": vignette, "vignette_attempts": attempts}


def step_analyse_vignette(label: str, state: dict, llms: dict, agent_classes: dict) -> dict:
    analyst = load_agent_class(agent_classes["vignette_analyst"])(
        name=f"{label}_VignetteAnalyst", role="VignetteAnalyst", llm=llms["analyst"],
    )
    vignette_analysis = analyst.respond(state["vignette"])
    logger.info("[%s] vignette analysis complete", label)
    return {"vignette_analysis": vignette_analysis}


def step_session_state(label: str, state: dict) -> dict:
    obs_nodes, obs_edges = sample_session_state(state["agg_edges"], state["active_nodes"])
    n_active_nodes = sum(bool(v) for v in obs_nodes.values())
    n_active_edges = count_active(obs_edges)
    logger.info("[%s] session_state — %d nodes active, %d edges active",
                label, n_active_nodes, n_active_edges)
    return {"obs_nodes": obs_nodes, "obs_edges": obs_edges}


def step_dialogue_state(label: str, state: dict, llms: dict) -> dict:
    session_history = format_session_history(state.get("session_summaries", []))
    dialogue_state = DialogueStateAgent(
        name=f"{label}_DialogueState", role="DialogueState",
        llm=llms["dialogue_state"],
        vignette=state["vignette"],
        nodes=state["obs_nodes"],
        edges=state["obs_edges"],
        session_history=session_history,
    )
    initial_message = dialogue_state.create_message()
    return {
        "initial_message":  initial_message,
        "message_attempts": [{"output": initial_message, "passed": True, "feedback": None}],
        "_dialogue_state":  dialogue_state,
    }


def step_validate_message(label: str, state: dict, validator: ValidatorAgent, max_retries: int) -> dict:
    initial_message, attempts = validator.validate_with_retry(
        initial_output=state["initial_message"],
        output_type="dialogue",
        context={"vignette": state["vignette"], "nodes": state["obs_nodes"], "edges": state["obs_edges"]},
        retry_fn=state["_dialogue_state"].create_message_with_feedback,
        max_retries=max_retries,
        label=label,
    )
    return {"initial_message": initial_message, "message_attempts": attempts}


def step_conversation(label: str, state: dict, llms: dict, agent_classes: dict, num_turns: int) -> dict:
    session_history = format_session_history(state.get("session_summaries", []))
    client = load_agent_class(agent_classes["client"])(
        name=label, role="Patient", llm=llms["client"],
        vignette=state["vignette"], nodes=state["obs_nodes"], edges=state["obs_edges"],
        initial_message=state["initial_message"], session_history=session_history,
        self_report=state.get("self_report", {}),
    )
    interviewer = load_agent_class(agent_classes["interviewer"])(
        name="Interviewer", role="Interviewer", llm=llms["interviewer"],
    )
    logger.info("[%s] conversation — starting %d turns", label, num_turns)

    conversation = [{"speaker": "client", "turn_index": 0, "text": state["initial_message"]}]
    interviewer_response = interviewer.respond(state["initial_message"])
    conversation.append({"speaker": "interviewer", "turn_index": 1, "text": interviewer_response})

    for i in range(num_turns - 1):
        client_response = client.respond(interviewer_response)
        conversation.append({"speaker": "client", "turn_index": 2 + i * 2, "text": client_response})
        interviewer_response = interviewer.respond(client_response)
        conversation.append({"speaker": "interviewer", "turn_index": 3 + i * 2, "text": interviewer_response})

    logger.info("[%s] conversation complete (%d messages)", label, len(conversation))
    return {"conversation": conversation}


def step_analysis(label: str, state: dict, llms: dict, agent_classes: dict) -> dict:
    analyst = load_agent_class(agent_classes["analyst"])(
        name="Analyst", role="Analyst", llm=llms["analyst"],
    )
    conversation_text = "\n".join(
        f"{t['speaker'].upper()} (turn {t['turn_index']}): {t['text']}"
        for t in state["conversation"]
    )
    analysis = analyst.respond(conversation_text)
    return {"analysis": analysis}
