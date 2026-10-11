"""
Pipeline step functions. Each step receives the current state dict,
updates it, and returns the updated dict.
"""
import logging
from agents.dialogue_state_agent import DialogueStateAgent
from agents.validator_agent import ValidatorAgent
from simulation.forma import load_persona
from simulation.sampling import sample_session_state
from simulation.factory import load_agent_class
from simulation.graph import count_active
from simulation.patient_store import format_recent_transcripts
from simulation.memory.kg import TypedKnowledgeGraph
from simulation.memory.render import build_interviewer_memory_context

logger = logging.getLogger(__name__)


def step_load_persona(label: str, persona_id: int,
                      vignette_condition: str = "no_formulation") -> dict:
    """Load one patient from the FORMA dataset.

    Replaces the old sample-and-craft phase entirely: the personas are already
    generated and clinician-validated, so building a patient costs no LLM calls
    and `persona_id` is the seed. The no_formulation vignette is used by default
    because it was written without the cognitive graph — a vignette that narrates
    the causal links hands the analyst the answer it is supposed to recover.
    """
    persona = load_persona(persona_id, vignette_condition)
    logger.info("[%s] load_persona %d — %d active nodes, %d active edges, %s vignette",
                label, persona_id, len(persona["active_nodes"]),
                count_active(persona["agg_edges"]), vignette_condition)
    return persona


def step_session_state(label: str, state: dict) -> dict:
    obs_nodes, obs_edges = sample_session_state(state["agg_edges"], state["active_nodes"])
    n_active_nodes = sum(bool(v) for v in obs_nodes.values())
    n_active_edges = count_active(obs_edges)
    logger.info("[%s] session_state — %d nodes active, %d edges active",
                label, n_active_nodes, n_active_edges)
    return {"obs_nodes": obs_nodes, "obs_edges": obs_edges}


def step_render_memory(label: str, state: dict) -> dict:
    memory_context = build_interviewer_memory_context(state.get("memory_graph"))
    return {"memory_context": memory_context}


def step_dialogue_state(label: str, state: dict, llms: dict) -> dict:
    session_history = format_recent_transcripts(state.get("recent_transcripts", []))
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
    session_history = format_recent_transcripts(state.get("recent_transcripts", []))
    client = load_agent_class(agent_classes["client"])(
        name=label, role="Patient", llm=llms["client"],
        vignette=state["vignette"], nodes=state["obs_nodes"], edges=state["obs_edges"],
        initial_message=state["initial_message"], session_history=session_history,
        self_report=state.get("self_report", {}),
    )
    interviewer = load_agent_class(agent_classes["interviewer"])(
        name="Interviewer", role="Interviewer", llm=llms["interviewer"],
        memory_context=state.get("memory_context", ""),
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
    memory_graph = state.get("memory_graph")
    existing_entities_text = (
        memory_graph.render_for_coreference() if memory_graph is not None
        else "(no existing entities yet — this is the first session)"
    )
    message = (
        f"{conversation_text}\n\n"
        "## Existing memory graph entities (for coreference — reuse an id below if this "
        "session mentions the same specific thing again, otherwise leave existing_id blank)\n"
        f"{existing_entities_text}"
    )
    analysis = analyst.respond(message)
    return {"analysis": analysis}


def step_update_memory(label: str, state: dict, session_idx: int) -> dict:
    memory_graph = state.get("memory_graph") or TypedKnowledgeGraph()
    analysis = state.get("analysis") or {}
    extraction = {
        "entities": analysis.get("entities", []),
        "relations": analysis.get("relations", []),
    }
    memory_graph.ingest(extraction, session_idx=session_idx)
    logger.info("[%s] update_memory — %d entities, %d relations extracted this session",
                label, len(extraction["entities"]), len(extraction["relations"]))
    return {"memory_graph": memory_graph}
