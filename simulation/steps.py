"""
Pipeline step functions. Each step receives the current state dict,
updates it, and returns the updated dict.
"""
import logging
from agents.patient_agent import PatientAgent
from config.scenarios import sample_scenario
from simulation.activation import ActivationParams, from_self_report, decay
from simulation.forma import load_persona
from simulation.sampling import sample_session_state
from simulation.factory import load_agent_class
from simulation.graph import count_active
from simulation.patient_store import format_recent_transcripts
from simulation.memory.kg import TypedKnowledgeGraph
from simulation.memory.render import build_interviewer_memory_context


def snapshot(activation: dict) -> dict:
    """A copy of the activation map, so a turn's state survives later mutation."""
    return {node: dict(items) for node, items in activation.items()}

logger = logging.getLogger(__name__)


def step_load_persona(label: str, persona_id: int,
                      patient_context: str = "biography") -> dict:
    """Load one patient from the FORMA dataset.

    Replaces the old sample-and-craft phase entirely: the personas are already
    generated and clinician-validated, so building a patient costs no LLM calls
    and `persona_id` is the seed.

    `patient_context` decides what the patient reads about itself. The default,
    a biography, gives identity and what happened and nothing else — every
    FORMA vignette narrates the person's symptoms, and a patient handed that
    has no reason to call a tool. The vignette conditions remain as ablation
    arms for measuring how much of the analyst's recovery comes from the
    mechanism rather than from the prompt.
    """
    persona = load_persona(persona_id, patient_context)
    logger.info("[%s] load_persona %d — %d active nodes, %d active edges, %s context",
                label, persona_id, len(persona["active_nodes"]),
                count_active(persona["agg_edges"]), patient_context)
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


def step_scenario(label: str, state: dict, session_idx: int) -> dict:
    """Draw this session's event.

    The patient reports on it; the interviewer and the analyst never see it.
    Seeds already used by this patient are excluded so they do not report the
    same kind of week twice.
    """
    used = set(state.get("used_scenario_seeds", []))
    scenario = sample_scenario(exclude_seeds=used)
    logger.info("[%s] scenario (session=%d) \u2014 %s", label, session_idx, scenario["id"])
    return {
        "scenario": scenario,
        "used_scenario_seeds": sorted(used | {scenario["seed_id"]}),
    }


def step_activation(label: str, state: dict) -> dict:
    """Zero the patient's activation map and fix this session's dynamics.

    One entry per self-report item, all cold. ActivationParams are the dials
    that decide how reactive this patient is and how long anything stays with
    them; defaults for now, sampled from severity later.
    """
    activation = from_self_report(state["self_report"])
    n_items = sum(len(v) for v in activation.values())
    logger.info("[%s] activation \u2014 %d items across %d nodes",
                label, n_items, len(activation))
    return {"activation": activation, "activation_params": ActivationParams(),
            "activation_trace": []}


def step_conversation(label: str, state: dict, llms: dict, agent_classes: dict,
                      num_turns: int) -> dict:
    """Run one session.

    The patient opens, reporting on the scenario \u2014 this is EMA, they are
    checking in, not answering a probe. Decay runs at the START of each
    subsequent turn, representing the gap since the last one: run at the end it
    would discount a tool call before the model ever spoke from it.
    """
    session_history = format_recent_transcripts(state.get("recent_transcripts", []))
    activation = state["activation"]
    params = state["activation_params"]
    trace = state["activation_trace"]

    patient = load_agent_class(agent_classes["patient"])(
        name=label, role="Patient", llm=llms["patient"],
        biography=state["biography"],
        scenario=state["scenario"]["text"],
        activation=activation,
        params=params,
        self_report=state["self_report"],
        obs_nodes=state["obs_nodes"],
        obs_edges=state["obs_edges"],
        session_history=session_history,
        trace=trace,
    )
    interviewer = load_agent_class(agent_classes["interviewer"])(
        name="Interviewer", role="Interviewer", llm=llms["interviewer"],
        memory_context=state.get("memory_context", ""),
    )
    logger.info("[%s] conversation \u2014 starting %d turns", label, num_turns)

    opening = patient.open_session()
    conversation = [{"speaker": "client", "turn_index": 0, "text": opening}]
    activation_by_turn = [snapshot(activation)]

    interviewer_response = interviewer.respond(opening)
    conversation.append({"speaker": "interviewer", "turn_index": 1,
                         "text": interviewer_response})

    for i in range(num_turns - 1):
        decay(activation, params)          # time passed since the last turn
        patient_response = patient.respond(interviewer_response)
        conversation.append({"speaker": "client", "turn_index": 2 + i * 2,
                             "text": patient_response})
        activation_by_turn.append(snapshot(activation))
        interviewer_response = interviewer.respond(patient_response)
        conversation.append({"speaker": "interviewer", "turn_index": 3 + i * 2,
                             "text": interviewer_response})

    live = sum(1 for turn in activation_by_turn for v in turn.values() for x in v.values() if x > 0)
    logger.info("[%s] conversation complete \u2014 %d messages, %d tool calls, "
                "%d live item-turns", label, len(conversation), len(trace), live)
    return {"conversation": conversation,
            "activation_trace": trace,
            "activation_by_turn": activation_by_turn}


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
