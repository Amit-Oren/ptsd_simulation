"""
Pipeline step functions. Each step receives the current patient state dict,
updates it, and returns the updated dict.
"""
import logging
from agents.persona_crafter_agent import PersonaCrafterAgent
from agents.dialogue_state_agent import DialogueStateAgent
from agents.validator_agent import ValidatorAgent
from data.input.input import get_session_components, extract_aggregated_edges, get_demographics, get_self_report
from simulation.factory import load_agent_class

logger = logging.getLogger(__name__)


def step_persona(label: str, patient_id, llms: dict, persona_context: bool = False) -> dict:
    agg_edges = extract_aggregated_edges(patient_id)

    persona_crafter = PersonaCrafterAgent(
        name=f"{label}_PersonaCrafter", role="PersonaCrafter",
        llm=llms["persona_crafter"], patient_id=patient_id,
        use_demographics=persona_context, use_self_report=persona_context,
    )
    demographics = get_demographics(patient_id)
    self_report  = get_self_report(patient_id)
    persona = persona_crafter.create_persona()
    return {
        "persona":          persona,
        "persona_attempts": [{"output": persona, "passed": True, "feedback": None}],
        "agg_edges":        agg_edges,
        "demographics":     demographics,
        "self_report":      self_report,
        "_persona_crafter": persona_crafter,
    }


def step_validate_persona(label: str, state: dict, validator: ValidatorAgent, max_retries: int) -> dict:
    edges = state["agg_edges"]
    strong   = sum(1 for v in edges.values() if v >= 0.7)
    moderate = sum(1 for v in edges.values() if 0.4 <= v < 0.7)
    forbidden = sum(1 for v in edges.values() if v == 0.0)
    logger.info("[%s] Validator (vignette) — context: %d strong, %d moderate, %d forbidden edges",
                label, strong, moderate, forbidden)

    persona, attempts = validator.validate_with_retry(
        initial_output=state["persona"],
        output_type="vignette",
        context={"edges": edges},
        retry_fn=state["_persona_crafter"].create_persona_with_feedback,
        max_retries=max_retries,
        label=label,
    )
    return {"persona": persona, "persona_attempts": attempts}


def step_analyse_vignette(label: str, state: dict, llms: dict, agent_classes: dict) -> dict:
    analyst = load_agent_class(agent_classes["vignette_analyst"])(
        name=f"{label}_VignetteAnalyst", role="VignetteAnalyst", llm=llms["analyst"],
    )
    vignette_analysis = analyst.respond(state["persona"])
    logger.info("[%s] Vignette analysis — complete", label)
    return {"vignette_analysis": vignette_analysis}


def step_dialogue_state(label: str, patient_id, state: dict, llms: dict, obs_index: int) -> dict:
    obs_datetime, obs_nodes, obs_edges = get_session_components(patient_id, obs_index)

    dialogue_state = DialogueStateAgent(
        name=f"{label}_DialogueState", role="DialogueState",
        llm=llms["dialogue_state"], patient_id=patient_id,
        persona=state["persona"], obs_index=obs_index,
    )
    initial_message = dialogue_state.create_message()
    return {
        "obs_datetime": obs_datetime,
        "obs_nodes": obs_nodes,
        "obs_edges": obs_edges,
        "initial_message": initial_message,
        "message_attempts": [{"output": initial_message, "passed": True, "feedback": None}],
        "_dialogue_state": dialogue_state,
    }


def step_validate_message(label: str, state: dict, validator: ValidatorAgent, max_retries: int) -> dict:
    initial_message, attempts = validator.validate_with_retry(
        initial_output=state["initial_message"],
        output_type="dialogue",
        context={"persona": state["persona"], "nodes": state["obs_nodes"], "edges": state["obs_edges"]},
        retry_fn=state["_dialogue_state"].create_message_with_feedback,
        max_retries=max_retries,
        label=label,
    )
    return {"initial_message": initial_message, "message_attempts": attempts}


def step_conversation(label: str, state: dict, llms: dict, agent_classes: dict, num_turns: int) -> dict:
    client = load_agent_class(agent_classes["client"])(
        name=label, role="Patient", llm=llms["client"],
        persona=state["persona"], nodes=state["obs_nodes"], edges=state["obs_edges"],
        initial_message=state["initial_message"],
    )
    bot = load_agent_class(agent_classes["bot"])(
        name="Bot", role="Bot", llm=llms["bot"],
    )
    logger.info("[%s] Conversation — starting %d turns", label, num_turns)

    conversation = [{"speaker": "client", "turn_index": 0, "text": state["initial_message"]}]
    logger.info("[%s] conversation turn 0 — client (initial message)", label)

    bot_response = bot.respond(state["initial_message"])
    conversation.append({"speaker": "bot", "turn_index": 1, "text": bot_response})
    logger.info("[%s] conversation turn 1 — bot", label)

    for i in range(num_turns - 1):
        client_response = client.respond(bot_response)
        conversation.append({"speaker": "client", "turn_index": 2 + i * 2, "text": client_response})
        logger.info("[%s] conversation turn %d — client", label, 2 + i * 2)

        bot_response = bot.respond(client_response)
        conversation.append({"speaker": "bot", "turn_index": 3 + i * 2, "text": bot_response})
        logger.info("[%s] conversation turn %d — bot", label, 3 + i * 2)

    logger.info("[%s] Conversation — complete (%d turns, %d messages)", label, num_turns, len(conversation))
    return {"conversation": conversation}


def step_conversation_moderated(label: str, state: dict, llms: dict, agent_classes: dict, num_turns: int) -> dict:
    client = load_agent_class(agent_classes["client"])(
        name=label, role="Patient", llm=llms["client"],
        persona=state["persona"], nodes=state["obs_nodes"], edges=state["obs_edges"],
        initial_message=state["initial_message"],
    )
    moderator = load_agent_class(agent_classes["moderator"])(
        name="Moderator", role="Moderator", llm=llms["moderator"], persona=state["persona"],
    )
    bot = load_agent_class(agent_classes["bot"])(
        name="Bot", role="Bot", llm=llms["bot"],
    )
    logger.info("[%s] Conversation (moderated) — starting %d turns", label, num_turns)

    moderation_log = []

    initial_text = moderator.moderate_message(state["initial_message"], state["obs_nodes"], state["obs_edges"])
    moderation_log.append({"turn_index": 0, "raw": state["initial_message"], "moderated": initial_text})
    logger.info("[%s] conversation turn 0 — client raw: %s", label, state["initial_message"])
    logger.info("[%s] conversation turn 0 — client moderated: %s", label, initial_text)

    conversation = [{"speaker": "client", "turn_index": 0, "text": initial_text}]

    bot_response = bot.respond(initial_text)
    conversation.append({"speaker": "bot", "turn_index": 1, "text": bot_response})
    logger.info("[%s] conversation turn 1 — bot: %s", label, bot_response)

    for i in range(num_turns - 1):
        client_response = client.respond(bot_response)
        moderated_response = moderator.moderate_message(client_response, state["obs_nodes"], state["obs_edges"])
        moderation_log.append({"turn_index": 2 + i * 2, "raw": client_response, "moderated": moderated_response})
        logger.info("[%s] conversation turn %d — client raw: %s", label, 2 + i * 2, client_response)
        logger.info("[%s] conversation turn %d — client moderated: %s", label, 2 + i * 2, moderated_response)

        conversation.append({"speaker": "client", "turn_index": 2 + i * 2, "text": moderated_response})

        bot_response = bot.respond(moderated_response)
        conversation.append({"speaker": "bot", "turn_index": 3 + i * 2, "text": bot_response})
        logger.info("[%s] conversation turn %d — bot: %s", label, 3 + i * 2, bot_response)

    logger.info("[%s] Conversation — complete (%d turns, %d messages)", label, num_turns, len(conversation))
    return {"conversation": conversation, "moderation_log": moderation_log}


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
