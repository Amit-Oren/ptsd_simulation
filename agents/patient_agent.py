"""
The patient — a person with PTSD reporting on their week to a check-in chatbot.

What makes this agent different from an ordinary role-played persona is what it
is NOT told. Its prompt carries who the person is and what happened to them
this week, and nothing about their symptoms: no self-report items, no active
nodes, no edge weights, no instruction to express any of them. Symptom content
reaches the model only through tool results, so every symptom that appears in a
transcript has a matching activation event behind it, and the chain from
sampled graph to spoken word stays intact.

Three things follow from that:

  the toolbox   built from obs_nodes, so an inactive component has no tool and
                therefore cannot be expressed at all
  the cascade   runs inside each tool call, so the model reads the whole chain
                before it speaks rather than a turn late
  decay         runs outside this class, once per turn, because it is about
                time passing rather than anything the patient did

The activation map is passed in and mutated in place; the agent and its tools
hold the same dict. Whoever constructs the agent owns that state and can read
it between turns.
"""
import logging

from .base_agent import BaseAgent
from .patient_toolbox import build_toolbox
from config.prompts import PATIENT_AGENT_PROMPT

logger = logging.getLogger(__name__)


class PatientAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm,
                 biography: str,
                 scenario: str,
                 activation: dict,
                 params,
                 self_report: dict,
                 obs_nodes: dict,
                 obs_edges: dict,
                 session_history: str = "",
                 trace: list | None = None):
        self.activation = activation
        self.params = params
        self.scenario = scenario
        # Every tool call is appended here: which item fired, how hard, and the
        # cascade that followed. This is the session's record of what actually
        # happened inside the patient, as against what they said.
        self.trace = trace if trace is not None else []

        tools = build_toolbox(activation, self_report, obs_nodes, obs_edges,
                              params, self.trace)
        prompt = PATIENT_AGENT_PROMPT.format(
            biography=biography,
            scenario=scenario,
            session_history=session_history or "(this is your first conversation)",
        )
        super().__init__(name, role, prompt, llm, tools=tools)
        logger.info("[%s] patient built with %d tools", name, len(tools))

    def open_session(self) -> str:
        """The patient's first message.

        Unprompted contact: they opened the app because they felt like it, not
        because a weekly report is due. This is also the simulation's first
        measurement - the scenario states what happened, and what the patient
        says about it is already their reading of it.
        """
        return self.respond(
            "You have opened the app and started typing. Send your first message."
        )
