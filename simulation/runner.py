import json
import textwrap
import numpy as np
from pathlib import Path
from datetime import datetime
from config.config import Config
from langchain_openai import ChatOpenAI
from agents.client_agent import ClientAgent
from agents.analyst_agent import AnalystAgent
from agents.response_crafter_agent import ResponseCrafterAgent
from agents.persona_crafter_agent import PersonaCrafterAgent
from agents.dialogue_state_agent import DialogueStateAgent
from agents.validator_agent import ValidatorAgent
from data.input import get_available_ids, prepare_patient_components, get_observation_components


OUTPUT_DIR = Path("data/output")
MAX_VALIDATION_RETRIES = 3
WRAP_WIDTH = 100


def _wrap(text: str) -> str:
    """Wrap a long string at word boundaries."""
    return "\n".join(
        textwrap.fill(line, width=WRAP_WIDTH) if line.strip() else ""
        for line in text.splitlines()
    )


def _write_json(output: dict, f):
    """Write output JSON with blank lines between conversation turns."""
    conversation = output.pop("conversation")
    analysis = output.pop("analysis")

    f.write(json.dumps(output, indent=2, ensure_ascii=False)[:-1])  # strip closing }

    f.write(',\n\n  "conversation": [\n')
    for i, turn in enumerate(conversation):
        f.write("    " + json.dumps(turn, ensure_ascii=False))
        f.write(",\n\n" if i < len(conversation) - 1 else "\n")
    f.write("  ],\n\n")

    f.write('  "analysis": ' + json.dumps(analysis, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    f.write("\n}\n")


def _write_txt(output: dict, path: Path):
    """Write a human-readable text report with wrapped long strings."""
    SEP = "─" * WRAP_WIDTH

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PATIENT {output['patient_id']}  |  seed: {output['seed']}  |  {output['obs_datetime']}\n")
        f.write(SEP + "\n\n")

        f.write("PERSONA\n")
        f.write(SEP + "\n")
        f.write(_wrap(output["persona"]) + "\n\n")

        f.write("AGGREGATE EDGES\n")
        f.write(SEP + "\n")
        for edge, val in output["agg_edges"].items():
            f.write(f"  {edge}: {val:.4f}\n")
        f.write("\n")

        f.write("OBSERVATION NODES\n")
        f.write(SEP + "\n")
        for node, val in output["obs_nodes"].items():
            f.write(f"  {node}: {val}\n")
        f.write("\n")

        f.write("OBSERVATION EDGES\n")
        f.write(SEP + "\n")
        for edge, val in output["obs_edges"].items():
            f.write(f"  {edge}: {val}\n")
        f.write("\n")

        f.write("CONVERSATION\n")
        f.write(SEP + "\n")
        for turn in output["conversation"]:
            speaker = turn["speaker"].upper()
            f.write(f"\n[{turn['turn_index']}] {speaker}\n")
            f.write(_wrap(turn["text"]) + "\n")
        f.write("\n")

        f.write("ANALYSIS\n")
        f.write(SEP + "\n")
        f.write(json.dumps(output["analysis"], indent=2, ensure_ascii=False) + "\n")


def _to_serializable(obj):
    if isinstance(obj, dict):
        def _key(k):
            if isinstance(k, tuple):
                return " -- ".join(k)
            if isinstance(k, np.integer):
                return int(k)
            return k
        return {_key(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


class SimulationRunner:
    def __init__(self, num_patients: int, num_turns: int, model: str, seed: int,
                 patient_ids: list = None, obs_index: int = 0):
        self.num_patients = num_patients
        self.num_turns = num_turns
        self.model = model
        self.seed = seed
        self.patient_ids = patient_ids
        self.obs_index = obs_index
        np.random.seed(seed)
        self.llm = ChatOpenAI(model=self.model, api_key=Config.OPENAI_API_KEY)

    def _sample_patients(self) -> list:
        available = get_available_ids()
        return list(np.random.choice(available, size=self.num_patients, replace=False))

    def _generate_validated_message(self, dialogue_state, validator, persona, nodes, edges, label) -> str:
        message = None
        for attempt in range(MAX_VALIDATION_RETRIES):
            message = dialogue_state.create_message()
            result = validator.validate(persona=persona, message=message, nodes=nodes, edges=edges)
            if result.valid:
                print(f"[{label}] Initial message validated on attempt {attempt + 1}.")
                return message
            print(f"[{label}] Validation failed (attempt {attempt + 1}): {result.reason}")
        print(f"[{label}] Max retries reached. Using last generated message.")
        return message

    def run(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        patient_ids = self.patient_ids if self.patient_ids is not None else self._sample_patients()

        for patient_id in patient_ids:
            label = f"Client_{patient_id}"
            print(f"\n=== Patient {patient_id} ===")

            # ── Step 1: Aggregate EMA → Persona ──────────────────────────────
            _, agg_edges = prepare_patient_components(patient_id)  # agg_nodes always zero
            persona_crafter = PersonaCrafterAgent(
                name=f"{label}_PersonaCrafter", role="PersonaCrafter",
                llm=self.llm, patient_id=patient_id
            )
            persona = persona_crafter.create_persona()
            print(f"[{label}] Persona created.")

            # ── Step 2: Conversation EMA → observation nodes & edges ──────────
            obs_datetime, obs_nodes, obs_edges = get_observation_components(patient_id, self.obs_index)
            print(f"[{label}] Observation datetime: {obs_datetime}")

            # ── Step 3: Dialogue State Agent → candidate initial message ──────
            dialogue_state = DialogueStateAgent(
                name=f"{label}_DialogueState", role="DialogueState",
                llm=self.llm, patient_id=patient_id, persona=persona
            )

            # ── Step 4: Validator → validate (with retry on failure) ──────────
            validator = ValidatorAgent(
                name=f"{label}_Validator", role="Validator", llm=self.llm
            )
            initial_message = self._generate_validated_message(
                dialogue_state, validator, persona, obs_nodes, obs_edges, label
            )

            # ── Step 5: Client Agent ──────────────────────────────────────────
            client = ClientAgent(
                name=label, role="Patient", llm=self.llm,
                persona=persona, nodes=obs_nodes, edges=obs_edges,
                initial_message=initial_message
            )

            # ── Step 6: PTSD Bot (Response Crafter) ──────────────────────────
            bot = ResponseCrafterAgent(name="Bot", role="Bot", llm=self.llm)

            # ── Step 7: Conversation loop ─────────────────────────────────────
            conversation = []
            turn_index = 0

            conversation.append({"speaker": "client", "turn_index": turn_index, "text": initial_message})
            turn_index += 1

            bot_response = bot.respond(initial_message)
            conversation.append({"speaker": "bot", "turn_index": turn_index, "text": bot_response})
            turn_index += 1

            for _ in range(self.num_turns - 1):
                client_response = client.respond(bot_response)
                conversation.append({"speaker": "client", "turn_index": turn_index, "text": client_response})
                turn_index += 1

                bot_response = bot.respond(client_response)
                conversation.append({"speaker": "bot", "turn_index": turn_index, "text": bot_response})
                turn_index += 1

            # ── Step 8: Conversation Analyst ──────────────────────────────────
            analyst = AnalystAgent(name="Analyst", role="Analyst", llm=self.llm)
            conversation_text = "\n".join(
                f"{t['speaker'].upper()} (turn {t['turn_index']}): {t['text']}"
                for t in conversation
            )
            analysis = analyst.respond(conversation_text)
            print(f"[{label}] Analysis complete.")

            # ── Save output ───────────────────────────────────────────────────
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            output = _to_serializable({
                "patient_id": patient_id,
                "seed": self.seed,
                "obs_datetime": str(obs_datetime),
                "persona": persona,
                "agg_edges": agg_edges,
                "obs_nodes": obs_nodes,
                "obs_edges": obs_edges,
                "conversation": conversation,
                "analysis": analysis,
            })

            json_file = OUTPUT_DIR / f"conversation_{patient_id}_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                _write_json(output.copy(), f)

            txt_file = OUTPUT_DIR / f"conversation_{patient_id}_{timestamp}.txt"
            _write_txt(output, txt_file)

            print(f"[{label}] Saved to {json_file.name} + {txt_file.name}")
