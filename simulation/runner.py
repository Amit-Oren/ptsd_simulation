import json
import numpy as np
from pathlib import Path
from datetime import datetime
from config.config import Config
from langchain_openai import ChatOpenAI
from agents.client_agent import ClientAgent
from agents.analyst_agent import AnalystAgent
from agents.response_crafter_agent import ResponseCrafterAgent
from data.input import sample_patients, prepare_patient_components


OUTPUT_DIR = Path("data/output")


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
                 subnums: list = None, row_index: int = None, conversation_pattern: dict = None):
        self.num_patients = num_patients
        self.num_turns = num_turns
        self.model = model
        self.seed = seed
        self.subnums = subnums
        self.row_index = row_index
        self.conversation_pattern = conversation_pattern
        np.random.seed(seed)
        self.llm = ChatOpenAI(model=self.model, api_key=Config.OPENAI_API_KEY)

    def run(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        subnums = self.subnums if self.subnums is not None else sample_patients(self.num_patients)

        for subnum in subnums:
            id, schedule, nodes, edges = prepare_patient_components(subnum, self.row_index)
            print(f"Patient {subnum} , id: {id} , schedule: {schedule} , nodes: {nodes} , edges: {edges}")

            patient = ClientAgent(name=f"Client_{subnum}", role="Patient", llm=self.llm, id=id, schedule=schedule, nodes=nodes, edges=edges, conversation_pattern=self.conversation_pattern)

            bot = ResponseCrafterAgent(name="Bot", role="Bot", llm=self.llm)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"conversation_{subnum}_{timestamp}.json"
            # --- Conversation ---
            conversation = []
            turn_index = 0
            message = "Hello, how are you?"
            conversation.append({"speaker": "bot", "turn_index": turn_index, "text": message})
            for _ in range(self.num_turns):
                patient_response = patient.respond(message)
                conversation.append({"speaker": "client", "turn_index": turn_index, "text": patient_response})
                turn_index += 1

                bot_response = bot.respond(patient_response)
                conversation.append({"speaker": "bot", "turn_index": turn_index, "text": bot_response})
                turn_index += 1

                message = bot_response

            # --- Analysis ---
            analyst = AnalystAgent(name="Analyst", role="Analyst", llm=self.llm)
            conversation_history = str(conversation)
            analysis = analyst.respond(conversation_history)

            output = _to_serializable({
                "id": id,
                "seed": self.seed,
                "schedule": schedule,
                "nodes": nodes,
                "edges": edges,
                "conversation_pattern": patient.conversation_pattern,
                "conversation": conversation,
                "analysis": analysis,
            })
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
