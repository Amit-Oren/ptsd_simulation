import logging
import re
from .base_agent import BaseAgent
from config.prompts import (VIGNETTE_CRAFTER_PROMPT_CONTEXT,
                             VIGNETTE_CRAFTER_RETRY_PROMPT,
                             VIGNETTE_CRAFTER_USER_PROMPT)
from config.formulation_config import COMPONENT_SYNONYMS
from config.formatting import fmt_demographics, fmt_self_report

logger = logging.getLogger(__name__)


class VignetteCrafterAgent(BaseAgent):
    """Writes a clinical vignette grounded in the patient's synthetic formulation."""

    _VIGNETTE_SEP = "---VIGNETTE---"

    def __init__(self, name: str, role: str, llm,
                 demographics: dict, self_report: dict, formulation: dict):
        user_prompt = VIGNETTE_CRAFTER_USER_PROMPT.format(
            demographics=fmt_demographics(demographics),
            self_report=fmt_self_report(self_report),
            active_nodes=self._fmt_active_nodes(formulation["nodes"]),
            required_edges=self._fmt_required_edges(formulation["edges"]),
        )
        self._user_prompt = user_prompt
        super().__init__(name, role, VIGNETTE_CRAFTER_PROMPT_CONTEXT, llm)

    def create_vignette(self) -> str:
        raw = self.respond(self._user_prompt)
        self.vignette = self._extract_vignette(raw)
        logger.info("[%s] vignette written", self.name)
        return self.vignette

    def create_vignette_with_feedback(self, feedback: str) -> str:
        retry_prompt = VIGNETTE_CRAFTER_RETRY_PROMPT.format(
            feedback=feedback,
            previous_vignette=self.vignette,
        )
        self.reset_memory()
        raw = self.respond(retry_prompt)
        self.vignette = self._apply_diff(self.vignette, raw)
        logger.info("[%s] vignette revised after feedback", self.name)
        return self.vignette

    def _extract_vignette(self, raw: str) -> str:
        if self._VIGNETTE_SEP in raw:
            return raw.split(self._VIGNETTE_SEP, 1)[1].strip()
        return raw.strip()

    def _apply_diff(self, vignette: str, diff_output: str) -> str:
        insert_blocks = re.findall(
            r'<<<INSERT_AFTER>>>(.*?)<<<NEW_SENTENCE>>>(.*?)<<<END>>>',
            diff_output, re.DOTALL
        )
        if not insert_blocks:
            logger.warning("[%s] no valid patch blocks found — vignette unchanged", self.name)
            return vignette

        result = vignette
        for anchor, new_sentence in insert_blocks:
            anchor, new_sentence = anchor.strip(), new_sentence.strip()
            if anchor in result:
                result = result.replace(anchor, anchor + " " + new_sentence, 1)
            else:
                result = self._insert_near_source(result, new_sentence)
                logger.warning("[%s] INSERT_AFTER anchor not found — inserting near source component", self.name)
        return result

    def _insert_near_source(self, vignette: str, new_sentence: str) -> str:
        component = self._source_component(new_sentence)
        if component:
            synonyms = COMPONENT_SYNONYMS[component]
            sentences = re.split(r'(?<=[.!?])\s+', vignette)
            last_idx = next(
                (i for i in reversed(range(len(sentences)))
                 if any(syn.lower() in sentences[i].lower() for syn in synonyms)),
                -1
            )
            if last_idx != -1:
                sentences.insert(last_idx + 1, new_sentence)
                return " ".join(sentences)
        logger.warning("[%s] fallback: source component not found — appending to end", self.name)
        return vignette.rstrip() + " " + new_sentence

    def _source_component(self, sentence: str) -> str | None:
        s = sentence.lower()
        earliest, result = len(s), None
        for component, synonyms in COMPONENT_SYNONYMS.items():
            for syn in synonyms:
                pos = s.find(syn.lower())
                if 0 <= pos < earliest:
                    earliest, result = pos, component
        return result

    @staticmethod
    def _fmt_active_nodes(nodes: dict) -> str:
        return "\n".join(f"  - {n}" for n in nodes) or "  (none)"

    @staticmethod
    def _fmt_required_edges(edges: dict) -> str:
        lines = []
        for p, children in edges.items():
            for c, meta in children.items():
                if meta.get("strength", 0) > 0:
                    lines.append(f"  - {p} → {c}")
        return "\n".join(lines) or "  (none)"
