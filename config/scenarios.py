"""
The scenario pool — the objective events a patient reports on each session.

Loads data/scenarios/scenarios.json, written once by
scripts/generate_scenarios.py. A simulation run never calls an LLM for a
scenario: generation happens offline, is validated there, and ships as data.
Same shape as the FORMA persona data.

THE RULE THE POOL IS BUILT ON: scenario text states only what a camera would
record. Never a reaction, never an interpretation.

    WRONG   "You smelled alcohol at the party and felt a wave of panic."
    RIGHT   "You stayed about twenty minutes and left without saying goodbye."

The first has already done the cognitive graph's work, so there is nothing left
to measure. The second states facts a patient and an observer would both agree
on; whether the alcohol registered, and whether leaving early was avoidance, is
for the graph to decide. test_scenarios.py enforces this over the whole pool.

Eight of the twelve seeds come from ATOMIC-2020 (Hwang et al., 2021). Its crowd
annotations — in data/atomic/seed_relations.json — are the normative anchor a
distortion measurement needs, since nobody here authored them:

    other_intent   xIntent   why the other person actually acted
    norm_feel      oReact    how a typical person feels afterwards
    norm_want      oWant     what a typical person does next

`other_intent` is the one the simulation holds and nobody inside it sees. The
patient has to infer the reason behind an ambiguous act from behaviour alone,
and the interviewer and analyst never learn it either.

Nothing here is tagged with which cognitive components a scenario might engage.
A scenario labelled "Triggers" and then scored on whether Triggers fired would
be testing the label rather than the mechanism.
"""
import json
import random
from pathlib import Path

POOL_PATH = Path("data/scenarios/scenarios.json")
_RELATIONS_PATH = Path("data/atomic/seed_relations.json")

_POOL: list[dict] = (
    json.loads(POOL_PATH.read_text(encoding="utf-8"))
    if POOL_PATH.exists() else []
)
_RELATIONS: dict = (
    json.loads(_RELATIONS_PATH.read_text(encoding="utf-8"))
    if _RELATIONS_PATH.exists() else {}
)


def _annotate(record: dict) -> dict:
    """Attach the ATOMIC baselines for this record's seed ({} for written seeds)."""
    relations = _RELATIONS.get(record["seed_id"], {}).get("relations", {})
    return {
        **record,
        "relations":    relations,
        "other_intent": relations.get("xIntent", []),
        "norm_feel":    relations.get("oReact", []),
        "norm_want":    relations.get("oWant", []),
    }


def all_scenarios() -> list[dict]:
    """Every scenario in the pool, annotated."""
    return [_annotate(r) for r in _POOL]


def get_scenario(scenario_id: str) -> dict:
    for record in _POOL:
        if record["id"] == scenario_id:
            return _annotate(record)
    raise KeyError(f"unknown scenario {scenario_id!r}")


def sample_scenario(rng: random.Random | None = None,
                    exclude_seeds: set | None = None) -> dict:
    """One scenario, uniformly over the pool.

    `exclude_seeds` holds the seeds a patient has already had, so they do not
    report the same kind of week twice across sessions. Falls back to the whole
    pool once every seed has been used.
    """
    if not _POOL:
        raise RuntimeError(
            f"{POOL_PATH} is empty or missing — run scripts/generate_scenarios.py")
    rng = rng or random
    candidates = [r for r in _POOL if r["seed_id"] not in (exclude_seeds or set())]
    return _annotate(rng.choice(candidates or _POOL))


def minimal_pairs(pair_id: str) -> list[dict]:
    """One minimal-pair group — identical specification but for one dimension.

    The dimensions are held fixed, but each member's prose was generated
    independently, so surface wording varies between them too. Controlled at
    the level of the specification, not of the sentence.
    """
    return [_annotate(r) for r in _POOL if r.get("pair_id") == pair_id]


def pair_groups() -> list[str]:
    return sorted({r["pair_id"] for r in _POOL if r.get("pair_id")})


def format_for_prompt(scenario: dict) -> str:
    return scenario["text"]
