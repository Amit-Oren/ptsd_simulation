"""
The scenario pool — the objective events a patient reports on each session.

Loads data/scenarios/scenarios.json, written by `python -m scenarios.generate`.
A simulation run never calls an LLM for a scenario: generation happens
offline, is validated there, and ships as data.

THE RULE THE POOL IS BUILT ON: scenario text states only what a camera would
record. Never a reaction, never an interpretation.

    WRONG   "You smelled alcohol at the party and felt a wave of panic."
    RIGHT   "You stayed about twenty minutes and left without saying goodbye."

The first has already done the cognitive graph's work, so there is nothing left
to measure. The second states facts a patient and an observer would both agree
on; whether the alcohol registered, and whether leaving early was avoidance, is
for the graph to decide. test_scenarios.py enforces this over the whole pool.

Nothing here is tagged with which cognitive components a scenario might engage.
A scenario labelled "Triggers" and then scored on whether Triggers fired would
be testing the label rather than the mechanism.
"""
import json
import random
from pathlib import Path

POOL_PATH = Path("data/scenarios/scenarios.json")

_POOL: list[dict] = (
    json.loads(POOL_PATH.read_text(encoding="utf-8"))
    if POOL_PATH.exists() else []
)


def all_scenarios() -> list[dict]:
    """Every scenario in the pool."""
    return list(_POOL)


def get_scenario(scenario_id: str) -> dict:
    for record in _POOL:
        if record["id"] == scenario_id:
            return record
    raise KeyError(f"unknown scenario {scenario_id!r}")


def sample_scenario(rng: random.Random | None = None,
                    exclude_events: set | None = None) -> dict:
    """One scenario, uniformly over the pool.

    `exclude_events` holds the event keys a patient has already had, so they
    do not report the same kind of thing happening twice across sessions.
    Keyed on event rather than place, since place is now optional context and
    not every scenario has one. Falls back to the whole pool once every event
    has been used.
    """
    if not _POOL:
        raise RuntimeError(
            f"{POOL_PATH} is empty or missing — run python -m scenarios.generate")
    rng = rng or random
    candidates = [r for r in _POOL if r["event_key"] not in (exclude_events or set())]
    return rng.choice(candidates or _POOL)


def format_for_prompt(scenario: dict) -> str:
    return scenario["text"]
