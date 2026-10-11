"""
Generate the scenario pool, once, offline.

Each scenario is a seed crossed with one combination of dimension values. An
LLM writes the prose — a template cannot reconcile "a manager" with "at their
home, after dark, no warning at all" into something that reads naturally, which
is why this step exists at all.

Output is data/scenarios/scenarios.json. A simulation run never calls an LLM
for a scenario: it reads this file. Generate, gate, ship — the same shape as
the FORMA persona data.

Two kinds of combination are produced:

  sampled   random dimension values, for variety. Most of the pool.
  paired    everything held fixed except one dimension, which takes every
            value in turn. These are minimal pairs: if a patient reads two of
            them differently, the varied dimension is the only thing that
            could have caused it.

Every generated scenario is checked for leaked reactions before it is written.
A scenario that says how the person felt has already done the cognitive
graph's work and is useless as a measurement, so a failure is dropped and
reported rather than silently kept.

    python scripts/generate_scenarios.py
    python scripts/generate_scenarios.py --per-seed 12 --pair-on expectedness
    python scripts/generate_scenarios.py --model gpt-4o --dry-run
"""
import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402

from config.scenario_seeds import SEEDS, dimensions_for, facet_of  # noqa: E402
from simulation.factory import build_llm  # noqa: E402

OUT = Path("data/scenarios/scenarios.json")
PROMPT_VERSION = "v1"

# Words that mean the prose has interpreted the event instead of reporting it.
BANNED = [
    "felt", "feeling", "feels", "panic", "anxious", "anxiety", "afraid",
    "scared", "frightened", "upset", "angry", "ashamed", "guilty", "distress",
    "overwhelm", "triggered", "reminded you", "flashback", "intrusive",
    "uneasy", "uncomfortable", "nervous", "worried", "dread", "tense",
    "couldn't cope", "struggled", "unsettl",
]

SYSTEM = """You write short factual accounts of ordinary events, for research use.

You will be given something that happened and the circumstances it happened in.
Write it as three or four short sentences, addressed to the person it happened
to, as "you".

THE RULE THAT MATTERS: write only what a camera and microphone would have
recorded. Never how anyone felt. Never what anything meant. Never why anyone
did what they did.

  WRONG   You felt uneasy when he didn't explain.
  WRONG   It was unsettling, and reminded you of the accident.
  WRONG   He wanted to reassure you.
  RIGHT   He didn't say what it was about. You waited outside for ten minutes.

The account must leave the reader genuinely unable to tell whether this was a
good thing, a bad thing, or nothing at all. That ambiguity is the point of the
exercise: another system will decide what it meant, and it cannot do that if
you have already decided.

Work all the given circumstances in naturally. Do not list them. If two of them
sit oddly together, write around the tension rather than stating it.

Plain prose. No markdown, no quotation marks around the whole thing, no title.
Reply with the account and nothing else."""

USER = """What happened: {gist}

Circumstances:
{circumstances}

Write the account."""

RETRY = """That account has a problem:
{problems}

Rewrite it, fixing only that. Keep everything else as it was."""


# A different model from the generator, deliberately: one checking its own
# output shares the blind spots that produced the problem.
#
# The keyword check catches named emotions. It does not catch "he looked at you
# a moment longer than he needed to" - no banned word, no broken rule, but an
# interpretation wearing a factual coat, which would hand the patient a reading
# before the mechanism ever runs. That is what the judge is for.

class Verdict(BaseModel):
    coherent: bool = Field(description="No internal contradiction: times, places, "
                                       "durations and people are consistent.")
    factual: bool = Field(description="States only what a camera and microphone "
                                      "would record. No feelings, no motives, no "
                                      "meanings, no implied significance.")
    natural: bool = Field(description="Reads as an account of a real occasion "
                                      "rather than a list of circumstances.")
    problems: list[str] = Field(default_factory=list,
                                description="One short sentence per problem. "
                                            "Empty if all three pass.")


JUDGE_SYSTEM = """You check short factual accounts of everyday events for research use.

You are shown the event the account was asked to describe, the circumstances it
was given, and the account itself.

WHAT WAS GIVEN IS NOT A FAULT. The writer was required to include the event and
every circumstance. If the event is "someone replies answering only part of what
you asked", then "the reply answered only part of it" is the assignment. Judge
only what the account ADDS beyond what it was given.

Work through three checks in order. For each, do the procedure, do not form a
general impression.

COHERENT
  Take each given circumstance one at a time and find where the account states
  it. Then ask whether anything else in the account contradicts it.
    given "nobody else was around" + "you waited with the others"   FAILS
    given "a couple of minutes" + an event that cannot be that brief  FAILS
    "after dark" alongside "early morning"                           FAILS
  A circumstance the account simply left out is not a contradiction.

FACTUAL  (be strictest here)
  Scan for words that smuggle in a judgement. These are the usual ones:
    only · still · even · finally · unusually · just · already · merely
    seemed · appeared · as if · longer than · more than · pointedly
    without a word · refused to · failed to · made a point of
  Each one is a flag, not an automatic failure: ask whether it is reporting or
  evaluating. "You waited only twenty minutes" evaluates. "There was only one
  chair" counts.
  Then ask the main question: does the account state, or strongly imply, how
  anyone felt, why anyone acted, or what any of it meant?
    "he looked at you a moment longer than he needed to"   FAILS
    "the room went quiet when you walked in"               FAILS
    "they did not explain"                                 passes
    "he looked at you, then at the floor"                  passes
  A reader must finish unable to tell whether this was good, bad, or nothing,
  beyond whatever the event itself already carries.

NATURAL
  Read each sentence and check it parses. A sentence that is garbled, or that
  repeats a circumstance it has already stated in different words, fails.
    "you were nearby when people were talking close by"    FAILS
  Mild awkwardness and plain prose are fine. This check is for sentences that
  do not work, not for style.

Quote the exact words at fault in each problem you report. If you cannot quote
them, it is not a problem and the account passes."""


JUDGE_USER = """What the account was asked to describe:
{gist}

Circumstances it was given:
{circumstances}

The account:
{text}"""


def _clean(content) -> str:
    return " ".join(str(content).strip().strip('"').split())


def _passed(v) -> bool:
    return bool(v.coherent and v.factual and v.natural)


def _judge(judge, gist: str, circumstances: str, text: str):
    return judge.invoke([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            gist=gist, circumstances=circumstances, text=text)}])


def leaked(text: str) -> list[str]:
    low = text.lower()
    return [w for w in BANNED if w in low]


def combinations(seed_id: str, n_sampled: int, pair_on: list[str],
                 rng: random.Random) -> list[dict]:
    """Sampled combinations plus, for each named dimension, a full sweep."""
    pools = dimensions_for(seed_id)
    out = []

    for _ in range(n_sampled):
        out.append({"dims": {d: rng.choice(v) for d, v in pools.items()},
                    "pair_id": None, "varied": None})

    for dim in pair_on:
        if dim not in pools:
            continue
        base = {d: rng.choice(v) for d, v in pools.items()}
        for value in pools[dim]:
            out.append({"dims": {**base, dim: value},
                        "pair_id": f"{seed_id}__{dim}", "varied": dim})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini",
                    help="model that writes the scenarios")
    ap.add_argument("--judge", default="gpt-4.1",
                    help="model that validates them - use a different "
                         "family from --model, since a model checking its "
                         "own output shares the blind spots that produced "
                         "the problem")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--per-seed", type=int, default=8,
                    help="sampled combinations per seed (default: 8)")
    ap.add_argument("--pair-on", nargs="*",
                    default=["expectedness", "others_present"],
                    help="dimensions to sweep fully, producing minimal pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be generated, make no API calls")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    plan = {s: combinations(s, args.per_seed, args.pair_on, rng) for s in SEEDS}
    total = sum(len(v) for v in plan.values())

    print(f"{len(SEEDS)} seeds, {total} scenarios "
          f"({args.per_seed} sampled each + sweeps on {', '.join(args.pair_on)})")
    if args.dry_run:
        for seed_id, combos in plan.items():
            paired = sum(1 for c in combos if c["pair_id"])
            print(f"  {seed_id:<20} {len(combos) - paired} sampled + {paired} paired")
            d = combos[0]["dims"]
            print(f"      e.g. {', '.join(f'{k}={v}' for k, v in list(d.items())[:4])}")
        return

    llm = build_llm(args.model, temperature=args.temperature)
    judge = build_llm(args.judge, temperature=0.0).with_structured_output(Verdict)
    scenarios, dropped = [], []
    repaired = 0

    for seed_id, combos in plan.items():
        seed = SEEDS[seed_id]
        for i, combo in enumerate(combos):
            lines = [f"  - {facet_of(d)}: {v}" for d, v in combo["dims"].items()]
            circumstances = "\n".join(lines)
            msg = USER.format(gist=seed["gist"], circumstances=circumstances)
            history = [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": msg}]
            text = _clean(llm.invoke(history).content)

            bad = leaked(text)
            verdict = None
            if not bad:
                verdict = _judge(judge, seed["gist"], circumstances, text)
                if not _passed(verdict):
                    # One repair attempt, with the judge's reasons. Cheaper than
                    # regenerating blind, and it records what needed fixing.
                    history += [{"role": "assistant", "content": text},
                                {"role": "user", "content": RETRY.format(
                                    problems="\n".join("- " + x for x in verdict.problems))}]
                    text = _clean(llm.invoke(history).content)
                    bad = leaked(text)
                    verdict = _judge(judge, seed["gist"], circumstances, text)
                    if _passed(verdict) and not bad:
                        repaired += 1
            record = {
                "id": f"{seed_id}__{i:03d}",
                "seed_id": seed_id,
                "text": text,
                "dimensions": combo["dims"],
                "pair_id": combo["pair_id"],
                "varied_dimension": combo["varied"],
                "atomic_event": seed["atomic_event"],
                "generator": args.model,
                "judge": args.judge,
                "prompt_version": PROMPT_VERSION,
                "generated_at": date.today().isoformat(),
            }
            failed = bool(bad) or (verdict is not None and not _passed(verdict))
            if verdict is not None:
                record["verdict"] = verdict.model_dump()
            if failed:
                record["leaked"] = bad
                dropped.append(record)
            else:
                scenarios.append(record)
            why = ", ".join(bad or (verdict.problems if verdict else []))
            note = "  DROPPED: " + why[:48] if failed else ""
            print(f"  {seed_id:<20} {i + 1}/{len(combos)}{note}", end="\r")
        print(" " * 80, end="\r")
        print(f"  {seed_id:<20} done ({len(combos)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}  -  {len(scenarios)} kept "
          f"({repaired} repaired on retry), {len(dropped)} dropped")

    if dropped:
        rej = OUT.with_name("rejected.json")
        rej.write_text(json.dumps(dropped, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"rejected scenarios kept for inspection: {rej}")


if __name__ == "__main__":
    main()
