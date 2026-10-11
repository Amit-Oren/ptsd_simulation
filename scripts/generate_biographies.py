"""
Write a biography for each persona — who they are and what happened to them.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402

from simulation.factory import build_llm  # noqa: E402

OUT = Path("data/forma/biographies.json")
PERSONAS = Path("data/forma/personas")
PROMPT_VERSION = "v1"

SYSTEM = """You write short biographies of people, for research use.

You are given some facts about someone and the kind of event they went through.
Write 80 to 120 words covering two things and nothing else:

  WHO THEY ARE   a name that fits their background, what they do, who they live
                 with, something ordinary about the life they had
  WHAT HAPPENED  the event itself — when, where, what occurred, what it cost
                 them physically

Then stop. Write nothing about how they have been since.

This is the hard part, so be deliberate about it. Do NOT mention: sleep,
nightmares, memories, flashbacks, reminders, noises that bother them, places
they avoid, things they stopped doing, how they feel now, what they think about
themselves, whether they are coping, whether anyone has noticed a change,
whether they sought help. None of it.

A reader must finish knowing who this person is and what happened to them, and
knowing nothing whatever about how it affected them. That is the point: another
system decides that, and cannot if you have already written it in.

Invent the event's specifics so they fit the kind of event given. Use the real
first-person details — a name, a place, an occupation — so it reads as a life
rather than a case description.

Plain prose, third person, past tense for the event. No title, no markdown.
Reply with the biography and nothing else."""

USER = """Facts:
{facts}

Kind of event: {trauma_type}

Write the biography."""

RETRY = """That biography has a problem:
{problems}

Rewrite it, fixing only that. Keep the name and the event as they were."""


class Verdict(BaseModel):
    identity: bool = Field(description="Says who the person is: a name, what "
                                       "they do, something of their ordinary life.")
    event: bool = Field(description="Says what happened — when, where, what "
                                    "occurred.")
    clean: bool = Field(description="Says NOTHING about how they have been "
                                    "since: no symptom, no reaction, no "
                                    "avoidance, no change in them, no mention "
                                    "of coping or seeking help.")
    problems: list[str] = Field(default_factory=list,
                                description="One short sentence per problem, "
                                            "quoting the words at fault.")


JUDGE_SYSTEM = """You check short biographies written for research use.

Each should say who someone is and what happened to them, and stop there.

Three checks:

IDENTITY   Does it say who this person is — a name, an occupation, something
           of the life they had?

EVENT      Does it say what happened to them: when, where, what occurred?

CLEAN      Does it say anything at all about how they have been SINCE? This is
           the one that matters. Any of these fails it:
             sleep, nightmares, memories returning, flashbacks, reminders
             noises or places that affect them, routes or things they avoid
             anything they stopped doing, anyone noticing a change in them
             how they feel now, what they believe about themselves
             whether they are coping, whether they sought help
           Factual consequences of the event itself are fine: broken bones,
           days in hospital, a job lost, a move. The line is between what
           happened TO them and what happened IN them.
             "he spent four days in intensive care"         passes
             "he began taking longer routes to work"        FAILS
             "his colleagues noticed he was quieter"        FAILS
             "he has not spoken to anyone about it"         FAILS

Quote the exact words at fault. If you cannot quote them, it passes."""

JUDGE_USER = """The facts it was written from:
{facts}

The biography:
{text}"""


def _clean(content) -> str:
    return " ".join(str(content).strip().strip('"').split())


def _passed(v: Verdict) -> bool:
    return bool(v.identity and v.event and v.clean)


def _facts(demographics: dict) -> str:
    # pcl5 is withheld: it is a severity score, and severity is what the
    # activation mechanism is supposed to produce, not something the patient
    # arrives already narrating.
    keep = ("age", "gender", "ethnicity", "relationship_status", "occupation")
    return "\n".join(f"  - {k.replace('_', ' ')}: {demographics[k]}"
                     for k in keep if k in demographics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=str, default="1,2,3",
                    help="comma-separated persona ids (default: 1,2,3)")
    ap.add_argument("--limit", type=int, default=None,
                    help="instead, do the first N personas on disk")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--judge", default="gpt-4.1",
                    help="validates the biographies — a different family from "
                         "--model, since a model checking its own output "
                         "shares the blind spots that produced the problem")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    if args.limit:
        ids = sorted(int(p.stem.split("_")[1])
                     for p in PERSONAS.glob("persona_*.json"))[:args.limit]
    else:
        ids = [int(x) for x in args.personas.split(",")]

    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    llm = build_llm(args.model, temperature=args.temperature)
    judge = build_llm(args.judge, temperature=0.0).with_structured_output(Verdict)

    print(f"{len(ids)} personas -> {OUT.as_posix()}")
    kept = repaired = dropped = 0

    for pid in ids:
        persona = json.loads((PERSONAS / f"persona_{pid}.json").read_text(encoding="utf-8"))
        demographics = persona["demographics"]
        facts = _facts(demographics)
        msg = USER.format(facts=facts, trauma_type=demographics["trauma_type"])

        history = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": msg}]
        text = _clean(llm.invoke(history).content)
        verdict = judge.invoke([{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user", "content": JUDGE_USER.format(
                                    facts=facts, text=text)}])

        if not _passed(verdict):
            history += [{"role": "assistant", "content": text},
                        {"role": "user", "content": RETRY.format(
                            problems="\n".join("- " + p for p in verdict.problems))}]
            text = _clean(llm.invoke(history).content)
            verdict = judge.invoke([{"role": "system", "content": JUDGE_SYSTEM},
                                    {"role": "user", "content": JUDGE_USER.format(
                                        facts=facts, text=text)}])
            if _passed(verdict):
                repaired += 1

        record = {
            "persona_id": pid,
            "text": text,
            "demographics": demographics,
            "verdict": verdict.model_dump(),
            "generator": args.model,
            "judge": args.judge,
            "prompt_version": PROMPT_VERSION,
            "generated_at": date.today().isoformat(),
        }
        if _passed(verdict):
            existing[str(pid)] = record
            kept += 1
            status = "ok"
        else:
            dropped += 1
            status = "DROPPED: " + "; ".join(verdict.problems)[:60]
        print(f"  persona {pid:<4} {status}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{kept} kept ({repaired} repaired), {dropped} dropped — "
          f"{len(existing)} total in {OUT.as_posix()}")


if __name__ == "__main__":
    main()
