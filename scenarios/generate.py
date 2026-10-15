"""
Build the scenario pool. Run once, offline.

    python -m scenarios.generate --dry-run
    python -m scenarios.generate --per-event 10

The event is the required anchor - every scenario starts from one whole entry
in EVENT_TEMPLATES (pools.py), which already supplies a situation, a concrete
event, a response and an outcome, so the narrator has a small causal sequence
to work from rather than a bare gist it has to invent a shape for. Everything
else - place, who, when, activity, other_action, environment - is optional
context. A template may declare its own `compatible` overrides (pools.py) for
which values of a dimension make sense for it; where it doesn't, context is
drawn from the normal pools (who/activity still filtered by place category).

Two kinds of problem get different treatment, not just different wording:
  - a judge failure on PLAUSIBLE means the material itself doesn't hang
    together - no rewrite fixes a broken combination, so the material is
    redrawn from scratch and written fresh.
  - a near-duplicate of something already kept for this event is the same
    failure in spirit - rewriting the same material produces the same text,
    so it's redrawn too.
  - every other judge failure (factual, coherent, specific, natural) is a
    wording problem, repaired with one rewrite pass on the same material.
Each gets exactly one retry; a scenario that still fails after that is
dropped rather than retried indefinitely.

EVENT_TEMPLATES currently holds 8 templates as a starting batch, not a full
pool - extend it in pools.py, same situation/event/response/outcome shape,
before relying on this for real coverage.
"""
import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

if __package__ is None:                       # allow `python scenarios/generate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenarios.judge import Judge
from scenarios.narrator import Narrator
from scenarios.pools import (
    EVENT_TEMPLATES, POOLS, WHERE, OPTIONAL_FIELDS,
    WHO_GENERIC, ACTIVITY_GENERIC, who_pool, activity_pool,
)

OUT = Path("data/scenarios/scenarios.json")
REJECTED = Path("data/scenarios/rejected.json")
POOL_VERSION = "v3"

_PLACE_FILTERED = {"who": who_pool, "activity": activity_pool}
_GENERIC_ONLY = {"who": WHO_GENERIC, "activity": ACTIVITY_GENERIC}
_TEMPLATE_FIELDS = ("situation", "event", "response", "outcome")
_DUPLICATE_THRESHOLD = 0.6  # word-overlap (Jaccard) above this counts as a near-duplicate


def _compatible(event_key: str, field: str) -> list[str] | None:
    """This event's own override for `field`, if it declares one - None means
    no override, fall back to the normal pool/category behaviour."""
    return EVENT_TEMPLATES[event_key].get("compatible", {}).get(field)


def draw_material(event_key: str, rng: random.Random, cli_places: list[str] | None) -> dict:
    material = {k: v for k, v in EVENT_TEMPLATES[event_key].items() if k in _TEMPLATE_FIELDS}
    fields = list(OPTIONAL_FIELDS)
    place = None
    if "place" in fields:
        fields.remove("place")
        override = _compatible(event_key, "place")
        eligible = override if override is not None else (cli_places or WHERE)
        if eligible and rng.random() < 0.7:
            place = rng.choice(eligible)
            material["place"] = place
    for field in rng.sample(fields, k=min(len(fields), rng.randint(1, 3))):
        override = _compatible(event_key, field)
        if override is not None:
            pool = override
        elif field in _PLACE_FILTERED:
            pool = _PLACE_FILTERED[field](place) if place else _GENERIC_ONLY[field]
        else:
            pool = POOLS[field]
        if pool:                               # an empty override: never draw this field
            material[field] = rng.choice(pool)
    return material


def combinations(event_key: str, n_sampled: int, rng: random.Random,
                 cli_places: list[str] | None) -> list[dict]:
    return [draw_material(event_key, rng, cli_places) for _ in range(n_sampled)]


def _is_near_duplicate(text: str, already_kept: list[str]) -> bool:
    words = set(text.lower().split())
    if not words:
        return False
    for prev in already_kept:
        prev_words = set(prev.lower().split())
        if not prev_words:
            continue
        overlap = len(words & prev_words) / len(words | prev_words)
        if overlap >= _DUPLICATE_THRESHOLD:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrator", default="gpt-5.4-mini")
    ap.add_argument("--judge", default="gpt-5.4-mini",
                    help="a different model family from --narrator, since one "
                         "checking its own output shares the blind spots that "
                         "produced the problem")
    ap.add_argument("--per-event", type=int, default=6,
                    help="sampled scenarios per event (default: 6)")
    ap.add_argument("--events", type=str, default=None,
                    help="comma-separated event keys (default: all in EVENT_TEMPLATES)")
    ap.add_argument("--places", type=str, default=None,
                    help="comma-separated places eligible as context, for "
                         "events with no compatible-place override "
                         "(default: all in WHERE)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan, make no API calls")
    args = ap.parse_args()

    event_keys = args.events.split(",") if args.events else list(EVENT_TEMPLATES)
    cli_places = args.places.split(",") if args.places else None
    rng = random.Random(args.seed)
    plan = {e: combinations(e, args.per_event, rng, cli_places) for e in event_keys}
    total = sum(len(v) for v in plan.values())

    print(f"{len(event_keys)} events, {total} scenarios ({args.per_event} each)")
    if args.dry_run:
        for event_key, combos in plan.items():
            print(f"  {event_key:<24} {len(combos)}")
            m = combos[0]
            extra = ", ".join(f"{k}={v}" for k, v in m.items() if k not in _TEMPLATE_FIELDS)
            print(f"      e.g. {m['event']} | {extra or '(no context)'}")
        return

    narrator = Narrator(args.narrator, args.temperature)
    judge = Judge(args.judge)
    kept, dropped, repaired, redrawn = [], [], 0, 0

    def save() -> None:
        # Called after every event, and again on interrupt or error, so a
        # killed run keeps whatever it already paid API calls for instead of
        # throwing the whole batch away for the sake of the last one.
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
        if dropped:
            REJECTED.write_text(json.dumps(dropped, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    try:
        for event_key, combos in plan.items():
            kept_texts: list[str] = []
            for i, material in enumerate(combos):
                text = narrator.write(material)
                verdict = judge.check(material, text)
                duplicate = Judge.passed(verdict) and _is_near_duplicate(text, kept_texts)

                if not Judge.passed(verdict) or duplicate:
                    material_problem = duplicate or not verdict.plausible.passed
                    if material_problem:
                        # The material itself is the problem - redraw it rather
                        # than rewrite prose around something broken or already said.
                        material = draw_material(event_key, rng, cli_places)
                        text = narrator.write(material)
                        verdict = judge.check(material, text)
                        duplicate = Judge.passed(verdict) and _is_near_duplicate(text, kept_texts)
                        if Judge.passed(verdict) and not duplicate:
                            redrawn += 1
                    else:
                        text = narrator.rewrite(material, text, verdict.problems)
                        verdict = judge.check(material, text)
                        duplicate = Judge.passed(verdict) and _is_near_duplicate(text, kept_texts)
                        if Judge.passed(verdict) and not duplicate:
                            repaired += 1

                passed = Judge.passed(verdict) and not duplicate
                problems = verdict.problems + (["duplicate: too similar to one already kept"]
                                               if duplicate else [])
                record = {
                    "id": f"{event_key}__{i:03d}",
                    "event_key": event_key,
                    "place": material.get("place"),
                    "text": text,
                    "material": material,
                    "verdict": verdict.model_dump(),
                    "narrator": args.narrator,
                    "judge": args.judge,
                    "pool_version": POOL_VERSION,
                    "generated_at": date.today().isoformat(),
                }
                if passed:
                    kept.append(record)
                    kept_texts.append(text)
                else:
                    dropped.append(record)
                note = "" if passed else "  DROPPED: " + "; ".join(problems)[:48]
                print(f"  {event_key:<24} {i + 1}/{len(combos)}{note}", end="\r")
            print(" " * 90, end="\r")
            print(f"  {event_key:<24} done ({len(combos)})")
            save()
    finally:
        save()
        print(f"\nwrote {OUT}  -  {len(kept)} kept ({repaired} repaired, "
              f"{redrawn} redrawn), {len(dropped)} dropped")
        if dropped:
            print(f"rejected kept for inspection: {REJECTED}")


if __name__ == "__main__":
    main()
