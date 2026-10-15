"""
Generate scenes in bulk.

    python -m scenarios.scene.run --n 3
    python -m scenarios.scene.run --n 50 --seed 7

For each scene: sample the ingredients from the pools, have the director make
a card, check it with the card judge, then play it and narrate it.

A card that fails gets one retry with the judge's problems as feedback - unless
it failed on fit, which means the ingredients themselves can't work together,
so they are resampled instead. A scene whose ingredients still produce no
passing card after a few resamples is skipped.

Every finished scene is saved as soon as it is done, so an interrupted run
keeps everything it already paid for. Rejected cards are saved alongside, for
seeing what the judge is catching.
"""
import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

if __package__ is None:                       # allow `python scenarios/scene/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scenarios.scene.card_judge import CardJudge
from scenarios.scene.director import Director, SceneCard
from scenarios.scene.narrator import Narrator
from scenarios.scene.play import play_scene
from scenarios.scene.pools import (
    CIRCUMSTANCES, ENTITY_TYPES, PLACES, RELATIONSHIPS, SURROUNDINGS, TASK_TYPES, TIMES)

OUT = Path("data/scenes/scenes.json")
REJECTED = Path("data/scenes/rejected_cards.json")


def sample_ingredients(rng: random.Random, n_tasks: int, entity_chance: float) -> dict:
    return {
        "place": rng.choice(PLACES),
        "time": rng.choice(TIMES),
        "surroundings": rng.choice(SURROUNDINGS),
        "circumstance": rng.choice(CIRCUMSTANCES),
        "tasks": [(t, rng.choice(RELATIONSHIPS)) for t in rng.sample(TASK_TYPES, k=n_tasks)],
        "entity_type": rng.choice(ENTITY_TYPES) if rng.random() < entity_chance else None,
    }


def setting_text(ing: dict, card: SceneCard) -> str:
    """Built from the card, not the ingredients: the director may have dropped
    surroundings that don't fit the place, and nobody downstream should see them."""
    text = f"{ing['place']}, {ing['time']}."
    return f"{text} Surroundings: {card.surroundings}." if card.surroundings.strip() else text


def make_card(director: Director, judge: CardJudge, rng: random.Random, n_tasks: int,
              entity_chance: float, resamples: int, rejected: list) -> tuple | None:
    """A passing card and the ingredients it came from, or None if none passed."""
    for _ in range(resamples):
        ing = sample_ingredients(rng, n_tasks, entity_chance)
        card = director.direct(**ing)
        verdict = judge.check(setting_text(ing, card), card)

        if not verdict.passed and verdict.fit.passed:
            rejected.append({"ingredients": ing, "card": card.model_dump(),
                             "problems": verdict.problems, "then": "retried with feedback"})
            card = director.direct(**ing, feedback=verdict.problems)
            verdict = judge.check(setting_text(ing, card), card)

        if verdict.passed:
            return ing, setting_text(ing, card), card, verdict
        rejected.append({"ingredients": ing, "card": card.model_dump(),
                         "problems": verdict.problems, "then": "resampled"})
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="scenes to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--big", default="gpt-5.4",
                    help="model for the director's setup, the card judge and the narrator")
    ap.add_argument("--small", default="gpt-5.4-mini",
                    help="model for the per-turn cues and the actors")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--tasks", type=int, default=2, help="task types per scene")
    ap.add_argument("--entity-chance", type=float, default=0.5)
    ap.add_argument("--resamples", type=int, default=3,
                    help="ingredient draws to try before skipping a scene")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    director = Director(args.big, cue_model=args.small)
    judge = CardJudge(args.big)
    narrator = Narrator(args.big)
    scenes, rejected, skipped = [], [], 0

    def save() -> None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(scenes, indent=2, ensure_ascii=False), encoding="utf-8")
        REJECTED.write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        for i in range(args.n):
            made = make_card(director, judge, rng, args.tasks, args.entity_chance,
                             args.resamples, rejected)
            if made is None:
                skipped += 1
                print(f"  scene {i + 1}/{args.n}: skipped - no passing card")
                continue
            ing, setting, card, verdict = made

            script = play_scene(director, setting, card, args.turns, model=args.small)
            account = narrator.narrate(setting, card, script)

            scenes.append({
                "id": f"scene_{args.seed}_{i:03d}",
                "ingredients": ing,
                "setting": setting,
                "card": card.model_dump(),
                "card_verdict": verdict.model_dump(),
                "script": script,
                "account": account,
                "models": {"big": args.big, "small": args.small},
                "generated_at": date.today().isoformat(),
            })
            save()
            print(f"  scene {i + 1}/{args.n}: {ing['place']}, {ing['time']} - done")
    finally:
        save()
        print(f"\nwrote {OUT} - {len(scenes)} scenes, {skipped} skipped, "
              f"{len(rejected)} cards rejected along the way ({REJECTED})")


if __name__ == "__main__":
    main()
