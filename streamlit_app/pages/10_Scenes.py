"""
10_Scenes.py - read the scene dataset, one scene at a time.

Reads data/scenes/scenes.json (written by `python -m scenarios.scene.run`) and
nothing from the simulation.
"""
import html
import json
from pathlib import Path

import streamlit as st

SCENES_DIR = Path(__file__).resolve().parents[2] / "data" / "scenes"
MAIN = "the main character"

st.set_page_config(page_title="Scenes", layout="wide")
st.title("Scenes")

st.markdown("""
<style>
.who { display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.8em;
       font-weight:600; margin-right:6px; }
.who-main { background:#dbe7ff; color:#1d3a8a; }
.who-secondary { background:#ffe7cc; color:#8a4b00; }
.who-entity { background:#e6e6e6; color:#444; }
</style>
""", unsafe_allow_html=True)


def load(name: str) -> list:
    path = SCENES_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


scenes = load("scenes.json")
if not scenes:
    st.info("No scenes yet. Run `python -m scenarios.scene.run --n 3`.")
    st.stop()

labels = [f"{s['ingredients']['place']}, {s['ingredients']['time']}" for s in scenes]
idx = st.sidebar.radio("Scene", range(len(scenes)), format_func=lambda i: labels[i])
scene = scenes[idx]
card, verdict = scene["card"], scene["card_verdict"]


# ── account ───────────────────────────────────────────────────────────────────

st.subheader("Account")
st.info(scene["account"])


# ── card and judge ────────────────────────────────────────────────────────────

left, right = st.columns([3, 2])

with left:
    st.subheader("Card")
    main = card["main"]
    lines = [f"**Main character:** {main['brief']} - {main['circumstance']}"]
    lines += [f"**{c['role']}:** {c['task']}" for c in card["secondary"]]
    lines += [f"**{e['name']}:** {e['task']}" for e in card["entities"]]
    st.markdown("  \n".join(lines))

with right:
    st.subheader("Judge")
    for key, label in (("fit", "Fit"), ("lands", "Lands"), ("neutral_main", "Neutral main")):
        check = verdict[key]
        if check["passed"]:
            st.success(label)
        else:
            st.error(f"{label}: {check['quote']}")


# ── agents ────────────────────────────────────────────────────────────────────
# What each agent was given for this scene, rebuilt from the saved record.

st.subheader("Agents")
st.caption("What each agent was given, and the model it ran on.")

ing, models, setting = scene["ingredients"], scene["models"], scene["setting"]
big, small = models["big"], models["small"]


def known(relationship: str) -> str:
    low = relationship.lower()
    return "not known to you" if ("stranger" in low or "service" in low) else "known to you"


agents = [
    (f"Director - set-up · {big}", [
        f"**Place / time:** {ing['place']}, {ing['time']}",
        f"**Surroundings:** {ing['surroundings']}",
        f"**Circumstance:** {ing['circumstance']}",
        *[f"**Task type:** {t} ({r})" for t, r in ing["tasks"]],
        f"**Entity type:** {ing.get('entity_type') or '-'}",
    ]),
    (f"Card judge · {big}", [
        f"**Setting:** {setting}",
        "**The scene card** (above)",
    ]),
    (f"Director - turn cues · {small}", [
        f"**Setting:** {setting}",
        "**The scene card** and **the script so far**, every turn",
    ]),
    (f"Main character · {small}", [
        f"**Setting:** {setting}",
        f"**Who they are:** {card['main']['brief']}",
        f"**Goal:** {card['main']['goal']}",
        f"**Circumstance:** {card['main']['circumstance']}",
        "**The script so far**, every turn - no one's task",
    ]),
    *[(f"{c['role']} · {small}", [
        f"**Setting:** {setting}",
        f"**To the main character:** {c['relationship']}",
        f"**Private task:** {c['task']}",
        "**The script so far**, every turn",
    ]) for c in card["secondary"]],
    *[(f"{e['name']} (entity) · {small}", [
        f"**Setting:** {setting}",
        f"**Task:** {e['task']}",
        "**The script so far**",
    ]) for e in card["entities"]],
    (f"Narrator · {big}", [
        f"**Setting:** {setting}",
        f"**You are:** {card['main']['brief']}",
        *[f"**Cast:** {c['role']} ({known(c['relationship'])})" for c in card["secondary"]],
        "**The full script** - not the tasks or the card",
    ]),
]

for title, lines in agents:
    with st.expander(title):
        st.markdown("  \n".join(lines))


# ── script ────────────────────────────────────────────────────────────────────

st.subheader("Script")
for line in scene["script"]:
    who, kind = line["who"], line.get("kind", "character")
    cls = "who-entity" if kind == "entity" else ("who-main" if who == MAIN else "who-secondary")
    st.markdown(f'<span class="who {cls}">{html.escape(who)}</span> {html.escape(line["text"])}',
                unsafe_allow_html=True)


# ── rejected cards ────────────────────────────────────────────────────────────

rejected = load("rejected_cards.json")
with st.expander(f"Cards the judge rejected in this run ({len(rejected)})"):
    for r in rejected:
        ing = r["ingredients"]
        st.markdown(f"**{ing['place']}, {ing['time']}** - {r['then']}  \n" + "  \n".join(r["problems"]))
