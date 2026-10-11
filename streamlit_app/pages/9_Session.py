"""
9_Session.py — one session, end to end.

Everything that produced a transcript, on one page: who the patient is, what
they were and were not told, the graph that was sampled for them, the event
they were reporting on, and then the conversation with every tool call shown
beside the message it produced.

The point of the layout is the gap it makes visible. The left of a turn is what
fired inside the patient; the right is what they actually said. Reading them
together is how you tell whether the mechanism is driving the transcript or the
model is improvising around it.
"""
import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import pandas as pd
import streamlit as st

from components.sidebar import render_sidebar_selector

st.set_page_config(page_title="Session", layout="wide")
st.title("Session")

experiment_dir, data = render_sidebar_selector()
if data is None:
    st.info("Select an experiment and a session from the sidebar.")
    st.stop()

demographics = data.get("demographics", {})
scenario     = data.get("scenario", {})
trace        = data.get("activation_trace", [])
by_turn      = data.get("activation_by_turn", [])
params       = data.get("activation_params") or {}
obs_nodes    = data.get("obs_nodes", {})
obs_edges    = data.get("obs_edges", {})
self_report  = data.get("self_report", {})
conversation = data.get("conversation", [])

NODE_ORDER = ["Triggers", "Memory", "Threat",
              "Negative Appraisals", "Maladaptive Strategies"]


# ── header ────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Persona", data.get("persona_id", "?"))
c2.metric("Session", data.get("session_idx", "?"))
c3.metric("Context", data.get("patient_context", "?"))
c4.metric("Tool calls", len(trace))
c5.metric("Turns", len(conversation))

if data.get("patient_context") != "biography":
    st.warning(
        f"This session ran on the **{data.get('patient_context')}** vignette, an "
        "ablation arm. The patient was told its own symptoms, so low tool use is "
        "expected — that is what the arm measures."
    )

st.divider()


# ── who the patient is, and what the graph says ───────────────────────────────

left, right = st.columns([1, 1])

with left:
    st.subheader("Who they are")
    st.caption("Everything the patient was told about itself.")
    st.markdown(f"> {data.get('biography', '(none)')}")

    st.markdown("**Demographics** — used to generate the biography; `pcl5` was withheld")
    st.dataframe(
        pd.DataFrame([{"field": k, "value": v} for k, v in demographics.items()]),
        hide_index=True, use_container_width=True,
    )

with right:
    st.subheader("What was sampled")
    st.caption("None of this reached the patient's prompt. It shapes which tools "
               "exist and where activation travels.")

    rows = []
    for node in NODE_ORDER:
        items = self_report.get(node, [])
        rows.append({
            "node": node,
            "active": "yes" if obs_nodes.get(node) else "—",
            "items": ", ".join(i["key"] if isinstance(i, dict) else str(i)
                               for i in items) or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    edges = [{"from": p, "to": c, "weight": round(w, 3)}
             for p, children in obs_edges.items()
             for c, w in children.items() if w > 0]
    st.markdown(f"**Active edges this session** ({len(edges)})")
    if edges:
        st.dataframe(pd.DataFrame(sorted(edges, key=lambda e: -e["weight"])),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No edges active — nothing can cascade this session.")

    if params:
        st.markdown("**Activation parameters**")
        st.dataframe(pd.DataFrame([params]), hide_index=True, use_container_width=True)


# ── the event ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader("This week")
st.caption(f"`{scenario.get('seed_id', '?')}` · the interviewer and the analyst "
           "never see this")
st.info(scenario.get("text", "(no scenario)"))

d1, d2 = st.columns([1, 1])
with d1:
    dims = scenario.get("dimensions", {})
    if dims:
        st.markdown("**Dimensions varied**")
        st.dataframe(pd.DataFrame([{"dimension": k, "value": v} for k, v in dims.items()]),
                     hide_index=True, use_container_width=True)
with d2:
    if scenario.get("atomic_event"):
        st.markdown("**Human baselines** · ATOMIC-2020")
        st.caption(f"`{scenario['atomic_event']}`")
        st.markdown(
            f"- **why they actually did it** — {', '.join(scenario.get('other_intent', [])) or '—'}\n"
            f"- **how a typical person feels** — {', '.join(scenario.get('norm_feel', [])) or '—'}\n"
            f"- **what they typically do next** — {', '.join(scenario.get('norm_want', [])) or '—'}"
        )
        st.caption("Nobody in the simulation sees the first line. The patient has "
                   "to infer it from behaviour alone.")
    else:
        st.markdown("**Human baselines**")
        st.caption("This seed was written rather than drawn from ATOMIC, so it "
                   "carries no crowd annotations.")


# ── the conversation, with what fired beside it ───────────────────────────────

st.divider()
st.subheader("Conversation")
st.caption("Left: what fired inside the patient. Right: what they said.")

by_index: dict[int, list] = {}
for entry in trace:
    by_index.setdefault(entry.get("turn_index", -1), []).append(entry)

untagged = by_index.pop(-1, [])

for turn in conversation:
    idx = turn["turn_index"]
    is_patient = turn["speaker"] == "client"
    fired = by_index.get(idx, []) if is_patient else []

    fire_col, text_col = st.columns([1, 2])

    with fire_col:
        if fired:
            for entry in fired:
                st.markdown(
                    f"<div style='border-left:3px solid #C2560E;padding-left:10px;"
                    f"margin-bottom:10px;font-size:0.84em;line-height:1.45'>"
                    f"<b>{entry['node']}</b><br>"
                    f"<code>{entry['item']}</code> &nbsp;<b>{entry['level']:.2f}</b>"
                    + (f"<br><i>matched:</i> {entry['matched']}" if entry.get("matched") else "")
                    + "".join(
                        f"<br>↳ {c['from_item']} → {c['to_item']} "
                        f"<b>+{c['amount']:.3f}</b>"
                        for c in entry.get("cascade", [])[:3])
                    + "</div>",
                    unsafe_allow_html=True,
                )
        elif is_patient:
            st.markdown(
                "<div style='color:#9aa;font-size:0.8em;padding-left:10px'>"
                "nothing fired</div>", unsafe_allow_html=True)

    with text_col:
        with st.chat_message("user" if is_patient else "assistant"):
            st.markdown(f"**{'Patient' if is_patient else 'Interviewer'}**  \n{turn['text']}")

if untagged:
    st.caption(f"{len(untagged)} tool call(s) from a run predating turn tagging, "
               "so they could not be placed.")


# ── activation over the session ───────────────────────────────────────────────

st.divider()
st.subheader("Activation over the session")

if by_turn:
    items = sorted({(node, item)
                    for snap in by_turn
                    for node, d in snap.items() for item in d})
    matrix = pd.DataFrame(
        [[snap.get(node, {}).get(item, 0.0) for snap in by_turn] for node, item in items],
        index=[f"{node} · {item}" for node, item in items],
        # One snapshot per patient message, so the columns carry the
        # conversation's own turn indices (0, 2, 4 ...) rather than 0..n -
        # otherwise they do not line up with the transcript above.
        columns=[f"turn {i * 2}" for i in range(len(by_turn))],
    )
    live = matrix[matrix.max(axis=1) > 0]
    if len(live):
        st.dataframe(
            live.style.background_gradient(cmap="Oranges", vmin=0, vmax=1)
                      .format("{:.2f}"),
            use_container_width=True,
        )
        cold = len(matrix) - len(live)
        if cold:
            st.caption(f"{cold} item(s) never activated — parts of this patient "
                       "the conversation did not reach.")
    else:
        st.info("Nothing activated this session. The patient spoke entirely from "
                "the biography and the scenario.")
else:
    st.caption("No per-turn activation recorded for this session.")


# ── analysis ──────────────────────────────────────────────────────────────────

analysis = data.get("analysis")
if analysis:
    st.divider()
    with st.expander("Analyst output"):
        st.json(analysis)
