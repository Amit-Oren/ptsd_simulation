"""
pipeline.py — parse simulation log lines into step statuses and render a
horizontal pipeline flow diagram with Plotly.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import plotly.graph_objects as go

# ── Constants ──────────────────────────────────────────────────────────────

# Ordered list of all possible steps
ALL_STEPS: list[str] = [
    "persona",
    "validate_persona",
    "dialogue_state",
    "validate_message",
    "conversation",
    "analysis",
]

# Human-readable labels for each step
STEP_LABELS: dict[str, str] = {
    "persona":           "Persona\nCrafter",
    "validate_persona":  "Validator\n(vignette)",
    "dialogue_state":    "Dialogue\nState",
    "validate_message":  "Validator\n(message)",
    "conversation":      "Client ↔\nBot",
    "analysis":          "Analyst",
}

# Box fill colours keyed by status
STATUS_COLORS: dict[str, str] = {
    "pending": "#bdc3c7",
    "running": "#f39c12",
    "done":    "#2ecc71",
    "failed":  "#e74c3c",
}

# ── Log parsing ────────────────────────────────────────────────────────────

# Mapping from log substring → (step, new_status)
_PATTERNS: list[tuple[str, str, str]] = [
    ("step_persona: starting",           "persona",          "running"),
    ("vignette written",                 "persona",          "done"),
    ("step_persona: done",               "persona",          "done"),
    ("step_validate_persona: starting",  "validate_persona", "running"),
    ("step_validate_persona: done",      "validate_persona", "done"),
    ("step_dialogue_state: starting",    "dialogue_state",   "running"),
    ("step_dialogue_state: done",        "dialogue_state",   "done"),
    ("step_validate_message: starting",  "validate_message", "running"),
    ("step_validate_message: done",      "validate_message", "done"),
    ("step_conversation: starting",      "conversation",     "running"),
    ("step_conversation: done",          "conversation",     "done"),
    ("step_analysis: starting",          "analysis",         "running"),
    ("analysis complete",                "analysis",         "done"),
    ("step_analysis: done",              "analysis",         "done"),
]


def parse_step_status(log_lines: list[str]) -> dict[str, str]:
    """
    Scan *log_lines* and derive the current status of every pipeline step.

    Parameters
    ----------
    log_lines : list[str]
        Lines read so far from simulation.log.

    Returns
    -------
    dict mapping step name → status string ("pending" | "running" | "done" | "failed")
    """
    status: dict[str, str] = {step: "pending" for step in ALL_STEPS}

    for line in log_lines:
        # If the run has saved output, mark everything as done
        if "saved \u2192" in line or "saved ->" in line:
            for step in ALL_STEPS:
                status[step] = "done"
            break

        for pattern, step, new_status in _PATTERNS:
            if pattern in line:
                # Only advance status — never go backwards
                current = status[step]
                rank = {"pending": 0, "running": 1, "done": 2, "failed": 2}
                if rank.get(new_status, 0) >= rank.get(current, 0):
                    status[step] = new_status

    return status


# ── Diagram rendering ──────────────────────────────────────────────────────

def render_pipeline(
    pipeline_steps: list[str],
    status: dict[str, str] = None,
) -> go.Figure:
    """
    Render a horizontal flow diagram showing only the steps in *pipeline_steps*.

    Parameters
    ----------
    pipeline_steps : list[str]
        Ordered list of step names to display (subset of ALL_STEPS).
    status : dict[str, str], optional
        Current status for each step.  Defaults to all "pending".

    Returns
    -------
    go.Figure
    """
    if status is None:
        status = {}

    # Filter to only steps present in this pipeline, preserving order
    visible_steps = [s for s in ALL_STEPS if s in pipeline_steps]

    n = len(visible_steps)
    if n == 0:
        return go.Figure()

    # Layout parameters
    box_w   = 1.4
    box_h   = 0.7
    gap     = 0.5          # horizontal gap between boxes
    step_x  = box_w + gap  # x-distance between box centres
    y_centre = 0.0

    shapes: list[dict] = []
    annotations: list[dict] = []

    for i, step in enumerate(visible_steps):
        cx = i * step_x
        x0, x1 = cx - box_w / 2, cx + box_w / 2
        y0, y1 = y_centre - box_h / 2, y_centre + box_h / 2

        step_status = status.get(step, "pending")
        fill_color  = STATUS_COLORS.get(step_status, STATUS_COLORS["pending"])

        # Box
        shapes.append(
            dict(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor=fill_color,
                line=dict(color="#2c3e50", width=1.5),
                layer="below",
            )
        )

        # Step label
        annotations.append(
            dict(
                x=cx,
                y=y_centre,
                xref="x",
                yref="y",
                text=STEP_LABELS.get(step, step).replace("\n", "<br>"),
                showarrow=False,
                font=dict(size=11, color="#2c3e50"),
                align="center",
            )
        )

        # Arrow to next box
        if i < n - 1:
            annotations.append(
                dict(
                    ax=x1,
                    ay=y_centre,
                    axref="x",
                    ayref="y",
                    x=x1 + gap,
                    y=y_centre,
                    xref="x",
                    yref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2,
                    arrowcolor="#7f8c8d",
                )
            )

    total_width = (n - 1) * step_x
    x_margin    = box_w / 2 + 0.3

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            visible=False,
            range=[-x_margin, total_width + x_margin],
        ),
        yaxis=dict(
            visible=False,
            range=[-0.8, 0.8],
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig
