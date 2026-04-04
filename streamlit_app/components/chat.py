"""
chat.py — renders a patient conversation as left/right chat bubbles.
"""

import base64
import html
import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

def _img_b64(filename: str) -> str:
    path = _STREAMLIT_APP_DIR / "assets" / filename
    return base64.b64encode(path.read_bytes()).decode()

_CLIENT_IMG = f"data:image/png;base64,{_img_b64('client.png')}"
_BOT_IMG    = f"data:image/png;base64,{_img_b64('bot.png')}"

import streamlit as st


def render_conversation(conversation: list[dict], moderation_log: list[dict] | None = None) -> None:
    if not conversation:
        st.info("No conversation turns available.")
        return

    # Build turn_index → raw text lookup from moderation log
    raw_by_turn: dict[int, str] = {}
    if moderation_log:
        for entry in moderation_log:
            raw_by_turn[entry["turn_index"]] = entry["raw"]

    for turn in conversation:
        speaker    = turn.get("speaker", "").lower()
        turn_index = turn.get("turn_index", "?")
        text       = turn.get("text", "")

        if speaker == "client":
            raw_text = raw_by_turn.get(turn_index)
            safe_text = html.escape(text)
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; align-items:flex-end; gap:8px; margin:8px 0;">
                    <div style="
                        max-width:70%;
                        background:#4E8D9C;
                        color:white;
                        padding:12px 16px;
                        border-radius:18px 18px 4px 18px;
                        font-size:14px;
                        line-height:1.5;
                    ">
                        {safe_text}
                        <div style="font-size:11px;opacity:0.75;margin-top:6px;text-align:right;">
                            Client · turn {turn_index}
                        </div>
                    </div>
                    <img src="{_CLIENT_IMG}" style="width:38px;height:38px;border-radius:50%;object-fit:cover;flex-shrink:0;"/>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if raw_text and raw_text != text:
                _, btn_col = st.columns([6, 1])
                with btn_col:
                    with st.popover("original", use_container_width=True):
                        st.markdown(
                            f"""<div style="font-size:13px;color:#555;line-height:1.5;">{html.escape(raw_text)}</div>""",
                            unsafe_allow_html=True,
                        )
        else:
            safe_text = html.escape(text)
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-start; align-items:flex-end; gap:8px; margin:8px 0;">
                    <img src="{_BOT_IMG}" style="width:38px;height:38px;border-radius:50%;object-fit:cover;flex-shrink:0;"/>
                    <div style="
                        max-width:70%;
                        background:#f0f0f0;
                        color:#2c3e50;
                        padding:12px 16px;
                        border-radius:18px 18px 18px 4px;
                        font-size:14px;
                        line-height:1.5;
                    ">
                        {safe_text}
                        <div style="font-size:11px;opacity:0.6;margin-top:6px;">
                            Bot · turn {turn_index}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
