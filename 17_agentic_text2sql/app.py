"""Berka branch desk. The same graph as notebook 3, with the SQL and the grid beside the sentence.

Students: README.md in this folder is the map. The buttons call text2sql.ask.
The first decision in the engine is marked START HERE.

    streamlit run app.py

What each suggested question is for:
  Greeting                 the database stays closed
  Not in the bank          HarborDesk leave is not in these tables, so the desk abstains
  Delete a loan            a write is refused before any SQL is run
  Card type                the answer is a label, so it does not wait
  Pisek accounts           the answer is a count, so a person signs it

Why the buttons sit in a form: Streamlit keeps a text box's old value until the
box loses focus. A plain button can send the original sentence. The form sends
the box and the button together, so an edit is the text the person typed.
"""

from __future__ import annotations

import text2sql as desk
import streamlit as st

st.set_page_config(page_title="Berka branch desk", layout="wide")

SUGGESTED = [
    ("Greeting", "Good morning"),
    ("Not in the bank", "How many annual leave days does a full-time HarborDesk employee receive?"),
    ("Delete a loan", "Delete every loan whose status is D."),
    ("Card type — no signature", "What type of credit card does the client with id 13539 own?"),
    ("Pisek accounts — needs a signature", "How many accounts are there in the district of Pisek?"),
]


def _grid(columns: list, rows: list) -> None:
    if not rows:
        return
    records = [dict(zip(columns, row)) for row in rows]
    st.dataframe(records, width="stretch", hide_index=True)


def _remember(values: dict) -> None:
    result = values.get("result") or {}
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": values.get("answer") or "",
            "steps": values.get("steps") or [],
            "sql": values.get("sql") or "",
            "columns": result.get("columns") or [],
            "rows": result.get("rows") or [],
        }
    )


def _finish(values: dict, payload: dict | None) -> None:
    """A payload means the graph is paused. No payload means the sentence can be shown."""
    if payload:
        st.session_state.pending = {
            "thread_id": values["thread_id"],
            "payload": payload,
            "steps": values.get("steps") or [],
        }
    else:
        st.session_state.pending = None
        _remember(values)


def _ask(question: str) -> None:
    values, payload = desk.ask(question)
    _finish(values, payload)


def _resume(action: str, text: str = "") -> None:
    """Continue the same thread. A new question would start a different graph run."""
    pending = st.session_state.pending
    values, payload = desk.ask("", thread_id=pending["thread_id"], resume={"action": action, "text": text})
    _finish(values, payload)


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.pending = None

st.title("Berka branch desk")
st.caption(
    "Local Ollama. The Czech bank file from FINCH. "
    "A count, a sum, or an average waits for a person. A write never runs."
)

with st.sidebar:
    st.header("Database")
    try:
        summary = desk.connect()
        try:
            counts = {
                "accounts": summary.execute("SELECT COUNT(*) FROM account").fetchone()[0],
                "transactions": summary.execute("SELECT COUNT(*) FROM trans").fetchone()[0],
            }
        finally:
            summary.close()
        st.write(f"{counts['accounts']} accounts · {counts['transactions']:,} transactions")
        st.caption("8 tables · read-only")
    except Exception as exc:
        st.warning(str(exc))
    st.caption(f"Chat {desk.CHAT_MODEL}")
    if not desk.ollama_alive():
        st.error("Ollama is not running.")
    st.divider()
    st.subheader("Try one")
    for label, question in SUGGESTED:
        # Disabled while a sentence is waiting, so a second question cannot abandon the signature.
        if st.button(label, disabled=st.session_state.pending is not None, width="stretch"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Working through the graph..."):
                try:
                    _ask(question)
                except Exception as exc:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"The desk stopped: {exc}", "steps": [], "sql": "", "columns": [], "rows": []}
                    )
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sql"):
            with st.expander("SQL"):
                st.code(message["sql"], language="sql")
            _grid(message.get("columns") or [], message.get("rows") or [])
        if message["role"] == "assistant" and message.get("steps"):
            with st.expander("Path"):
                for step in message["steps"]:
                    st.text(step)

pending = st.session_state.pending
if pending:
    payload = pending["payload"]
    st.warning(payload.get("reason") or "A person has to sign this sentence.")
    with st.expander("Path so far", expanded=False):
        for step in pending.get("steps") or []:
            st.text(step)
    st.code(payload.get("sql") or "", language="sql")
    _grid(payload.get("columns") or [], payload.get("rows") or [])
    # Form, not three loose buttons: the edited sentence and the click travel together.
    with st.form("signoff"):
        edited = st.text_area("Sentence", value=payload.get("draft") or "", height=120)
        left, middle, right = st.columns(3)
        approve = left.form_submit_button("Approve and send", type="primary", width="stretch")
        send_edit = middle.form_submit_button("Send the edited text", width="stretch")
        reject = right.form_submit_button("Reject and try another SQL", width="stretch")
    if approve:
        with st.spinner("Recording the approval..."):
            _resume("approve")
        st.rerun()
    if send_edit:
        with st.spinner("Recording the edit..."):
            _resume("edit", edited)
        st.rerun()
    if reject:
        with st.spinner("Trying another SQL..."):
            _resume("reject")
        st.rerun()

prompt = st.chat_input("Ask about the bank", disabled=st.session_state.pending is not None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Working through the graph..."):
        try:
            _ask(prompt)
        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"The desk stopped: {exc}", "steps": [], "sql": "", "columns": [], "rows": []}
            )
    st.rerun()
