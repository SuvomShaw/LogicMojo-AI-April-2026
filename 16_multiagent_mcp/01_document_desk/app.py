"""Ledgerline document desk. The same graph as notebook 3, with the page beside the draft.

Students: README.md in this folder is the map. The buttons call document_rag.ask.
The first decision in the engine is marked START HERE.

    streamlit run app.py

What each suggested question is for:
  Greeting          the packet stays closed
  Not in the packet HarborDesk leave is not in these files, so the desk abstains
  Walker Ranch      the answer is in the paragraphs, so it does not wait
  PNC table         1.7 is billions of dollars and 5.4% is the loss rate; this one waits

Why the buttons sit in a form: Streamlit keeps a text box's old value until the
box loses focus. A plain button can send the original draft. The form sends the
box and the button together, so an edit is the text the person typed.
"""

from __future__ import annotations

import document_rag as desk


import streamlit as st

st.set_page_config(page_title="Ledgerline desk", layout="wide")

SUGGESTED = [
    ("Greeting", "Good morning"),
    ("Not in the packet", "How many annual leave days does a full-time HarborDesk employee receive?"),
    ("Walker Ranch figure", "In the Walker Ranch seismic figures, what do the red dots show?"),
    (
        "PNC table — needs a signature",
        "In PNC's adverse scenario covering Q3 2017 through Q3 2019, what is the estimated loss percentage for Commercial Real Estate?",
    ),
]


def _image(rel: str):
    if not rel:
        return
    path = desk.HERE / rel
    if path.exists():
        st.image(str(path), width="stretch")


def _remember(values: dict) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": values.get("answer") or "",
            "steps": values.get("steps") or [],
            "hits": values.get("hits") or [],
        }
    )


def _finish(values: dict, payload: dict | None) -> None:
    """A payload means the graph is paused. No payload means the answer can be shown."""
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


st.title("Ledgerline document desk")
st.caption(
    "Local Ollama. Real PDFs from UniDoc-Bench and one DOE Word file. "
    "A measured number or a table or a figure waits for a person."
)

with st.sidebar:
    st.header("Packet")
    try:
        summary = desk.corpus_summary()
        counts = summary["by_type"]
        st.write(f"{summary['documents']} documents · {desk._collection_count()} chunks")
        st.caption(
            f"{counts.get('text', 0)} text · {counts.get('table', 0)} tables · {counts.get('figure', 0)} figures"
        )
    except Exception as exc:
        st.warning(str(exc))
        
    st.caption(f"Chat {desk.CHAT_MODEL}")
    st.caption(f"Vectors {desk.EMBED_MODEL}")
    if not desk.ollama_alive():
        st.error("Ollama is not running.")
    st.divider()
    
    
    st.subheader("Try one")
    for label, question in SUGGESTED:
        # Disabled while a draft is waiting, so a second question cannot abandon the signature.
        if st.button(label, disabled=st.session_state.pending is not None, width="stretch"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Working through the graph..."):
                try:
                    _ask(question)
                except Exception as exc:
                    st.session_state.messages.append({"role": "assistant", "content": f"The desk stopped: {exc}", "steps": [], "hits": []})
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("steps"):
            with st.expander("Path"):
                for step in message["steps"]:
                    st.text(step)
        seen = set()
        for hit in message.get("hits") or []:
            image = hit.get("page_image") or ""
            if not image or image in seen:
                continue
            seen.add(image)
            st.caption(f"{hit['doc_id']} p.{hit['page']} {hit['element_type']} · {hit.get('license', '')}")
            _image(image)

pending = st.session_state.pending
if pending:
    payload = pending["payload"]
    st.warning(payload.get("reason") or "A person has to sign this draft.")
    with st.expander("Path so far", expanded=False):
        for step in pending.get("steps") or []:
            st.text(step)
    # Form, not three loose buttons: the edited draft and the click travel together.
    with st.form("signoff"):
        edited = st.text_area("Draft", value=payload.get("draft") or "", height=160)
        left, middle, right = st.columns(3)
        approve = left.form_submit_button("Approve and send", type="primary", width="stretch")
        send_edit = middle.form_submit_button("Send the edited text", width="stretch")
        reject = right.form_submit_button("Reject and search again", width="stretch")
    if approve:
        with st.spinner("Recording the approval..."):
            _resume("approve")
        st.rerun()
    if send_edit:
        with st.spinner("Recording the edit..."):
            _resume("edit", edited)
        st.rerun()
    if reject:
        with st.spinner("Searching again..."):
            _resume("reject")
        st.rerun()
    st.subheader("Evidence")
    seen = set()
    for source in payload.get("sources") or []:
        st.markdown(f"**{source['label']}**")
        st.caption(source.get("license") or "")
        image = source.get("page_image") or ""
        if image and image not in seen:
            seen.add(image)
            _image(image)
        st.text(source.get("excerpt") or "")

prompt = st.chat_input(
    "Ask about a document in the packet",
    disabled=st.session_state.pending is not None,
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Working through the graph..."):
        try:
            _ask(prompt)
        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"The desk stopped: {exc}", "steps": [], "hits": []}
            )
    st.rerun()
