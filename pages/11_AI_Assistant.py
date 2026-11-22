import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.ai_assistant import generate_assistant_response
from src.ai_context import build_assistant_context
from src.ui import initialize_session_state, page_header


initialize_session_state()
page_header(
    "AI Assistant",
    "Ask about current-session student performance, engagement, risk screening, model results, or dataset quality.",
)

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

frame = st.session_state.dataset
student_index = None
if frame is not None and not frame.empty:
    student_options = ["No specific student"] + [f"Record {index + 1}" for index in range(len(frame))]
    selected_student = st.selectbox("Student context (optional)", student_options)
    if selected_student != "No specific student":
        student_index = student_options.index(selected_student) - 1
else:
    st.caption("No dataset is loaded. General explanations are available; dataset-specific facts will be reported as unavailable.")

with st.expander("Privacy and assistant mode", expanded=False):
    st.write(
        "The local fallback uses only current-session summaries. When an API key is configured, the assistant sends "
        "your question and a small relevant context summary—not the full dataset. Selected-student context omits "
        "identifiers and common direct identifiers. Risk and prediction outputs are decision-support estimates."
    )
    try:
        api_key_available = bool(
            os.getenv("EDUPREDICT_AI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or st.secrets.get("EDUPREDICT_AI_API_KEY")
            or st.secrets.get("OPENAI_API_KEY")
        )
    except StreamlitSecretNotFoundError:
        api_key_available = bool(os.getenv("EDUPREDICT_AI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    st.caption(
        "Configured AI service is available for requests."
        if api_key_available
        else "No API key configured; replies use the clearly labeled local rule-based assistant."
    )

if st.button("Clear chat", key="clear_ai_chat"):
    st.session_state.ai_chat_history = []
    st.rerun()

for message in st.session_state.ai_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(message.get("source", ""))

with st.form("ai_assistant_form", clear_on_submit=True):
    question = st.text_area(
        "Your question",
        placeholder="For example: Which model performed best on the held-out test data?",
        height=90,
        max_chars=2000,
    )
    submitted = st.form_submit_button("Send", type="primary", width="stretch")

if submitted:
    question = question.strip()
    if not question:
        st.warning("Enter a question before sending.")
    else:
        st.session_state.ai_chat_history.append({"role": "user", "content": question})
        context = build_assistant_context(
            question=question,
            frame=frame,
            bundle=st.session_state.model_bundle,
            prediction_history=st.session_state.prediction_history,
            student_index=student_index,
        )
        try:
            with st.spinner("EduPredict AI is preparing a response..."):
                response, source = generate_assistant_response(question, context)
        except (ValueError, TypeError) as error:
            response = f"I could not prepare a response for this question: {error}"
            source = "Local fallback"
        st.session_state.ai_chat_history.append(
            {"role": "assistant", "content": response, "source": source}
        )
        st.rerun()
