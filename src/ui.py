from __future__ import annotations

from datetime import date, datetime
from numbers import Number

import pandas as pd
import streamlit as st

from src.model_store import load_model_bundle


def initialize_session_state() -> None:
    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = load_model_bundle()
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def prepare_dataframe_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with heterogeneous object columns made Arrow-safe."""
    display_frame = frame.copy()
    for column in display_frame.columns:
        values = display_frame[column]
        if not pd.api.types.is_object_dtype(values.dtype):
            continue
        kinds: set[object] = set()
        for value in values.dropna():
            if isinstance(value, str):
                kinds.add("string")
            elif isinstance(value, bool):
                kinds.add("boolean")
            elif isinstance(value, Number):
                kinds.add("number")
            elif isinstance(value, (date, datetime, pd.Timestamp)):
                kinds.add("datetime")
            else:
                kinds.add(type(value))
        if len(kinds) > 1:
            display_frame[column] = values.astype("string")
    return display_frame


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        """
        <div class="app-header">
          <div class="app-header-mark">E</div>
          <div>
            <div class="app-header-title">EduPredict AI</div>
            <div class="app-header-subtitle">AI-Powered Student Performance and Engagement Analytics</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="eyebrow">EDUPREDICT AI</div><h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.info(message)
