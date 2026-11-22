from pathlib import Path

import streamlit as st

from src.ui import initialize_session_state


ROOT = Path(__file__).parent

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

with (ROOT / "assets" / "style.css").open(encoding="utf-8") as stylesheet:
    st.markdown(f"<style>{stylesheet.read()}</style>", unsafe_allow_html=True)

initialize_session_state()

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">E</div>
          <div><div class="brand-title">EduPredict <span>AI</span></div>
          <div class="brand-caption">Learning intelligence platform</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Explore academic outcomes, engagement patterns, and model insights.")
    if st.session_state.dataset is not None:
        st.success(f"Dataset ready · {len(st.session_state.dataset):,} records")
        st.caption(f"Source: {st.session_state.get('dataset_name', 'Loaded dataset')}")
    else:
        st.info("Upload a dataset or load the optional demo in Dataset Explorer.")
    if st.session_state.model_bundle:
        st.success(f"Model ready · {len(st.session_state.model_bundle['models'])} classifier(s)")
    else:
        st.caption("No trained model loaded")
    st.markdown("---")
    st.caption("EDUPREDICT AI  ·  VERSION 1.0")

pages = [
    st.Page("pages/1_Dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("pages/2_Student_Prediction.py", title="Student Performance Prediction", icon=":material/psychology:"),
    st.Page("pages/3_Engagement_Analysis.py", title="Engagement Analysis", icon=":material/insights:"),
    st.Page("pages/4_Model_Comparison.py", title="Model Comparison", icon=":material/monitoring:"),
    st.Page("pages/5_Dataset_Explorer.py", title="Dataset Explorer", icon=":material/database:"),
    st.Page("pages/6_Reports.py", title="Reports and Downloads", icon=":material/download:"),
    st.Page("pages/7_About.py", title="About Project", icon=":material/school:"),
    st.Page("pages/8_Early_Warning_System.py", title="Early Warning System", icon=":material/notification_important:"),
    st.Page("pages/9_Student_Insights.py", title="Student Insights", icon=":material/person_search:"),
    st.Page("pages/10_Model_Training_Center.py", title="Model Training Center", icon=":material/model_training:"),
    st.Page("pages/11_AI_Assistant.py", title="AI Assistant", icon=":material/smart_toy:"),
]
navigation = st.navigation(pages)
navigation.run()
