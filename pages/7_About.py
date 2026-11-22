import streamlit as st

from src.ui import initialize_session_state, page_header


initialize_session_state()
page_header("About Project", "Artificial Intelligence-Based Student Performance and Engagement Prediction System")
st.markdown(
    """
    <div class="welcome-card">
      <div class="eyebrow">ACADEMIC PROJECT · EDUPREDICT AI</div>
      <h2>Evidence-informed learning analytics</h2>
      <p>EduPredict AI is a local educational analytics platform for exploring student data, comparing
      classification models, and examining cohort-relative learning activity. It is an academic demonstration
      and research prototype, not an automated decision system.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
left, right = st.columns(2)
with left:
    st.markdown("### Problem statement")
    st.write("Academic and engagement data can be distributed across files and may be difficult to interpret consistently. This project demonstrates how structured data analysis and supervised machine learning can support early review of learning patterns.")
    st.markdown("### Objectives")
    st.write("- Explore student datasets and identify data quality issues.\n- Train and evaluate four classifiers on a shared held-out split.\n- Provide cohort-relative engagement summaries when multiple activity measures exist.\n- Export transparent evaluation and analysis reports.")
    st.markdown("### Proposed methodology")
    st.write("Selected predictors are imputed and transformed in a fitted preprocessing pipeline; a stratified split is used when class counts allow it. Models are evaluated on the same held-out test records using weighted precision, recall, and F1, accuracy, optional supported ROC-AUC, and confusion matrices.")
    st.markdown("### Data preprocessing")
    st.write("Exact duplicate source rows are removed for training; missing numeric values use training-set medians, missing categorical values use training-set modes, numeric predictors are standardized, and categorical features are one-hot encoded. The target is not included in the preprocessing feature matrix.")
with right:
    st.markdown("### Machine learning algorithms")
    st.write("Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers.")
    st.markdown("### Dataset information")
    st.write("The app accepts CSV and Excel data. OULAD student information can be joined to aggregated assessment and VLE activity, optionally using course length and activity-type metadata. OULAD does not provide every commonly discussed feature; absent attendance or prior-result fields are not fabricated. Synthetic demo data is explicitly labeled.")
    st.markdown("### Evaluation metrics")
    st.write("Accuracy, weighted precision, weighted recall, weighted F1-score, confusion matrices, and ROC-AUC only when class probabilities and class representation support it. Best model ranking defaults to held-out weighted F1.")
    st.markdown("### Applications")
    st.write("Teaching demonstrations, exploratory cohort analysis, model comparison, and discussion of supportive, data-informed academic review.")
    st.markdown("### Limitations")
    st.write("Predictions depend on the quality, representativeness, and timing of uploaded data. Classifier outputs can reflect dataset bias and should not be used as automatic decisions. Engagement scores are relative ranks over selected indicators, not validated measures of learning or motivation; feature importance is not causation. Synthetic results are not research findings.")
    st.markdown("### Future enhancements")
    st.write("Add longitudinal validation, calibration and fairness assessments, institution-specific evaluation, authenticated role-based access, and educator-validated intervention workflows.")
st.markdown("### System architecture")
st.write(
    "Streamlit provides the interface and session state. Pandas/NumPy handle uploads and aggregation. Scikit-learn "
    "pipelines fit preprocessing and classifiers; Joblib stores the complete model bundle and metadata. Plotly "
    "renders interactive analytics and ReportLab generates PDF exports."
)
st.markdown("### Academic project details")
st.write(
    "**Student:** Karthik S. Tevari  \n"
    "**Department:** CSE (AI & ML)  \n"
    "**College:** Nagarjuna College of Engineering and Technology, Bengaluru  \n"
    "**Guide:** Prof. NithyaKalyani A."
)
st.markdown("**Technology:** Python 3.11 · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · Joblib · ReportLab")
