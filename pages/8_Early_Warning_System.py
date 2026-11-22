import streamlit as st

from src.risk import assess_student_risk, student_identifier_column
from src.report_generator import csv_download, excel_download
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header(
    "Early Warning System",
    "Use model outcome probabilities and observed academic indicators to prioritize supportive educator review.",
)
frame = st.session_state.dataset
bundle = st.session_state.model_bundle
if frame is None:
    st.info("Load a dataset in Dataset Explorer before assessing student risk.")
    st.page_link("pages/5_Dataset_Explorer.py", label="Open Dataset Explorer", icon=":material/database:")
    st.stop()
if not bundle:
    st.info("Train a model in Model Training Center to enable model-based risk screening.")
    st.page_link("pages/10_Model_Training_Center.py", label="Open Model Training Center", icon=":material/model_training:")
    st.stop()

model_name = st.selectbox("Model used for screening", list(bundle["models"]))
try:
    risks = assess_student_risk(frame, bundle, model_name)
except ValueError as error:
    st.warning(str(error))
    st.caption("Risk categories are not assigned to unrecognized target classes.")
    st.stop()

identifier = student_identifier_column(frame)
risks["Student"] = (
    frame[identifier].astype(str).to_numpy()
    if identifier
    else [f"Record {index + 1}" for index in range(len(frame))]
)
if "Risk probability" in risks and risks["Risk probability"].notna().any():
    st.caption(
        "Logic: sum predicted probabilities for recognizable high-risk outcome classes plus half the probability "
        "for recognizable moderate-risk classes. High ≥ 60%; moderate ≥ 25% and < 60%; low < 25%. "
        "If target classes are not all semantically mapped, the predicted class alone is mapped; unknown outcomes remain unclassified."
    )
else:
    st.caption(
        "Logic: map only recognizable outcome labels (for example Fail/Withdrawn, Borderline, Pass). "
        "Unknown labels are not forced into a risk category."
    )
st.warning(
    "Risk flags are screening estimates for supportive educator review, not guaranteed outcomes, diagnoses, or "
    "grounds for punitive decisions. Engagement indicators are experimental cohort comparisons."
)

counts = (
    risks["Risk category"]
    .value_counts()
    .reindex(["Low Risk", "Moderate Risk", "High Risk", "Unclassified"], fill_value=0)
    .rename_axis("Risk category")
    .reset_index(name="Students")
)
st.markdown("### Risk overview")
metric_columns = st.columns(4)
for column, category in zip(metric_columns, counts["Risk category"]):
    with column:
        st.metric(category, f"{int(counts.loc[counts['Risk category'] == category, 'Students'].iloc[0]):,}")

left, right = st.columns(2)
with left:
    st.bar_chart(counts.set_index("Risk category"), color="#5967e8")
with right:
    class_counts = risks["Predicted outcome"].value_counts().rename_axis("Predicted outcome").to_frame("Students")
    st.bar_chart(class_counts, color="#7868d8")

st.markdown("### Students for supportive review")
categories = ["High Risk", "Moderate Risk", "Low Risk", "Unclassified"]
selected_categories = st.multiselect(
    "Risk categories",
    categories,
    default=["High Risk", "Moderate Risk"],
)
query = st.text_input("Search by student identifier", placeholder="Type an identifier")
display = risks[risks["Risk category"].isin(selected_categories)].copy()
if query:
    display = display[display["Student"].str.contains(query, case=False, na=False)]
ordered = ["High Risk", "Moderate Risk", "Low Risk", "Unclassified"]
display["_sort"] = display["Risk category"].map({category: index for index, category in enumerate(ordered)})
display = display.sort_values(["_sort", "Risk probability"], ascending=[True, False], na_position="last")
report = display[["Student", "Predicted outcome", "Risk probability", "Risk category", "Risk reasons", "Suggested support", "Model"]]
st.dataframe(prepare_dataframe_for_display(report), width="stretch", hide_index=True)
st.download_button("Download risk assessment CSV", csv_download(report), "student_risk_assessment.csv", "text/csv")
st.download_button("Download risk assessment Excel", excel_download(report), "student_risk_assessment.xlsx")
