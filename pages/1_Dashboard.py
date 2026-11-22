import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import find_column, is_identifier_column
from src.engagement import score_engagement, suggest_activity_columns
from src.prediction import global_feature_importance
from src.risk import assess_student_risk
from src.ui import initialize_session_state, page_header


initialize_session_state()
page_header("Student Performance Dashboard", "AI-Powered Student Performance and Engagement Analytics")
frame = st.session_state.dataset
bundle = st.session_state.model_bundle

if frame is None:
    st.markdown(
        '<div class="welcome-card"><div class="eyebrow">WELCOME TO EDUPREDICT AI</div>'
        '<h2>Turn learning data into meaningful support.</h2>'
        '<p>Upload an academic dataset to explore outcomes, train classifiers, and identify engagement patterns. '
        'Start with the Dataset Explorer, or use its clearly labeled synthetic demo to test the workflow.</p></div>',
        unsafe_allow_html=True,
    )
    st.info("Dataset status: No dataset loaded · Model status: " + ("Trained model loaded" if bundle else "No trained model"))
    links = st.columns(2)
    links[0].page_link("pages/5_Dataset_Explorer.py", label="Open Dataset Explorer", icon=":material/database:")
    links[1].page_link("pages/10_Model_Training_Center.py", label="Open Model Training Center", icon=":material/model_training:")
    st.stop()

score_column = find_column(frame, ("assessment score", "average score", "mean score", "mark"))
attendance_column = find_column(frame, ("attendance",))
activity_columns = suggest_activity_columns(frame)
target_column = bundle.get("target") if bundle else find_column(frame, ("final result", "performance", "grade", "outcome"))
engagement = None
if len(activity_columns) >= 2:
    try:
        engagement = score_engagement(frame, activity_columns)
    except ValueError:
        engagement = None

risk_result = None
best_model = None
if bundle:
    best_model = max(bundle["metrics"], key=lambda name: bundle["metrics"][name]["f1"])
    if all(column in frame.columns for column in bundle["feature_columns"]):
        try:
            risk_result = assess_student_risk(frame, bundle, best_model)
        except ValueError:
            risk_result = None

pass_percentage = None
if target_column and target_column in frame.columns:
    labels = frame[target_column].dropna().astype(str).str.strip().str.lower()
    if labels.isin({"pass", "passed", "distinction", "merit"}).any():
        pass_percentage = float(labels.isin({"pass", "passed", "distinction", "merit"}).mean() * 100)

high_risk_count = (
    int((risk_result["Risk category"] == "High Risk").sum())
    if risk_result is not None
    else None
)
metric_values = [
    ("Total students", f"{len(frame):,}", "Current dataset"),
    (
        "Average assessment",
        f"{frame[score_column].mean():.1f}" if score_column and pd.api.types.is_numeric_dtype(frame[score_column]) else "—",
        score_column or "No numeric assessment score identified",
    ),
    (
        "Average engagement indicator",
        f"{engagement['Engagement Score'].mean():.1f}/100" if engagement is not None else "—",
        "Cohort-relative; experimental, not validated",
    ),
    ("High-risk students", f"{high_risk_count:,}" if high_risk_count is not None else "—", "Requires mapped trained target classes"),
    ("Pass percentage", f"{pass_percentage:.1f}%" if pass_percentage is not None else "—", "Observed target labels only"),
    ("Best model", best_model or "—", "Ranked by held-out weighted F1"),
]
for column, (label, value, help_text) in zip(st.columns(6), metric_values):
    with column:
        st.metric(label, value, help=help_text)

dataset_name = st.session_state.get("dataset_name", "Uploaded dataset")
model_status = (
    f"Model ready · {len(bundle['models'])} algorithm(s) · target `{bundle['target']}`"
    if bundle
    else "No trained model"
)
st.markdown(
    f"**Dataset status:** Ready · **Source:** {dataset_name} · **Dimensions:** {len(frame):,} × {len(frame.columns):,} "
    f"· **Training status:** {model_status}"
)
if bundle and bundle.get("trained_at"):
    st.caption(
        f"Last model training (UTC): {bundle['trained_at']} · Training dataset: "
        f"{bundle.get('dataset_name', 'Not recorded')}"
    )
if bundle and bundle.get("dataset_name") != dataset_name:
    st.warning(
        "The active data source differs from the model's training dataset. Confirm that feature definitions and units match before interpreting estimates."
    )
if risk_result is None and bundle:
    st.info("Risk summary unavailable: the target labels may not support risk mapping, or model features are missing.")

st.markdown("### Performance and engagement overview")
if frame.empty:
    st.info("The loaded dataset contains no student records.")
else:
    left, right = st.columns(2)
    with left:
        if score_column and pd.api.types.is_numeric_dtype(frame[score_column]):
            chart = px.histogram(frame.dropna(subset=[score_column]), x=score_column, nbins=24, color_discrete_sequence=["#5967e8"])
            chart.update_layout(title="Assessment score distribution", xaxis_title=score_column, yaxis_title="Records")
            st.plotly_chart(chart, width="stretch")
        elif target_column and target_column in frame:
            values = frame[target_column].fillna("Missing").astype(str).value_counts().rename_axis("Outcome").reset_index(name="Records")
            chart = px.bar(values, x="Outcome", y="Records", color="Outcome", color_discrete_sequence=px.colors.qualitative.Pastel)
            chart.update_layout(title="Student performance distribution", showlegend=False)
            st.plotly_chart(chart, width="stretch")
        else:
            st.info("No numeric score or selected outcome is available for a performance chart.")
    with right:
        if engagement is not None:
            counts = (
                engagement["Engagement Level"]
                .value_counts()
                .reindex(["Low Engagement", "Moderate Engagement", "High Engagement"], fill_value=0)
                .rename_axis("Engagement")
                .reset_index(name="Students")
            )
            chart = px.bar(
                counts,
                x="Engagement",
                y="Students",
                color="Engagement",
                color_discrete_map={"Low Engagement": "#e4777a", "Moderate Engagement": "#e3a442", "High Engagement": "#45a787"},
            )
            chart.update_layout(title="Student engagement distribution", showlegend=False)
            st.plotly_chart(chart, width="stretch")
        else:
            st.info("Select two or more numeric activity indicators in Engagement Analysis.")

if risk_result is not None:
    st.markdown("### Risk category distribution")
    risk_counts = (
        risk_result["Risk category"]
        .value_counts()
        .reindex(["Low Risk", "Moderate Risk", "High Risk", "Unclassified"], fill_value=0)
        .rename_axis("Risk category")
        .reset_index(name="Students")
    )
    chart = px.bar(
        risk_counts,
        x="Risk category",
        y="Students",
        color="Risk category",
        color_discrete_map={"Low Risk": "#45a787", "Moderate Risk": "#e3a442", "High Risk": "#e4777a", "Unclassified": "#9aa4b6"},
    )
    chart.update_layout(showlegend=False)
    st.plotly_chart(chart, width="stretch")

if target_column and target_column in frame:
    outcome_counts = (
        frame[target_column].dropna().astype(str).value_counts().rename_axis("Outcome").reset_index(name="Students")
    )
    outcome_chart = px.bar(outcome_counts, x="Outcome", y="Students", color="Outcome", color_discrete_sequence=px.colors.qualitative.Pastel)
    outcome_chart.update_layout(title="Student performance distribution", showlegend=False)
    st.plotly_chart(outcome_chart, width="stretch")

if bundle:
    st.markdown("### Model performance summary")
    model_metrics = pd.DataFrame(
        [
            {"Model": name, "Accuracy": data["accuracy"], "Weighted F1": data["f1"]}
            for name, data in bundle["metrics"].items()
        ]
    )
    model_chart = px.bar(model_metrics, x="Model", y="Weighted F1", color="Model")
    model_chart.update_layout(title="Held-out weighted F1 by model", yaxis_tickformat=".0%")
    st.plotly_chart(model_chart, width="stretch")
    importance = global_feature_importance(bundle["models"][best_model], bundle["feature_columns"])
    if importance:
        st.markdown("### Feature importance overview")
        importance_frame = pd.DataFrame(importance[:10], columns=["Feature", "Importance"])
        importance_chart = px.bar(importance_frame.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
        st.plotly_chart(importance_chart, width="stretch")
        st.caption("Model importance describes reliance, not causal influence.")

categorical_candidates = [
    column
    for column in frame.select_dtypes(exclude="number").columns
    if not is_identifier_column(column)
    and column != target_column
    and 2 <= frame[column].nunique(dropna=True) <= 12
]
if score_column and categorical_candidates and pd.api.types.is_numeric_dtype(frame[score_column]):
    group_by = st.selectbox("Compare assessment by available group", categorical_candidates)
    group_data = frame[[group_by, score_column]].dropna()
    if not group_data.empty:
        group_chart = px.box(group_data, x=group_by, y=score_column, points="outliers")
        group_chart.update_layout(title=f"{score_column} by {group_by}")
        st.plotly_chart(group_chart, width="stretch")
        st.caption("Descriptive cohort comparison only; group characteristics are not used to recommend punitive actions.")

st.markdown("### Quick access")
quick = st.columns(4)
quick[0].page_link("pages/8_Early_Warning_System.py", label="Early warning", icon=":material/notification_important:")
quick[1].page_link("pages/9_Student_Insights.py", label="Student insights", icon=":material/person_search:")
quick[2].page_link("pages/4_Model_Comparison.py", label="Compare models", icon=":material/monitoring:")
quick[3].page_link("pages/6_Reports.py", label="Reports", icon=":material/download:")
