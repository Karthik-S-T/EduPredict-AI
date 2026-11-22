import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import find_column
from src.prediction import predict_student
from src.risk import assess_student_risk, class_risk_band, student_identifier_column
from src.report_generator import csv_download
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header(
    "Student Insights",
    "Review one learner's available academic indicators alongside the current cohort, with privacy-conscious details.",
)
frame = st.session_state.dataset
if frame is None or frame.empty:
    st.info("Load student records in Dataset Explorer to view individual insights.")
    st.page_link("pages/5_Dataset_Explorer.py", label="Open Dataset Explorer", icon=":material/database:")
    st.stop()

identifier = student_identifier_column(frame)
display_name = find_column(frame, ("student name", "full name", "name"))
selector_column = identifier or display_name
if selector_column:
    selections = [
        f"{frame.iloc[position][selector_column]} · row {position + 1}"
        for position in range(len(frame))
    ]
    selected_index = st.selectbox("Select a student", range(len(frame)), format_func=lambda index: selections[index])
else:
    st.info("No student identifier or name field was detected. Select a dataset row; this row number is for this dataset session only.")
    selected_index = st.selectbox("Dataset row", range(len(frame)), format_func=lambda index: f"Record {index + 1}")

student = frame.iloc[selected_index]
bundle = st.session_state.model_bundle
identifier_value = str(student[selector_column]) if selector_column else f"Record {selected_index + 1}"
st.markdown(f"### Learner · {identifier_value}")

feature_columns = bundle["feature_columns"] if bundle else []
safe_columns = [column for column in feature_columns if column in frame.columns]
if bundle and bundle["target"] in frame.columns:
    safe_columns.append(bundle["target"])
if not safe_columns:
    safe_columns = frame.select_dtypes(include="number").columns.tolist()
if not safe_columns:
    safe_columns = [column for column in frame.columns if column != selector_column][:12]

st.markdown("#### Available academic and model indicators")
profile = pd.DataFrame(
    [{"Indicator": column, "Observed value": student[column]} for column in safe_columns]
)
profile["Observed value"] = profile["Observed value"].astype("string")
st.dataframe(prepare_dataframe_for_display(profile), width="stretch", hide_index=True)

numeric_columns = [
    column
    for column in frame.select_dtypes(include="number").columns
    if column in safe_columns
]
if numeric_columns:
    comparison = []
    for column in numeric_columns:
        cohort = pd.to_numeric(frame[column], errors="coerce").dropna()
        value = pd.to_numeric(pd.Series([student[column]]), errors="coerce").iloc[0]
        if cohort.empty or pd.isna(value):
            continue
        comparison.append(
            {
                "Indicator": column,
                "Selected student": value,
                "Cohort median": cohort.median(),
                "Cohort percentile": float((cohort <= value).mean() * 100),
            }
        )
    if comparison:
        comparison_frame = pd.DataFrame(comparison)
        st.markdown("#### Student versus cohort")
        chart = px.bar(
            comparison_frame,
            x="Indicator",
            y="Cohort percentile",
            color_discrete_sequence=["#5967e8"],
            range_y=[0, 100],
        )
        chart.update_layout(yaxis_title="Approximate cohort percentile")
        st.plotly_chart(chart, width="stretch")
        st.dataframe(prepare_dataframe_for_display(comparison_frame), width="stretch", hide_index=True)

prediction_result = None
if bundle and all(column in frame.columns for column in bundle["feature_columns"]):
    model_name = max(bundle["metrics"], key=lambda name: bundle["metrics"][name]["f1"])
    try:
        values = student[bundle["feature_columns"]].to_dict()
        prediction, probabilities = predict_student(bundle["models"][model_name], bundle["feature_columns"], values)
        st.markdown("#### Model estimate")
        st.metric(f"Predicted {bundle['target']}", prediction)
        risk_band = class_risk_band(prediction)
        if risk_band:
            st.metric("Mapped risk category", risk_band)
        else:
            st.info("This target label has no safe built-in risk interpretation.")
        if probabilities:
            confidence = max(probabilities.values())
            st.caption(
                f"Highest class probability: {confidence:.1%}. This is the model's estimated class probability, "
                "not a guarantee of correctness."
            )
            st.bar_chart(
                pd.Series(probabilities, name="Model probability"),
                color="#7868d8",
            )
        prediction_result = {
            "Student": identifier_value,
            "Target": bundle["target"],
            "Predicted outcome": prediction,
            "Model": model_name,
            "Highest class probability": max(probabilities.values()) if probabilities else None,
            **{f"Probability: {label}": probability for label, probability in probabilities.items()},
        }
    except (ValueError, TypeError) as error:
        st.warning(f"Could not generate an estimate for this row: {error}")
else:
    st.info("A compatible trained model and all required predictor columns are needed for a model estimate.")

if bundle:
    try:
        risk_frame = assess_student_risk(frame, bundle)
        risk_row = risk_frame.loc[risk_frame["_row_index"] == student.name]
        if not risk_row.empty:
            risk = risk_row.iloc[0]
            st.markdown("#### Support considerations")
            st.write(f"**Risk category:** {risk['Risk category']}")
            st.write(f"**Observed reasons:** {risk['Risk reasons']}")
            st.write(f"**Suggested support:** {risk['Suggested support']}")
    except ValueError:
        st.caption("A risk category is not shown because the trained target labels do not support a safe mapping.")

student_report = pd.concat([profile, pd.DataFrame([prediction_result])], ignore_index=True) if prediction_result else profile
st.download_button(
    "Download individual student insight CSV",
    csv_download(student_report),
    "student_insights.csv",
    "text/csv",
)
