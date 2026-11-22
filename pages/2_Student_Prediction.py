import pandas as pd
import plotly.express as px
import streamlit as st

from src.prediction import (
    global_feature_importance,
    logistic_logit_contributions,
    predict_student,
)
from src.risk import class_risk_band, student_identifier_column
from src.report_generator import csv_download
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header(
    "Student Performance Prediction",
    "Estimate the selected target from a trained pipeline. Predictions are decision-support estimates, never guaranteed outcomes.",
)
bundle = st.session_state.model_bundle
if not bundle:
    st.info("No trained model is available. Load data in Dataset Explorer and train a model in Model Training Center.")
    st.page_link("pages/10_Model_Training_Center.py", label="Open Model Training Center", icon=":material/model_training:")
    st.stop()

frame = st.session_state.dataset
model_name = st.selectbox("Trained model", list(bundle["models"]))
model = bundle["models"][model_name]
features = bundle["feature_columns"]
defaults = bundle.get("feature_defaults", {})
numeric_features = set(bundle.get("numeric_features", []))
identifier = student_identifier_column(frame) if frame is not None else None

prediction_mode = "Manual entry"
selected_row = None
if frame is not None and identifier:
    prediction_mode = st.radio(
        "Prediction input",
        ["Select existing student", "Enter manually"],
        horizontal=True,
    )
if prediction_mode == "Select existing student" and frame is not None and identifier:
    choices = list(range(len(frame)))
    selected_index = st.selectbox(
        "Student",
        choices,
        format_func=lambda index: f"{frame.iloc[index][identifier]} · row {index + 1}",
    )
    selected_row = frame.iloc[selected_index]

input_values = {}
student_reference = "Manual entry"
if selected_row is not None:
    student_reference = str(selected_row[identifier])

with st.form("prediction_form"):
    st.markdown("#### Student indicators")
    st.caption("The input fields match the chosen model's exact feature list.")
    if prediction_mode == "Manual entry":
        student_reference = st.text_input("Optional student reference", value="")
    columns = st.columns(2)
    for index, feature in enumerate(features):
        default = defaults.get(feature, 0)
        if selected_row is not None and feature in selected_row and pd.notna(selected_row[feature]):
            default = selected_row[feature]
        with columns[index % 2]:
            if feature in numeric_features:
                try:
                    value = float(default)
                except (TypeError, ValueError):
                    value = float(defaults.get(feature, 0))
                observed = frame[feature].dropna() if frame is not None and feature in frame else pd.Series(dtype=float)
                bounded_percentage = (
                    any(word in feature.lower() for word in ("score", "attendance", "percent", "submission", "participation"))
                    and not observed.empty
                    and pd.to_numeric(observed, errors="coerce").min() >= 0
                    and pd.to_numeric(observed, errors="coerce").max() <= 100
                )
                input_values[feature] = st.number_input(
                    feature,
                    value=value,
                    min_value=0.0 if bounded_percentage else None,
                    max_value=100.0 if bounded_percentage else None,
                    help="Enter a value using the same units as the training data.",
                )
            else:
                options = []
                if frame is not None and feature in frame:
                    options = frame[feature].dropna().astype(str).unique().tolist()[:100]
                if options:
                    default_text = str(default)
                    selected = default_text if default_text in options else options[0]
                    input_values[feature] = st.selectbox(
                        feature,
                        options,
                        index=options.index(selected),
                    )
                else:
                    input_values[feature] = st.text_input(feature, value=str(default))
    submitted = st.form_submit_button("Predict Student Performance", type="primary", width="stretch")

if submitted:
    try:
        prediction, probabilities = predict_student(model, features, input_values)
        risk = class_risk_band(prediction)
        confidence = max(probabilities.values()) if probabilities else None
        st.session_state.prediction_history.append(
            {
                "Student": student_reference or "Manual entry",
                "Predicted result": prediction,
                "Target": bundle["target"],
                "Model": model_name,
                "Highest class probability": confidence,
                "Risk interpretation": risk or "Not mapped",
                "Dataset": bundle.get("dataset_name", "Not recorded"),
            }
        )
        st.markdown("### Prediction result")
        result_columns = st.columns(3)
        result_columns[0].metric(f"Predicted {bundle['target']}", prediction)
        result_columns[1].metric("Mapped academic category", risk or "Not inferred")
        result_columns[2].metric("Highest class probability", f"{confidence:.1%}" if confidence is not None else "Unavailable")
        st.caption(
            "Model probabilities express estimated class likelihood under this fitted model; they are not guaranteed "
            "probabilities of correctness and must not be interpreted as certainty."
        )
        if probabilities:
            probability_frame = pd.DataFrame(
                [{"Class": category, "Estimated probability": probability} for category, probability in probabilities.items()]
            ).sort_values("Estimated probability", ascending=False)
            probability_chart = px.bar(
                probability_frame,
                x="Class",
                y="Estimated probability",
                color="Class",
                text_auto=".1%",
            )
            probability_chart.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(probability_chart, width="stretch")

        st.markdown("#### Prediction explanation")
        contributions = logistic_logit_contributions(model, input_values, prediction)
        if contributions:
            st.caption(
                "For Logistic Regression these are additive feature terms in the predicted class logit, after the "
                "trained preprocessing transformation. They are model terms, not causal effects."
            )
            contribution_frame = pd.DataFrame(contributions[:12], columns=["Transformed feature", "Logit contribution"])
            contribution_frame["Direction"] = contribution_frame["Logit contribution"].map(
                lambda value: "Raises class logit" if value >= 0 else "Lowers class logit"
            )
            st.dataframe(prepare_dataframe_for_display(contribution_frame), width="stretch", hide_index=True)
        else:
            importance = global_feature_importance(model, features)
            if importance:
                st.caption(
                    "This model exposes global feature importance but not a supported individual contribution method. "
                    "Global importance describes model reliance, not this student's causal explanation."
                )
                st.dataframe(
                    prepare_dataframe_for_display(
                        pd.DataFrame(importance, columns=["Feature", "Global importance"]).head(10)
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(f"{model_name} does not expose a supported explanation method.")

        if risk == "High Risk":
            st.warning("Consider supportive educator review and an individualized check-in; the prediction is not a guaranteed outcome.")
        elif risk == "Moderate Risk":
            st.info("Consider progress monitoring and checking whether targeted academic support would help.")
        elif risk == "Low Risk":
            st.success("Continue regular progress reviews and reinforce effective learning practices.")
        else:
            st.info("No risk category is inferred for this target label.")

        report_row = {
            "Student": student_reference or "Manual entry",
            "Dataset": st.session_state.get("dataset_name", "No active dataset"),
            "Target": bundle["target"],
            "Predicted outcome": prediction,
            "Model": model_name,
            "Highest class probability": confidence,
            "Mapped risk": risk or "Not inferred",
            **{f"Input: {key}": value for key, value in input_values.items()},
            **{f"Probability: {key}": value for key, value in probabilities.items()},
        }
        st.download_button(
            "Download prediction report CSV",
            csv_download(pd.DataFrame([report_row])),
            "student_prediction_report.csv",
            "text/csv",
        )
    except (ValueError, TypeError) as error:
        st.error(f"Prediction could not be generated: {error}")
