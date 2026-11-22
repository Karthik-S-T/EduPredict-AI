import pandas as pd
import streamlit as st

from src.data_loader import find_column
from src.engagement import score_engagement, suggest_activity_columns
from src.prediction import global_feature_importance
from src.report_generator import (
    csv_download,
    excel_workbook_download,
    pdf_report_download,
)
from src.risk import assess_student_risk
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header("Reports and Downloads", "Export measured dataset, engagement, risk, model, and prediction results.")
frame = st.session_state.dataset
bundle = st.session_state.model_bundle
history = pd.DataFrame(st.session_state.prediction_history)
tables: dict[str, pd.DataFrame] = {}
report_meta = {
    "Generated at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "Dataset": st.session_state.get("dataset_name", "No active dataset"),
    "Model": "No trained model",
}

if frame is not None:
    dataset_summary = frame.describe(include="all").transpose().reset_index(names="Column")
    tables["Dataset overview"] = dataset_summary
    target_column = bundle.get("target") if bundle else find_column(frame, ("final result", "performance", "grade", "outcome"))
    if target_column and target_column in frame:
        performance = (
            frame[target_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Students")
        )
        tables["Performance summary"] = performance
        st.markdown("### Performance summary")
        st.dataframe(prepare_dataframe_for_display(performance), width="stretch", hide_index=True)

    activity_columns = suggest_activity_columns(frame)
    if len(activity_columns) >= 2:
        engagement = score_engagement(frame, activity_columns)
        student_column = find_column(frame, ("student id", "id student", "student_id", "id_student"))
        export_columns = ([student_column] if student_column else []) + activity_columns + ["Engagement Score", "Engagement Level"]
        engagement_export = engagement[export_columns]
        tables["Engagement analysis"] = engagement_export
        tables["Engagement summary"] = (
            engagement["Engagement Level"].value_counts().rename_axis("Engagement level").reset_index(name="Students")
        )
        st.markdown("### Engagement analysis")
        st.caption(
            "Cohort-relative percentiles across: "
            + ", ".join(activity_columns)
            + ". This is experimental and not a validated measure."
        )
        st.download_button(
            "Engagement analysis CSV",
            csv_download(engagement_export),
            "engagement_analysis.csv",
            "text/csv",
        )
else:
    st.info("Load a dataset to generate dataset, engagement, performance, and risk reports.")

if bundle:
    report_meta["Model"] = ", ".join(bundle["models"])
    report_meta["Target"] = bundle["target"]
    report_meta["Training dataset"] = bundle.get("dataset_name", "Not recorded")
    report_meta["Last trained (UTC)"] = bundle.get("trained_at", "Not recorded")
    report_meta["Test records"] = bundle["test_records"]
    report_meta["Train/test split"] = (
        "Stratified" if bundle.get("stratified_split") else "Unstratified"
    )
    evaluation = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": data["accuracy"],
                "Precision (weighted)": data["precision"],
                "Recall (weighted)": data["recall"],
                "F1-score (weighted)": data["f1"],
                "ROC-AUC": data.get("roc_auc"),
                "Training time (seconds)": data.get("training_seconds"),
            }
            for name, data in bundle["metrics"].items()
        ]
    )
    tables["Model comparison"] = evaluation
    st.markdown("### Model evaluation")
    st.dataframe(prepare_dataframe_for_display(evaluation), width="stretch", hide_index=True)
    st.download_button("Model comparison CSV", csv_download(evaluation), "model_comparison.csv", "text/csv")

    best_model = max(bundle["metrics"], key=lambda name: bundle["metrics"][name]["f1"])
    best_metrics = bundle["metrics"][best_model]
    labels = [str(label) for label in bundle["classes"]]
    confusion = pd.DataFrame(
        best_metrics["confusion_matrix"],
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels],
    ).reset_index(names="Class")
    tables["Confusion matrix"] = confusion
    importance = global_feature_importance(bundle["models"][best_model], bundle["feature_columns"])
    if importance:
        tables["Feature importance"] = pd.DataFrame(importance, columns=["Feature", "Importance"])

    if frame is not None and all(column in frame.columns for column in bundle["feature_columns"]):
        try:
            risk = assess_student_risk(frame, bundle, best_model)
            risk_report = risk.drop(columns="_row_index")
            tables["Risk assessment"] = risk_report
            tables["Risk summary"] = (
                risk["Risk category"].value_counts().rename_axis("Risk category").reset_index(name="Students")
            )
            st.markdown("### Risk summary")
            st.dataframe(prepare_dataframe_for_display(tables["Risk summary"]), width="stretch", hide_index=True)
        except ValueError:
            st.info("Risk summary unavailable: target labels are not safely mappable or required features are absent.")

    st.caption(
        f"Best model by held-out weighted F1: {best_model} ({best_metrics['f1']:.1%}). "
        "Predictions and model scores are estimates based on the current training run."
    )

if not history.empty:
    tables["Prediction results"] = history
    st.markdown("### Prediction results")
    st.dataframe(prepare_dataframe_for_display(history), width="stretch", hide_index=True)
    st.download_button("Prediction results CSV", csv_download(history), "prediction_results.csv", "text/csv")
else:
    st.info("Prediction results will appear here after predictions are generated.")

if tables:
    st.markdown("### Complete report")
    summary_frame = pd.DataFrame([{"Item": key, "Value": value} for key, value in report_meta.items()])
    workbook_tables = {"Report details": summary_frame, **tables}
    left, middle, right = st.columns(3)
    left.download_button(
        "Download report workbook (Excel)",
        excel_workbook_download(workbook_tables),
        "edupredict_analytics_report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    right.download_button(
        "Download professional report (PDF)",
        pdf_report_download(
            "EduPredict AI · Analytics Report",
            str(report_meta["Dataset"]),
            {name: data.head(40) for name, data in workbook_tables.items()},
            "Disclaimer: Predictions are decision-support estimates, not guaranteed outcomes. Engagement scores are "
            "experimental cohort-relative indicators; model importance does not establish causality. Synthetic demo "
            "results are not research findings.",
        ),
        "edupredict_analytics_report.pdf",
        "application/pdf",
    )
    st.caption("PDF tables are limited to the first 40 rows per section for readability; the Excel workbook contains full tables.")
