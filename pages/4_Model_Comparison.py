import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src.prediction import global_feature_importance
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header("Model Comparison", "Compare held-out test results, ROC performance, confusion patterns, and model-wide feature importance.")
bundle = st.session_state.model_bundle
if not bundle:
    st.info("No trained model is loaded. Select a dataset and train models in the Model Training Center.")
    st.page_link("pages/10_Model_Training_Center.py", label="Open Model Training Center", icon=":material/model_training:")
    st.stop()

metrics = bundle["metrics"]
rows = [
    {
        "Model": name,
        "Accuracy": values["accuracy"],
        "Precision (weighted)": values["precision"],
        "Recall (weighted)": values["recall"],
        "F1-score (weighted)": values["f1"],
        "ROC-AUC": values.get("roc_auc"),
        "Training time (s)": values.get("training_seconds"),
    }
    for name, values in metrics.items()
]
comparison = pd.DataFrame(rows)
st.markdown(
    f"**Dataset:** `{bundle.get('dataset_name', 'Not recorded')}` · **Target:** `{bundle['target']}` · "
    f"**Test records:** {bundle['test_records']:,} · **Weighted averages** · "
    f"**Random state:** {bundle.get('random_state', 'legacy model')} · "
    f"**Split:** {'stratified' if bundle.get('stratified_split') else 'unstratified'}"
)
if bundle.get("trained_at"):
    st.caption(f"Last trained (UTC): {bundle['trained_at']}")

ranking_metric = st.selectbox(
    "Select best-model metric",
    ["F1-score (weighted)", "Accuracy", "Precision (weighted)", "Recall (weighted)"],
    help="Best model is ranked by this selected metric, measured on the held-out test split.",
)
best = comparison.loc[comparison[ranking_metric].idxmax()]
st.success(f"Best by {ranking_metric}: **{best['Model']}** ({best[ranking_metric]:.1%})")

st.markdown("### Held-out test results")
st.dataframe(
    prepare_dataframe_for_display(comparison).style.format(
        {
            "Accuracy": "{:.1%}",
            "Precision (weighted)": "{:.1%}",
            "Recall (weighted)": "{:.1%}",
            "F1-score (weighted)": "{:.1%}",
            "ROC-AUC": lambda value: f"{value:.3f}" if pd.notna(value) else "Not supported",
            "Training time (s)": lambda value: f"{value:.2f}" if pd.notna(value) else "—",
        },
        na_rep="—",
    ),
    width="stretch",
    hide_index=True,
)
st.caption(
    "Preprocessing is fitted on training records only. Accuracy is the correct fraction; precision, recall and F1 use "
    "weighted class averages. ROC-AUC is omitted when probabilities/classes are not suitable."
)

left, right = st.columns(2)
with left:
    chart = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        color="Model",
        text_auto=".1%",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    chart.update_layout(title="Accuracy comparison", showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(chart, width="stretch")
with right:
    metric_long = comparison.melt(
        id_vars="Model",
        value_vars=["Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"],
        var_name="Metric",
        value_name="Score",
    )
    chart = px.bar(metric_long, x="Model", y="Score", color="Metric", barmode="group")
    chart.update_layout(title="Precision, recall, and F1-score", yaxis_tickformat=".0%")
    st.plotly_chart(chart, width="stretch")

selected_model = st.selectbox("Inspect model", list(metrics))
selected_metrics = metrics[selected_model]
labels = [str(label) for label in bundle["classes"]]
matrix = np.asarray(selected_metrics["confusion_matrix"])
heatmap = go.Figure(
    data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="Blues",
        colorbar={"title": "Students"},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    )
)
for row_index, row in enumerate(matrix):
    for column_index, count in enumerate(row):
        heatmap.add_annotation(
            x=labels[column_index],
            y=labels[row_index],
            text=str(count),
            showarrow=False,
            font={"color": "white" if count > matrix.max() * 0.55 else "#17213b"},
        )
heatmap.update_layout(
    title=f"{selected_model} · Confusion matrix",
    xaxis_title="Predicted label",
    yaxis_title="Actual label",
)
st.plotly_chart(heatmap, width="stretch")

probabilities = selected_metrics.get("test_probabilities")
actual = selected_metrics.get("test_actual")
roc_classes = selected_metrics.get("roc_classes")
if probabilities and actual and roc_classes:
    probabilities = np.asarray(probabilities)
    roc_figure = go.Figure()
    curve_count = 0
    for class_index, label in enumerate(roc_classes):
        binary_actual = (np.asarray(actual) == label).astype(int)
        if binary_actual.min() == binary_actual.max():
            continue
        false_positive, true_positive, _ = roc_curve(binary_actual, probabilities[:, class_index])
        auc = selected_metrics.get("roc_auc")
        trace_name = f"{label} (AUC {auc:.3f})" if len(roc_classes) == 2 and auc is not None else str(label)
        roc_figure.add_trace(
            go.Scatter(x=false_positive, y=true_positive, mode="lines", name=trace_name)
        )
        curve_count += 1
    if curve_count:
        roc_figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash", "color": "#9aa4b6"})
        )
        roc_figure.update_layout(
            title=f"{selected_model} · One-vs-rest ROC",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        st.plotly_chart(roc_figure, width="stretch")
else:
    st.info("ROC curves are not available for this saved model.")

report = selected_metrics["classification_report"]
report_frame = pd.DataFrame(report).transpose()
if "accuracy" in report_frame.index:
    report_frame = report_frame.drop(index="accuracy")
st.markdown("### Classification report")
st.dataframe(
    prepare_dataframe_for_display(report_frame).style.format("{:.3f}", na_rep="—"),
    width="stretch",
)

importance = global_feature_importance(bundle["models"][selected_model], bundle["feature_columns"])
if importance:
    importance_frame = pd.DataFrame(importance, columns=["Feature", "Importance"]).head(15)
    st.markdown("### Global feature importance")
    st.caption("Importance indicates model reliance, not a causal relationship.")
    importance_chart = px.bar(
        importance_frame.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#5967e8"],
    )
    importance_chart.update_layout(showlegend=False)
    st.plotly_chart(importance_chart, width="stretch")
else:
    st.info(f"{selected_model} does not expose a supported coefficient or tree-importance explanation.")
