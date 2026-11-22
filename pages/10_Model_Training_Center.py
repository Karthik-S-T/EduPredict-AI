import pandas as pd
import streamlit as st

from src.data_loader import is_identifier_column, suggest_feature_columns
from src.model_store import MODEL_PATH, load_model_bundle, save_model_bundle
from src.train_model import MODEL_NAMES, train_models
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header(
    "Model Training Center",
    "Configure a reproducible training run, evaluate selected algorithms on a held-out split, and optionally save the pipeline.",
)
frame = st.session_state.dataset
if frame is None:
    st.info("Load a CSV/Excel dataset or the labeled demo in Dataset Explorer before training.")
    st.page_link("pages/5_Dataset_Explorer.py", label="Open Dataset Explorer", icon=":material/database:")
    st.stop()

eligible_targets = [column for column in frame.columns if not is_identifier_column(column)]
if not eligible_targets:
    st.error("This dataset has no non-identifier column that can serve as a classification target.")
    st.stop()
preferred = next(
    (
        column
        for column in eligible_targets
        if str(column).strip().lower().replace("_", " ").replace("-", " ")
        in {"final result", "outcome", "performance category", "final grade", "grade", "result"}
    ),
    eligible_targets[0],
)

st.markdown(f"**Dataset:** `{st.session_state.get('dataset_name', 'Uploaded dataset')}` · {len(frame):,} records")
with st.form("model_training_form"):
    target = st.selectbox("Prediction target", eligible_targets, index=eligible_targets.index(preferred))
    available_features: list[str] = [str(column) for column in frame.columns if column != target]
    defaults = suggest_feature_columns(frame, target)
    features: list[str] = st.multiselect(
        "Predictor features",
        available_features,
        default=defaults,
        help="Identifiers and obvious outcome aliases are excluded. Review other possible leakage fields below.",
    )
    algorithms = st.multiselect("Algorithms", MODEL_NAMES, default=list(MODEL_NAMES))
    test_size = st.slider("Held-out test fraction", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=2**31 - 1, value=42, step=1)
    with st.expander("Random Forest settings"):
        estimators = st.slider("Number of trees", 50, 1000, 180, 10)
        depth_choice = st.selectbox("Maximum tree depth", ["8", "12", "20", "Unlimited"], index=0)
    with st.expander("Other tree model settings"):
        decision_depth_choice = st.selectbox("Decision Tree maximum depth", ["8", "12", "20", "Unlimited"], index=0)
        boosting_estimators = st.slider("Gradient Boosting estimators", 50, 500, 100, 10)
    save_locally = st.checkbox("Save this trained bundle to models/model_bundle.joblib")
    saved_bundle_exists = MODEL_PATH.exists()
    confirm_overwrite = False
    if save_locally and saved_bundle_exists:
        confirm_overwrite = st.checkbox(
            "I confirm replacing the existing locally saved model bundle.",
            value=False,
        )
    start_training = st.form_submit_button("Train selected algorithms", type="primary", width="stretch")

leakage_terms = ("final", "outcome", "result", "grade", "withdraw", "pass fail", "target")
possible_leaks = [
    str(column)
    for column in features
    if any(term in str(column).lower().replace("_", " ") for term in leakage_terms)
]
if possible_leaks:
    st.warning(
        "Review possible target-derived predictors before training: "
        + ", ".join(possible_leaks)
        + ". Remove fields that would not be known when the prediction is made."
    )
if saved_bundle_exists:
    st.caption(f"A local saved bundle exists at `{MODEL_PATH.name}`. Saving a new bundle requires explicit confirmation.")

if start_training:
    if save_locally and saved_bundle_exists and not confirm_overwrite:
        st.error("The existing saved model was not overwritten. Confirm replacement in the form and submit again, or train for this session only.")
    else:
        try:
            with st.spinner("Training and evaluating selected algorithms…"):
                bundle = train_models(
                    frame=frame,
                    target=target,
                    feature_columns=features,
                    test_size=float(test_size),
                    selected_models=algorithms,
                    random_state=int(random_state),
                    random_forest_estimators=int(estimators),
                    random_forest_max_depth=None if depth_choice == "Unlimited" else int(depth_choice),
                    decision_tree_max_depth=(
                        None if decision_depth_choice == "Unlimited" else int(decision_depth_choice)
                    ),
                    gradient_boosting_estimators=int(boosting_estimators),
                    dataset_name=st.session_state.get("dataset_name", "Uploaded dataset"),
                )
                if st.session_state.get("dataset_metadata"):
                    bundle["dataset_metadata"] = st.session_state.dataset_metadata
            st.session_state.model_bundle = bundle
            if save_locally:
                save_model_bundle(bundle)
                st.success("Training complete. The selected model pipelines and metadata were saved locally.")
            else:
                st.success("Training complete. The new bundle is available for this session; the saved model was not changed.")
        except (ValueError, OSError, TypeError) as error:
            st.error(f"Model training failed: {error}")

bundle = st.session_state.model_bundle
if bundle:
    st.markdown("### Current model bundle")
    st.write(
        f"**Target:** {bundle['target']} · **Algorithms:** {', '.join(bundle['models'])} · "
        f"**Train records:** {bundle['train_records']:,} · **Test records:** {bundle['test_records']:,} · "
        f"**Random state:** {bundle.get('random_state', 'legacy')} · "
        f"**Split:** {'stratified' if bundle.get('stratified_split') else 'unstratified'}"
    )
    st.caption(f"Last trained (UTC): {bundle.get('trained_at', 'Not recorded')}")
    scores = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": data["accuracy"],
                "Precision (weighted)": data["precision"],
                "Recall (weighted)": data["recall"],
                "F1-score (weighted)": data["f1"],
                "ROC-AUC": data.get("roc_auc"),
                "Training time (s)": data.get("training_seconds"),
            }
            for name, data in bundle["metrics"].items()
        ]
    )
    st.dataframe(prepare_dataframe_for_display(scores), width="stretch", hide_index=True)
    st.caption("Evaluation metrics use the held-out test split. Rank the best model by weighted F1 on Model Comparison.")

if st.button("Reload locally saved model"):
    try:
        saved = load_model_bundle()
        if saved is None:
            st.warning("There is no saved model bundle yet.")
        else:
            st.session_state.model_bundle = saved
            st.success("Loaded the locally saved model bundle.")
            st.rerun()
    except RuntimeError as error:
        st.error(str(error))
