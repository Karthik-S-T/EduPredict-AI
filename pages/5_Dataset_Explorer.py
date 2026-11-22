from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import is_identifier_column, make_demo_dataset, read_dataset
from src.oulad_processor import build_oulad_student_table, oulad_processing_notes
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header("Dataset Explorer", "Upload, inspect, prepare, and train classifiers on your own academic data.")
uploader_column, demo_column = st.columns([3, 1])
with uploader_column:
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], key="single_dataset")
with demo_column:
    st.markdown("<div style='height: 1.85rem'></div>", unsafe_allow_html=True)
    load_demo = st.button("Load synthetic demo", width="stretch", help="Creates clearly labeled, synthetic example records.")

if uploaded is not None:
    try:
        st.session_state.dataset = read_dataset(uploaded)
        st.session_state.dataset_name = uploaded.name
        st.session_state.dataset_metadata = {"source_type": "Uploaded file", "files_used": [uploaded.name]}
        st.success(f"Loaded {uploaded.name}. The data is available across all pages for this session.")
    except (ValueError, pd.errors.ParserError, OSError) as error:
        st.error(f"Could not load the uploaded dataset: {error}")
if load_demo:
    st.session_state.dataset = make_demo_dataset()
    st.session_state.dataset_name = "Synthetic demonstration data"
    st.session_state.dataset_metadata = {"source_type": "Synthetic demonstration data"}
    st.warning("Synthetic demonstration data is active. It is not real student data or OULAD.")

st.markdown("### OULAD table join (optional)")
with st.expander("Upload and aggregate OULAD tables", expanded=False):
    st.write(
        "Upload **studentInfo** and any available supporting tables in one batch. The join produces one row per "
        "student/course presentation, aggregates assessment scores and VLE clicks, and optionally adds course length "
        "and click totals by VLE activity type."
    )
    oulad_files = st.file_uploader(
        "OULAD CSV/Excel tables",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="oulad_tables",
    )
    if st.button("Build OULAD student table", disabled=not oulad_files):
        try:
            tables = {}
            table_filenames = {}
            for file in oulad_files:
                table_name = Path(file.name).stem.lower().replace("_", "").replace(" ", "")
                if table_name in {"studentinfo", "studentassessment", "assessments", "studentvle", "courses", "vle"}:
                    tables[table_name] = read_dataset(file)
                    table_filenames[table_name] = file.name
            if "studentinfo" not in tables:
                raise ValueError("Include the studentInfo.csv table; supporting tables are optional.")
            st.session_state.dataset = build_oulad_student_table(
                tables["studentinfo"],
                tables.get("studentassessment"),
                tables.get("assessments"),
                tables.get("studentvle"),
                tables.get("courses"),
                tables.get("vle"),
            )
            st.session_state.dataset_name = "Aggregated OULAD tables"
            st.session_state.dataset_metadata = {
                "source_type": "OULAD",
                "files_used": sorted(table_filenames.values()),
                "join_keys": ["id_student", "code_module", "code_presentation"],
                "columns_used": {name: list(table.columns) for name, table in tables.items()},
                "processing_notes": oulad_processing_notes(tables),
            }
            st.success("OULAD tables were aggregated without multiplying student records.")
            for note in st.session_state.dataset_metadata["processing_notes"]:
                st.warning(note)
        except (ValueError, pd.errors.ParserError, OSError) as error:
            st.error(f"Could not build the OULAD table: {error}")

frame = st.session_state.dataset
if frame is None:
    st.info("Upload a dataset or load the optional synthetic demo to begin. No dataset is bundled or treated as real data.")
    st.markdown("**OULAD:** Download the Open University Learning Analytics Dataset from [the official dataset site](https://analyse.kmi.open.ac.uk/open_dataset). The OULAD join accepts `studentInfo.csv` and optionally `studentAssessment.csv`, `assessments.csv`, `studentVle.csv`, `courses.csv`, and `vle.csv`.")
    st.stop()

st.markdown(f"### Preview · {st.session_state.get('dataset_name', 'Uploaded dataset')}")
st.caption(f"{len(frame):,} rows · {len(frame.columns):,} columns")
st.markdown(
    f"**Dataset status:** Ready · **Source:** {st.session_state.get('dataset_name', 'Uploaded dataset')} · "
    f"**Dimensions:** {len(frame):,} × {len(frame.columns):,}"
)
metadata = st.session_state.get("dataset_metadata", {})
if metadata.get("files_used"):
    st.caption(
        f"OULAD tables used: {', '.join(metadata['files_used'])} · "
        f"Join keys: {', '.join(metadata['join_keys'])}"
    )
    with st.expander("OULAD input files and columns"):
        for table_name, columns in metadata.get("columns_used", {}).items():
            st.write(f"**{table_name}:** {', '.join(columns)}")
        if metadata.get("processing_notes"):
            for note in metadata["processing_notes"]:
                st.warning(note)
st.dataframe(prepare_dataframe_for_display(frame.head(20)), width="stretch", hide_index=True)

identifier_columns = [column for column in frame if is_identifier_column(column)]
if identifier_columns:
    st.info(f"Potential identifiers (excluded from model feature suggestions): {', '.join(identifier_columns)}")
leakage_terms = ("final", "outcome", "result", "grade", "withdraw", "pass fail", "target")
leakage_candidates = [
    str(column)
    for column in frame.columns
    if any(term in str(column).lower().replace("_", " ") for term in leakage_terms)
]
if leakage_candidates:
    st.warning(
        "Review possible target/outcome-derived columns before training: "
        + ", ".join(leakage_candidates)
        + ". Select only information available at the time of prediction."
    )

with st.expander("Dataset quality and statistics"):
    first, second = st.columns(2)
    with first:
        st.markdown("#### Column types")
        st.dataframe(
            prepare_dataframe_for_display(frame.dtypes.astype(str).rename("Data type").to_frame()),
            width="stretch",
        )
        st.markdown("#### Missing values")
        missing = frame.isna().sum().rename("Missing values").to_frame()
        missing["Missing %"] = (frame.isna().mean() * 100).round(1)
        st.dataframe(prepare_dataframe_for_display(missing), width="stretch")
    with second:
        st.metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
        st.markdown("#### Descriptive statistics")
        st.dataframe(prepare_dataframe_for_display(frame.describe(include="all").transpose()), width="stretch")
    st.download_button(
        "Download dataset summary",
        data=frame.describe(include="all").transpose().to_csv().encode("utf-8-sig"),
        file_name="dataset_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cleaned dataset (exact duplicates removed)",
        data=frame.drop_duplicates().to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

target_options = [column for column in frame.columns if not is_identifier_column(column)]
if target_options:
    selected_target = st.selectbox("Inspect target distribution", target_options, key="explorer_target")
    distribution = frame[selected_target].fillna("Missing").astype(str).value_counts()
    st.bar_chart(distribution, color="#5967e8")
else:
    st.info("No non-identifier columns are available for target inspection.")

st.markdown("### Model training")
st.caption("Select a target, predictors, algorithms, and training settings in the dedicated training workspace.")
st.page_link("pages/10_Model_Training_Center.py", label="Open Model Training Center", icon=":material/model_training:")
