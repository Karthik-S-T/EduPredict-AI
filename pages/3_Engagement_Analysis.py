import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import find_column, is_identifier_column
from src.engagement import score_engagement, suggest_activity_columns
from src.ui import initialize_session_state, page_header, prepare_dataframe_for_display


initialize_session_state()
page_header(
    "Engagement Analysis",
    "Compare a student's activity with peers using multiple observed learning indicators.",
)
frame = st.session_state.dataset
if frame is None:
    st.info("Load a dataset in Dataset Explorer to begin engagement analysis.")
    st.page_link("pages/5_Dataset_Explorer.py", label="Open Dataset Explorer", icon=":material/database:")
    st.stop()

available = [
    column
    for column in frame.select_dtypes(include="number").columns
    if not is_identifier_column(column)
]
suggested = suggest_activity_columns(frame)
selected = st.multiselect(
    "Numeric activity indicators",
    available,
    default=suggested,
    help="Engagement is the mean within-cohort percentile rank across the selected indicators. Select at least two; platform visits alone are not sufficient.",
)
if len(selected) < 2:
    st.info("Select at least two numeric activity indicators to calculate a cohort-relative engagement score.")
    st.stop()

try:
    result = score_engagement(frame, selected)
except ValueError as error:
    st.error(str(error))
    st.stop()

filter_columns = st.columns(3)
course_column = find_column(frame, ("code module", "course", "module"))
student_column = find_column(frame, ("student id", "id student", "student_id", "id_student"))
date_column = next(
    (
        column
        for column in frame.columns
        if any(term in str(column).lower() for term in ("timestamp", "date", "time"))
        and not pd.api.types.is_numeric_dtype(frame[column])
    ),
    None,
)
filtered = result.copy()
with filter_columns[0]:
    if course_column:
        course_values = sorted(frame[course_column].dropna().astype(str).unique())
        chosen_courses = st.multiselect("Course filter", course_values, default=course_values)
        filtered = filtered[filtered[course_column].astype(str).isin(chosen_courses)]
with filter_columns[1]:
    if student_column:
        student_values = filtered[student_column].dropna().astype(str).unique().tolist()
        chosen_students = st.multiselect(
            "Student filter (optional)",
            student_values[:1000],
            default=student_values[:0],
            help="Leave empty to include all records.",
        )
        if chosen_students:
            filtered = filtered[filtered[student_column].astype(str).isin(chosen_students)]
with filter_columns[2]:
    if date_column:
        parsed_dates = pd.to_datetime(filtered[date_column], errors="coerce")
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            first_date, last_date = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.date_input("Date range", value=(first_date, last_date))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                date_mask = parsed_dates.dt.date.between(date_range[0], date_range[1])
                filtered = filtered.loc[date_mask]

st.markdown(
    """
    <div class="note-card"><strong>How the score works</strong><br>
    Each selected indicator is converted to a percentile rank within this dataset. The student's score is the
    average of the available indicator percentiles; the cohort is then split into thirds: low, moderate, and high.
    This is a relative activity measure, not a validated measure of motivation or learning quality.</div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    counts = filtered["Engagement Level"].value_counts().reindex(
        ["Low Engagement", "Moderate Engagement", "High Engagement"], fill_value=0
    ).rename_axis("Engagement Level").reset_index(name="Students")
    chart = px.bar(
        counts,
        x="Engagement Level",
        y="Students",
        color="Engagement Level",
        color_discrete_map={"Low Engagement": "#e4777a", "Moderate Engagement": "#e3a442", "High Engagement": "#45a787"},
    )
    chart.update_layout(title="Engagement distribution", showlegend=False)
    st.plotly_chart(chart, width="stretch")

performance = find_column(frame, ("final result", "performance", "grade", "outcome", "assessment score", "average score"))
with right:
    if performance:
        chart = px.box(filtered, x="Engagement Level", y=performance, color="Engagement Level", points="outliers")
        chart.update_layout(title=f"Engagement versus {performance}", showlegend=False)
        st.plotly_chart(chart, width="stretch")
    else:
        st.info("No recognizable academic outcome column was found for comparison.")

st.markdown("### Activity indicator comparison")
st.caption("Each indicator is displayed as a cohort percentile so values with different units can be compared.")
activity_percentiles = filtered[selected].rank(pct=True, method="average") * 100
activity_long = activity_percentiles.melt(var_name="Indicator", value_name="Cohort percentile")
chart = px.box(activity_long, x="Indicator", y="Cohort percentile", color="Indicator", points=False)
chart.update_layout(showlegend=False, yaxis_title="Cohort percentile")
st.plotly_chart(chart, width="stretch")

st.markdown("### Engagement heatmap")
heatmap_frame = filtered
if student_column:
    heatmap_frame = heatmap_frame.sort_values("Engagement Score", ascending=False).head(50)
heatmap_values = heatmap_frame[selected].rank(pct=True, method="average") * 100
if not heatmap_values.empty:
    identifiers = (
        heatmap_frame[student_column].astype(str).tolist()
        if student_column
        else [f"Record {index + 1}" for index in range(len(heatmap_values))]
    )
    heatmap = px.imshow(
        heatmap_values.to_numpy(),
        x=selected,
        y=identifiers,
        labels={"x": "Activity indicator", "y": "Student", "color": "Cohort percentile"},
        color_continuous_scale="Purples",
        aspect="auto",
    )
    heatmap.update_layout(title="Student activity profile · cohort percentiles")
    st.plotly_chart(heatmap, width="stretch")
else:
    st.info("No records match the current engagement filters.")

if date_column:
    trend_data = filtered.assign(_date=pd.to_datetime(filtered[date_column], errors="coerce")).dropna(subset=["_date"])
    if not trend_data.empty:
        trend = trend_data.groupby(trend_data["_date"].dt.to_period("W").astype(str))["Engagement Score"].mean().reset_index()
        trend.columns = ["Week", "Average engagement score"]
        trend_chart = px.line(trend, x="Week", y="Average engagement score", markers=True)
        trend_chart.update_layout(title="Engagement trend over time")
        st.plotly_chart(trend_chart, width="stretch")

st.markdown("### Activity patterns")
pattern_columns = st.columns(2)
with pattern_columns[0]:
    st.write("**Highest cohort medians**")
    st.dataframe(
        prepare_dataframe_for_display(
            filtered[selected].median(numeric_only=True).sort_values(ascending=False).head(5).rename("Median").to_frame()
        ),
        width="stretch",
    )
with pattern_columns[1]:
    st.write("**Lowest cohort medians**")
    st.dataframe(
        prepare_dataframe_for_display(
            filtered[selected].median(numeric_only=True).sort_values().head(5).rename("Median").to_frame()
        ),
        width="stretch",
    )

categories = ["Low Engagement", "Moderate Engagement", "High Engagement"]
filter_category = st.multiselect("Filter engagement categories", categories, default=categories)
display = filtered[filtered["Engagement Level"].isin(filter_category)].copy()
columns = ([student_column] if student_column else []) + selected + ["Engagement Score", "Engagement Level"]
st.markdown("### Student-level engagement")
st.dataframe(
    prepare_dataframe_for_display(display[columns].sort_values("Engagement Score", ascending=False)),
    width="stretch",
    hide_index=True,
)
