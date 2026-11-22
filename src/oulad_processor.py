from __future__ import annotations

import pandas as pd


def oulad_processing_notes(tables: dict[str, pd.DataFrame]) -> list[str]:
    notes = []
    student_assessment = tables.get("studentassessment")
    assessments = tables.get("assessments")
    if student_assessment is not None and assessments is None:
        notes.append("studentAssessment was not aggregated because assessments metadata was not uploaded.")
    elif student_assessment is None and assessments is not None:
        notes.append("assessments metadata was not used because studentAssessment was not uploaded.")
    elif student_assessment is not None and assessments is not None:
        if not {"id_student", "id_assessment", "score"} <= set(student_assessment.columns):
            notes.append("studentAssessment is missing id_student, id_assessment, or score; assessment features were skipped.")
        if not {"id_assessment", "code_module", "code_presentation"} <= set(assessments.columns):
            notes.append("assessments is missing a join key; assessment features were skipped.")
    student_vle = tables.get("studentvle")
    if student_vle is not None and not {
        "id_student",
        "code_module",
        "code_presentation",
        "sum_click",
    } <= set(student_vle.columns):
        notes.append("studentVle is missing required identifiers or sum_click; VLE features were skipped.")
    courses = tables.get("courses")
    if courses is not None and not {
        "code_module",
        "code_presentation",
        "module_presentation_length",
    } <= set(courses.columns):
        notes.append("courses is missing module keys or module_presentation_length; course length was skipped.")
    return notes


def build_oulad_student_table(
    student_info: pd.DataFrame,
    student_assessment: pd.DataFrame | None = None,
    assessments: pd.DataFrame | None = None,
    student_vle: pd.DataFrame | None = None,
    courses: pd.DataFrame | None = None,
    vle: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate OULAD activity to one row per student/course presentation."""
    required = {"id_student", "code_module", "code_presentation"}
    missing = required - set(student_info.columns)
    if missing:
        raise ValueError(f"studentInfo is missing required columns: {', '.join(sorted(missing))}")
    keys = ["id_student", "code_module", "code_presentation"]
    students = student_info.drop_duplicates(keys).copy()

    if courses is not None:
        course_columns = keys[1:] + ["module_presentation_length"]
        if set(course_columns) <= set(courses.columns):
            course_data = courses[course_columns].drop_duplicates(keys[1:])
            students = students.merge(course_data, on=keys[1:], how="left", validate="many_to_one")

    if student_assessment is not None and assessments is not None:
        assessment_required = {"id_student", "id_assessment", "score"}
        metadata_required = {"id_assessment", "code_module", "code_presentation"}
        if assessment_required <= set(student_assessment.columns) and metadata_required <= set(assessments.columns):
            scores = student_assessment.merge(
                assessments[list(metadata_required)], on="id_assessment", how="inner", validate="many_to_one"
            )
            scores = scores.groupby(keys, as_index=False).agg(
                Assessment_Average_Score=("score", "mean"),
                Assessment_Submissions=("id_assessment", "count"),
            )
            students = students.merge(scores, on=keys, how="left", validate="one_to_one")

    if student_vle is not None:
        vle_required = required | {"sum_click"}
        if vle_required <= set(student_vle.columns):
            activity = student_vle.groupby(keys, as_index=False).agg(
                Total_Clicks=("sum_click", "sum"),
                Active_Days=("date", "nunique") if "date" in student_vle.columns else ("sum_click", "count"),
            )
            students = students.merge(activity, on=keys, how="left", validate="one_to_one")
            if vle is not None and {"id_site", "activity_type"} <= set(vle.columns):
                metadata_keys = keys[1:] + ["id_site", "activity_type"]
                if set(metadata_keys) <= set(vle.columns) and "id_site" in student_vle.columns:
                    enriched = student_vle.merge(
                        vle[metadata_keys].drop_duplicates(["code_module", "code_presentation", "id_site"]),
                        on=["id_site", "code_module", "code_presentation"],
                        how="left",
                        validate="many_to_one",
                    )
                    type_clicks = enriched.dropna(subset=["activity_type"]).pivot_table(
                        index=keys,
                        columns="activity_type",
                        values="sum_click",
                        aggfunc="sum",
                        fill_value=0,
                    )
                    type_clicks.columns = [f"Clicks_{column}" for column in type_clicks.columns]
                    students = students.merge(
                        type_clicks.reset_index(), on=keys, how="left", validate="one_to_one"
                    )
    return students
