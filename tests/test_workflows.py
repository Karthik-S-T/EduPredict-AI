from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa

from src.ai_assistant import generate_assistant_response
from src.ai_context import build_assistant_context
from src.data_loader import make_demo_dataset, read_dataset, suggest_feature_columns
from src.engagement import score_engagement, suggest_activity_columns
from src.model_store import load_model_bundle, save_model_bundle
from src.oulad_processor import build_oulad_student_table
from src.prediction import logistic_logit_contributions, predict_student
from src.report_generator import csv_download, excel_workbook_download, pdf_report_download
from src.risk import assess_student_risk, class_risk_band
from src.train_model import train_models
from src.ui import prepare_dataframe_for_display


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = make_demo_dataset(120)
        cls.frame["Course"] = ["A" if index % 2 else "B" for index in range(len(cls.frame))]
        cls.target = "Final Result"
        cls.features = suggest_feature_columns(cls.frame, cls.target)
        cls.bundle = train_models(
            cls.frame,
            cls.target,
            cls.features,
            test_size=0.2,
            random_state=17,
            random_forest_estimators=50,
        )

    def test_training_evaluates_selected_models_on_a_held_out_split(self) -> None:
        self.assertEqual(len(self.bundle["models"]), 4)
        self.assertEqual(self.bundle["train_records"] + self.bundle["test_records"], len(self.frame))
        for result in self.bundle["metrics"].values():
            self.assertTrue(0 <= result["accuracy"] <= 1)
            self.assertTrue(0 <= result["f1"] <= 1)
            self.assertGreaterEqual(result["training_seconds"], 0)
            self.assertIsNotNone(result["roc_auc"])
        self.assertEqual(self.bundle["random_state"], 17)

    def test_selected_model_and_logistic_individual_explanation(self) -> None:
        bundle = train_models(
            self.frame,
            self.target,
            self.features,
            selected_models=["Logistic Regression"],
            random_forest_estimators=50,
        )
        self.assertEqual(list(bundle["models"]), ["Logistic Regression"])
        values = bundle["feature_defaults"]
        prediction, probabilities = predict_student(
            bundle["models"]["Logistic Regression"], self.features, values
        )
        self.assertIn(prediction, bundle["classes"])
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        terms = logistic_logit_contributions(
            bundle["models"]["Logistic Regression"], values, prediction
        )
        self.assertTrue(terms)

    def test_risk_mapping_requires_semantic_target_classes(self) -> None:
        risk = assess_student_risk(self.frame, self.bundle, "Random Forest")
        self.assertEqual(len(risk), len(self.frame))
        self.assertTrue(
            set(risk["Risk category"]).issubset(
                {"Low Risk", "Moderate Risk", "High Risk"}
            )
        )
        self.assertEqual(class_risk_band("High Risk"), "High Risk")
        self.assertEqual(class_risk_band("High Performance"), "Low Risk")
        self.assertEqual(class_risk_band("Not Pass"), "High Risk")

        unknown = self.frame.copy()
        unknown[self.target] = ["Group A" if index % 2 else "Group B" for index in range(len(unknown))]
        unknown_bundle = train_models(
            unknown,
            self.target,
            self.features,
            selected_models=["Logistic Regression"],
        )
        with self.assertRaisesRegex(ValueError, "cannot be inferred safely"):
            assess_student_risk(unknown, unknown_bundle)

    def test_activity_score_uses_multiple_available_indicators(self) -> None:
        selected = suggest_activity_columns(self.frame)
        scored = score_engagement(self.frame, selected)
        self.assertTrue(scored["Engagement Score"].between(0, 100).all())
        with self.assertRaisesRegex(ValueError, "at least two"):
            score_engagement(self.frame, selected[:1])

    def test_oulad_joins_remain_one_record_per_student_presentation(self) -> None:
        info = pd.DataFrame(
            {
                "id_student": [1, 2],
                "code_module": ["A", "A"],
                "code_presentation": ["2013J", "2013J"],
                "final_result": ["Pass", "Fail"],
            }
        )
        assessments = pd.DataFrame(
            {"id_assessment": [10], "code_module": ["A"], "code_presentation": ["2013J"]}
        )
        student_assessment = pd.DataFrame(
            {"id_student": [1, 1], "id_assessment": [10, 10], "score": [80, 90]}
        )
        student_vle = pd.DataFrame(
            {
                "id_student": [1, 1],
                "code_module": ["A", "A"],
                "code_presentation": ["2013J", "2013J"],
                "id_site": [20, 21],
                "sum_click": [3, 4],
                "date": [1, 2],
            }
        )
        courses = pd.DataFrame(
            {
                "code_module": ["A"],
                "code_presentation": ["2013J"],
                "module_presentation_length": [240],
            }
        )
        vle = pd.DataFrame(
            {
                "code_module": ["A", "A"],
                "code_presentation": ["2013J", "2013J"],
                "id_site": [20, 21],
                "activity_type": ["forum", "resource"],
            }
        )
        joined = build_oulad_student_table(
            info, student_assessment, assessments, student_vle, courses, vle
        )
        self.assertEqual(len(joined), len(info))
        self.assertEqual(joined.loc[0, "Total_Clicks"], 7)
        self.assertEqual(joined.loc[0, "Assessment_Average_Score"], 85)
        self.assertEqual(joined.loc[0, "module_presentation_length"], 240)
        self.assertEqual(joined.loc[0, "Clicks_forum"], 3)

    def test_csv_excel_and_report_exports(self) -> None:
        csv_upload = SimpleNamespace(
            name="sample.csv",
            getvalue=lambda: self.frame.head(4).to_csv(index=False).encode(),
        )
        self.assertEqual(len(read_dataset(csv_upload)), 4)
        excel = BytesIO()
        self.frame.head(4).to_excel(excel, index=False)
        excel_upload = SimpleNamespace(name="sample.xlsx", getvalue=excel.getvalue)
        self.assertEqual(len(read_dataset(excel_upload)), 4)

        report_frame = pd.DataFrame([{"Model": "Logistic Regression", "F1": 0.8}])
        csv_bytes = csv_download(report_frame)
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        workbook = excel_workbook_download({"Metrics": report_frame})
        self.assertTrue(workbook.startswith(b"PK"))
        pdf = pdf_report_download(
            "EduPredict AI Report",
            "Dataset & demo",
            {"Metrics": report_frame},
            "Predictions are decision-support estimates.",
        )
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_model_bundle_persists_pipeline_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bundle.joblib"
            with patch("src.model_store.MODEL_PATH", path):
                save_model_bundle(self.bundle)
                loaded = load_model_bundle()
            self.assertEqual(loaded["target"], self.target)
            self.assertEqual(list(loaded["models"]), list(self.bundle["models"]))
            self.assertEqual(loaded["random_state"], 17)

    def test_mixed_display_columns_are_arrow_safe_without_changing_numeric_features(self) -> None:
        profile = pd.DataFrame(
            {
                "Indicator": ["Assessment Score", "Final Result"],
                "Observed value": [65.5, "Borderline"],
                "Mixed column": [12, "not recorded"],
                "Numeric feature": [65.5, 72.0],
            }
        )
        display = prepare_dataframe_for_display(profile)
        display["Observed value"] = display["Observed value"].astype("string")

        self.assertEqual(str(display["Observed value"].dtype), "string")
        self.assertEqual(str(display["Mixed column"].dtype), "string")
        self.assertTrue(pd.api.types.is_numeric_dtype(display["Numeric feature"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(profile["Numeric feature"]))
        table = pa.Table.from_pandas(display, preserve_index=False)
        self.assertEqual(table.num_rows, 2)
        self.assertEqual(display["Observed value"].tolist(), ["65.5", "Borderline"])
        self.assertEqual(profile["Observed value"].tolist(), [65.5, "Borderline"])

    def test_ai_fallback_uses_measured_model_and_student_context(self) -> None:
        selected = self.frame.iloc[0]
        model_only_context = build_assistant_context(
            "Which model performed best?",
            self.frame,
            self.bundle,
            [],
            student_index=0,
        )
        self.assertNotIn("selected_student", model_only_context)
        context = build_assistant_context(
            "Why is this student at risk and which factors affect performance?",
            self.frame,
            self.bundle,
            [],
            student_index=0,
        )
        self.assertIn("models", context)
        self.assertIn("selected_student", context)
        serialized = json.dumps(context)
        self.assertNotIn("Student ID", serialized)
        self.assertNotIn(str(selected["Student ID"]), serialized)
        self.assertIn("risk_assessment", context["selected_student"])
        response, source = generate_assistant_response(
            "Why is this student at risk and which factors affect performance?",
            context,
            api_key="",
        )
        self.assertEqual(source, "Local fallback")
        self.assertIn(context["selected_student"]["risk_assessment"]["category"], response)
        self.assertIn("screening indicator", response)
        if self.frame.iloc[0]["Attendance Percentage"] > self.frame["Attendance Percentage"].quantile(0.25):
            self.assertNotIn(
                "Attendance Percentage is in the cohort's lower quartile",
                context["selected_student"]["risk_assessment"]["reasons"],
            )

    def test_ai_fallback_does_not_invent_unavailable_student_or_model_facts(self) -> None:
        context = build_assistant_context(
            "What is this student's prediction and which model performed best?",
            None,
            None,
            [],
        )
        response, source = generate_assistant_response(
            "What is this student's prediction and which model performed best?",
            context,
            api_key="",
        )
        self.assertEqual(source, "Local fallback")
        self.assertIn("not available", response)
        self.assertNotIn("Predicted outcome is", response)
        self.assertNotIn("ranks highest", response)

    def test_ai_engagement_and_dataset_answers_use_measured_summaries(self) -> None:
        question = "How can we improve student engagement in this cohort?"
        context = build_assistant_context(question, self.frame, self.bundle, [])
        measured = score_engagement(self.frame, suggest_activity_columns(self.frame))
        self.assertAlmostEqual(
            context["engagement_cohort"]["average_score"],
            measured["Engagement Score"].mean(),
            places=1,
        )
        answer, _ = generate_assistant_response(question, context, api_key="")
        self.assertIn("current cohort's mean engagement indicator", answer)
        self.assertIn("experimental cohort-relative", answer.lower())
        self.assertIn("Attendance Percentage", answer)

        dataset_question = "Which columns are numerical or categorical, and are there missing values?"
        dataset_context = build_assistant_context(dataset_question, self.frame, self.bundle, [])
        dataset_answer, _ = generate_assistant_response(dataset_question, dataset_context, api_key="")
        self.assertTrue(any(column["type"] == "numerical" for column in dataset_context["dataset"]["columns"]))
        self.assertIn("number stored as text", dataset_answer.lower())

    def test_ai_service_sends_compact_context_and_handles_rate_limits(self) -> None:
        api_response = {
            "choices": [{"message": {"content": "The current F1 is available in the local context."}}]
        }
        successful_response = MagicMock()
        successful_response.__enter__.return_value = successful_response
        successful_response.read.return_value = json.dumps(api_response).encode("utf-8")
        context = {"models": {"best_by_weighted_f1": "Tree"}, "record_count": 10}
        with patch("src.ai_assistant.urlopen", return_value=successful_response) as request:
            answer, source = generate_assistant_response("Which model performed best?", context, "secret-key")
        self.assertEqual(source, "Configured AI service")
        self.assertIn("current F1", answer)
        sent = json.loads(request.call_args.args[0].data.decode("utf-8"))
        user_content = sent["messages"][1]["content"]
        self.assertIn('"best_by_weighted_f1": "Tree"', user_content)
        self.assertNotIn("full dataset", user_content)
        self.assertNotIn("secret-key", user_content)

        from urllib.error import HTTPError

        with patch(
            "src.ai_assistant.urlopen",
            side_effect=HTTPError("https://example.test", 429, "rate", {}, None),
        ):
            fallback, fallback_source = generate_assistant_response("What is F1?", {}, "secret-key")
        self.assertIn("rate limit", fallback_source)
        self.assertIn("harmonic mean", fallback)


if __name__ == "__main__":
    unittest.main()
