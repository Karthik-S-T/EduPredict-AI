from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 20
ENGAGEMENT_TERMS = ("engagement", "attendance", "activity", "participation", "click", "visit", "submission")


def generate_assistant_response(
    question: str,
    context: dict[str, Any],
    api_key: str | None = None,
) -> tuple[str, str]:
    """Use an OpenAI-compatible chat API when configured, otherwise answer locally."""
    configured_key = (
        api_key
        if api_key is not None
        else os.getenv("EDUPREDICT_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    if configured_key is not None:
        configured_key = configured_key.strip()
    if not configured_key:
        return rule_based_response(question, context), "Local fallback"

    api_url = os.getenv("EDUPREDICT_AI_API_URL", DEFAULT_API_URL).strip()
    model = os.getenv("EDUPREDICT_AI_MODEL", DEFAULT_MODEL).strip()
    try:
        response = _request_ai_service(question, context, configured_key, api_url, model)
        return response, "Configured AI service"
    except AssistantServiceError as error:
        return rule_based_response(question, context), f"Local fallback · AI service unavailable ({error})"


def _request_ai_service(
    question: str,
    context: dict[str, Any],
    api_key: str,
    api_url: str,
    model: str,
) -> str:
    parsed_url = urlparse(api_url)
    local_http = parsed_url.scheme == "http" and parsed_url.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if (parsed_url.scheme != "https" and not local_http) or not parsed_url.hostname or not model:
        raise AssistantServiceError("invalid endpoint or model configuration")
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are EduPredict AI's educational analytics assistant. Answer clearly and supportively. "
                    "Use only facts in the supplied session context for data-specific claims. If a fact is absent, "
                    "say it is unavailable; never invent student records, metrics, predictions, or explanations. "
                    "Treat predictions and risk as uncertain decision-support indicators and advise educator review. "
                    "Do not infer causation from feature importance. The context is a small summary, not the full dataset."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question[:2000]}\n\n"
                    f"Relevant current-session context (JSON): {json.dumps(context, ensure_ascii=False)[:12000]}"
                ),
            },
        ],
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        reason = "rate limit" if error.code == 429 else f"HTTP {error.code}"
        raise AssistantServiceError(reason) from None
    except (URLError, TimeoutError, OSError):
        raise AssistantServiceError("network error or timeout") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise AssistantServiceError("invalid response from AI service") from None

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AssistantServiceError("invalid response from AI service") from None
    if not isinstance(content, str) or not content.strip():
        raise AssistantServiceError("empty response from AI service")
    return content.strip()


class AssistantServiceError(RuntimeError):
    """A safe, user-presentable AI service failure category."""


def rule_based_response(question: str, context: dict[str, Any]) -> str:
    """Answer common analytics questions from definitions and supplied session facts."""
    prompt = question.casefold()
    sections: list[str] = []
    dataset = context.get("dataset")
    models = context.get("models")
    student = context.get("selected_student")

    if _has(prompt, ("risk", "high-risk", "high risk", "moderate-risk", "moderate risk", "low-risk", "low risk")):
        sections.append(
            "EduPredict's screening maps recognizable outcome labels to low, moderate, or high risk. When every "
            "class label is safely mapped and probabilities are available, its score is P(high-risk outcome) + "
            "0.5 * P(moderate outcome): high at 60% or above, moderate at 25% to under 60%, and low below 25%. "
            "If labels cannot be safely mapped, a category may be unavailable or based only on the predicted class."
        )

    if _has(prompt, ("f1", "f-score")):
        sections.append(
            "**F1-score** is the harmonic mean of precision and recall. It balances false-positive and false-negative "
            "trade-offs; a high score does not guarantee equally good results for every subgroup."
        )
    elif _has(prompt, ("precision",)):
        sections.append("**Precision** is the fraction of a model's positive predictions that are correct.")
    elif _has(prompt, ("recall",)):
        sections.append("**Recall** is the fraction of actual positive cases that the model identifies.")
    elif _has(prompt, ("accuracy",)):
        sections.append("**Accuracy** is the fraction of all test-set predictions that are correct.")
    elif _has(prompt, ("confusion matrix", "confusion")):
        sections.append(
            "A **confusion matrix** counts actual classes by predicted classes. Its diagonal contains correct "
            "predictions; off-diagonal cells show which classes were confused."
        )
    elif _has(prompt, ("logistic regression", "decision tree", "random forest", "gradient boosting", "machine learning")):
        sections.append(
            "**Logistic Regression** estimates class probabilities from a linear combination of transformed features. "
            "**Decision Tree** applies a sequence of feature-based splits. **Random Forest** averages many trees to "
            "reduce variance. **Gradient Boosting** adds trees sequentially to improve prior errors. Their held-out "
            "results depend on the dataset and training setup."
        )

    if _has(prompt, ("which model", "best model", "model performed", "compare model")):
        if models and models.get("evaluation"):
            evaluation = models["evaluation"]
            best_models = models.get("weighted_f1_ties") or [models["best_by_weighted_f1"]]
            best_model_summary = " and ".join(best_models)
            if len(best_models) > 1:
                best_model_summary += " (tied)"
                best_model_verb = "rank"
            else:
                best_model_verb = "ranks"
            sections.append(
                f"On this session's held-out test split ({models.get('test_records', 'unknown')} records), "
                f"**{best_model_summary}** {best_model_verb} highest by weighted F1 "
                f"({models.get('best_weighted_f1', 'unavailable')}). "
                + " ".join(
                    f"{name}: F1 {scores.get('f1', 'unavailable'):.3f}, accuracy {scores.get('accuracy', 'unavailable'):.3f}."
                    for name, scores in evaluation.items()
                    if isinstance(scores.get("f1"), (int, float))
                    and isinstance(scores.get("accuracy"), (int, float))
                )
            )
        else:
            sections.append("Current model evaluation results are not available in this session.")

    if _has(prompt, ("feature", "factor", "affect", "influence", "important")):
        if models and models.get("feature_importance_available"):
            ranked = models.get("feature_importance", [])
            if ranked:
                sections.append(
                    "Available global feature importance for the best-by-weighted-F1 model: "
                    + ", ".join(f"{item['feature']} ({item['importance']:.3g})" for item in ranked[:5])
                    + ". This describes model reliance, not a causal effect or necessarily an individual explanation."
                )
            else:
                sections.append("Feature importance values are not available for this model.")
        elif student and student.get("individual_logit_terms"):
            sections.append(
                "For this Logistic Regression estimate, available additive class-logit terms (not causal effects) are: "
                + ", ".join(
                    f"{item['feature']} ({item['term']:+.3g})"
                    for item in student["individual_logit_terms"][:5]
                )
                + "."
            )
        elif _has(prompt, ("feature", "factor", "affect", "influence", "important")):
            sections.append("No supported feature-importance or individual explanation is available for this model.")

    if student:
        prediction = student.get("prediction")
        risk = student.get("risk_assessment")
        engagement = student.get("engagement")
        if _has(prompt, ("prediction", "predict", "outcome", "performance")):
            if prediction and prediction.get("predicted_outcome"):
                probabilities = prediction.get("class_probabilities", {})
                probability_text = (
                    " Class estimates: "
                    + ", ".join(f"{label} {value:.1%}" for label, value in probabilities.items())
                    + "."
                    if probabilities
                    else ""
                )
                sections.append(
                    f"The selected record's {prediction['target']} estimate is **{prediction['predicted_outcome']}** "
                    f"from {prediction['model']}.{probability_text} This is not a guaranteed outcome."
                )
            else:
                sections.append("A prediction for the selected record is not available.")
        if _has(prompt, ("risk", "support", "intervention", "improve", "high risk")):
            if risk and risk.get("category"):
                sections.append(
                    f"The selected record is classified **{risk['category']}** by the available screening context. "
                    f"Observed reasons: {risk['reasons']}. Suggested supportive steps: {risk['suggested_support']}. "
                    "This is a screening indicator, not a final judgment; a teacher should review the student's context."
                )
            elif risk and risk.get("available") is False:
                sections.append("A safe risk classification is not available for the selected record: " + risk["reason"])
            elif _has(prompt, ("risk", "support", "intervention", "improve", "high risk")):
                sections.append("Student-specific risk evidence is not available; no individual risk reason can be inferred.")
        if _has(prompt, ENGAGEMENT_TERMS):
            if engagement and engagement.get("score") is not None:
                indicators = ", ".join(
                    f"{name}: {value}" for name, value in engagement.get("indicators", {}).items()
                )
                sections.append(
                    f"The selected record's cohort-relative engagement score is {engagement['score']}/100 "
                    f"({engagement['cohort_relative_level']}). Available indicators: {indicators}. "
                    + engagement["note"]
                )
            else:
                sections.append("Engagement indicators for the selected record are not available.")
        if _has(prompt, ("student", "learner", "strength", "weakness", "improve")) and student.get(
            "observed_academic_indicators"
        ):
            observed = (
                "Observed academic indicators for the selected record: "
                + ", ".join(
                    f"{name}: {value}"
                    for name, value in student["observed_academic_indicators"].items()
                )
                + ". These values describe this record only; compare with educator knowledge and course expectations."
            )
            percentiles = student.get("cohort_percentiles", {})
            if percentiles:
                observed += (
                    " Approximate cohort percentiles for numeric indicators: "
                    + ", ".join(f"{name}: {_ordinal(value)} percentile" for name, value in percentiles.items())
                    + "; these are descriptive comparisons, not validated academic benchmarks."
                )
            sections.append(observed)
        elif _has(prompt, ("strength", "weakness", "academic indicator")):
            sections.append("No usable academic indicators are available for the selected record.")
    elif _has(prompt, ("student", "learner", "their", "this", "risk", "intervention", "support", "improve")):
        sections.append(
            "No student-specific evidence is available because no record was selected. Select a record above for an "
            "individual prediction, screening result, observed indicators, or cohort-relative engagement context."
        )

    if _has(prompt, ENGAGEMENT_TERMS):
        cohort_engagement = context.get("engagement_cohort")
        if cohort_engagement and cohort_engagement.get("average_score") is not None:
            distribution = ", ".join(
                f"{label}: {count}"
                for label, count in cohort_engagement["category_counts"].items()
            )
            sections.append(
                f"The current cohort's mean engagement indicator is {cohort_engagement['average_score']:.1f}/100. "
                f"Available measures: {', '.join(cohort_engagement['indicators'])}. Relative categories: {distribution}. "
                + cohort_engagement["note"]
                + " To support engagement, offer accessible learning resources, regular low-stakes check-ins, and "
                "timely feedback; review the learner's circumstances before choosing an intervention."
            )
            related = cohort_engagement.get("related_academic_indicators", {})
            if related:
                sections.append(
                    "Separate available attendance/assessment context (not included in the activity score): "
                    + "; ".join(
                        f"{name}: mean {details['mean']} ({details['missing']} missing)"
                        for name, details in related.items()
                    )
                    + "."
                )
        elif not student:
            sections.append(
                "Student engagement data is not available in the current session. In general, consider accessible "
                "learning resources, regular check-ins, timely feedback, and a teacher conversation before selecting support."
            )

    if _has(prompt, ("dataset", "column", "missing", "categorical", "numerical", "numeric", "data quality", "uploaded")):
        if dataset and dataset.get("available") is False:
            sections.append("No dataset is loaded in the current session.")
        elif dataset:
            sections.append(
                f"The current dataset is **{dataset.get('source', 'unnamed')}**, with "
                f"{dataset['record_count']} records and {dataset['column_count']} columns."
            )
            if dataset.get("columns"):
                details = [
                    f"{column['name']} ({column['type']}, {column['missing']} missing)"
                    for column in dataset["columns"][:15]
                ]
                sections.append("Available columns: " + "; ".join(details) + ".")
            if dataset.get("numeric_summary"):
                summaries = [
                    f"{name}: mean {stats['mean']}, median {stats['median']}, "
                    f"{stats['missing']} missing"
                    for name, stats in dataset["numeric_summary"].items()
                ]
                sections.append("Numeric summaries: " + "; ".join(summaries) + ".")
            if dataset.get("duplicate_rows") is not None:
                sections.append(f"Exact duplicate rows detected: {dataset['duplicate_rows']}.")
        else:
            sections.append("No current-session dataset details are available to answer that data-specific question.")

    if _has(prompt, ("numeric", "numerical", "categorical")):
        sections.append(
            "Numerical features represent measured quantities; categorical features represent labels or groups. "
            "A number stored as text remains categorical until its format and meaning are validated."
        )

    if _has(prompt, ("last prediction", "recent prediction", "my prediction")):
        recent = context.get("recent_prediction")
        if recent and recent.get("available") is False:
            sections.append("There is no prediction history in this session.")
        elif recent:
            sections.append("Most recent prediction from this session: " + ", ".join(
                f"{key.replace('_', ' ')}: {value}" for key, value in recent.items()
            ) + ".")

    if not sections:
        sections.append(
            "I can explain performance predictions, risk screening, engagement indicators, model metrics, feature "
            "importance, and the loaded dataset. For student-specific answers, select a record above. "
            "Current session information is used only when it is available; unavailable values are not inferred."
        )
    return "\n\n".join(sections)


def _has(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _ordinal(value: float) -> str:
    rounded = round(value)
    if 10 <= rounded % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix}"
