# EduPredict AI

EduPredict AI is a local Streamlit application for student performance analysis, cohort-relative engagement summaries, model comparison, supportive risk screening, and report downloads. It accepts CSV and Excel data. No real student dataset is bundled.

## Requirements and launch

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The additional PDF dependency is ReportLab. The current workspace environment has the required dependencies installed.

## Workflow

1. In **Dataset Explorer**, upload CSV/Excel data or choose the explicitly labeled synthetic demo.
2. Inspect dimensions, types, missing values, duplicates, identifiers, potential outcome leakage, descriptive statistics, and the target distribution. Download the cleaned CSV if useful.
3. In **Model Training Center**, select target, features, algorithms, test fraction, random seed, and available tree parameters. Avoid predictors derived from the target or unavailable when an intervention would be made.
4. Review the held-out results in **Model Comparison**. Pipelines fit imputers, numerical scaling, and one-hot encoding using training records only. Duplicate source records are removed; different records with matching selected values are retained.
5. Explore **Student Performance Prediction**, **Student Insights**, **Early Warning System**, and **Engagement Analysis**. Export reports in **Reports and Downloads**.
6. Ask the **AI Assistant** about the current dataset, selected student, measured model results, or general ML concepts.

## Optional AI Assistant service

The AI Assistant works without external credentials using its labeled local rule-based mode. To enable an OpenAI-compatible chat-completions service, provide an API key outside source control:

PowerShell environment variable (for the current terminal session):

```powershell
$env:EDUPREDICT_AI_API_KEY = "your-api-key"
```

Or add this to the local `.streamlit/secrets.toml` file (do not commit that file):

```toml
EDUPREDICT_AI_API_KEY = "your-api-key"
```

The optional service defaults to `https://api.openai.com/v1/chat/completions` with model `gpt-4o-mini`. Configure a compatible endpoint and model with `EDUPREDICT_AI_API_URL` and `EDUPREDICT_AI_MODEL`. `OPENAI_API_KEY` is also accepted as a key name. API errors, unavailable networks, and missing keys fall back to the local assistant and are labeled in chat. The request contains only the user's question and a compact relevant context summary; it does not include the full dataset, direct student identifiers, or conversation history.

Predictions and risk classifications are decision-support estimates, not definitive judgments. Teachers should review student-specific conclusions and interventions.

Models use the same train/test split for a run. Stratification is applied when class counts and split sizes allow it. Logistic Regression, Decision Tree, and Random Forest use class balancing; Gradient Boosting uses balanced sample weights. Weighted precision, recall, and F1, accuracy, training time, confusion matrices, and supported ROC-AUC are measured from test predictions. The best-model default is weighted F1 on the held-out test split.

The trained pipelines and metadata are stored at `models/model_bundle.joblib` only when explicitly selected in Model Training Center. Saving over an existing bundle requires a separate confirmation. Without saving, the new bundle is available only in the current Streamlit session. Uploaded datasets are kept in session state and are not written to disk by the app.

## OULAD

Download the Open University Learning Analytics Dataset from [the official dataset website](https://analyse.kmi.open.ac.uk/open_dataset). In Dataset Explorer, upload `studentInfo.csv` and any available supporting tables in the OULAD expander:

- `studentAssessment.csv` plus `assessments.csv`: scores are aggregated by student, module, and presentation.
- `studentVle.csv`: click totals and active days are aggregated at the same grain.
- `vle.csv`: optionally adds click totals by activity type.
- `courses.csv`: optionally adds module presentation length.

The output uses `id_student`, `code_module`, and `code_presentation` and avoids joins that multiply student records. The explorer lists files and columns supplied to the aggregation. Attendance or previous results are not fabricated.

## Engagement method

Select at least two numeric activity indicators. Each selected value is converted to a percentile rank in the cohort; the mean of observed ranks is the student's 0–100 engagement score. Cohort thirds define low, moderate, and high relative activity. This is an experimental indicator, not a validated measure of motivation, behavior, or learning. Filters and trends are shown only when suitable course, student, or date fields are present.

## Risk and explainability

Risk mapping is available only for recognizable outcome classes (for example Pass/Borderline/Fail or letter grades). If every model class is mapped and probabilities are supported, the screening score is `P(high-risk outcome) + 0.5 × P(moderate outcome)`: High at 60% or above, Moderate at 25% to under 60%, otherwise Low. When classes are not all mapped, the predicted class alone is mapped; unknown outcomes remain unclassified. Numeric features in the cohort's lower quartile may be listed as context. Recommendations are supportive, not punitive.

Logistic Regression predictions can show additive terms in the predicted class logit. Other supported classifiers show global model importance, not an individual causal explanation. Probabilities are estimates, not guarantees of correctness. Model predictions and risk flags are decision-support only and can reflect data bias.

## Pages

- **Dashboard:** dataset/model status, calculated KPIs, performance/engagement/risk distributions, model summary, and available group comparisons.
- **Student Performance Prediction:** select a dataset student or enter model features; inspect output probabilities and supported explanations.
- **Engagement Analysis:** filter available records and compare multiple activity indicators, cohort percentiles, and trends when dates exist.
- **Model Comparison:** held-out metrics, training time, supported ROC curves, Plotly confusion matrix, class report, and model-wide importance.
- **Dataset Explorer:** uploads, OULAD aggregation, data quality, identifier/leakage warnings, and cleaned dataset export.
- **Reports and Downloads:** combined Excel workbook, PDF, and CSVs for available dataset, performance, engagement, risk, model, confusion matrix, importance, and prediction results.
- **About Project:** methodology, limitations, architecture, and academic project details.
- **Early Warning System:** risk distribution, search/filter, reasons, supportive recommendations, and CSV/Excel export.
- **Student Insights:** individual available indicators, cohort comparison, optional prediction/risk, and CSV export.
- **Model Training Center:** training configuration, explicit persistence/overwrite confirmation, and saved-model reload.
- **AI Assistant:** session chat with student, engagement, dataset, and model context; uses a clearly labeled local fallback unless an optional compatible AI API is configured.

Synthetic demo outcomes and model scores are for application testing only, not research findings.
