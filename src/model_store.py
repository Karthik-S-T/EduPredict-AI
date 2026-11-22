from __future__ import annotations

from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_bundle.joblib"


def save_model_bundle(bundle: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)


def load_model_bundle() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, EOFError, ValueError, AttributeError) as error:
        raise RuntimeError(f"Could not load the saved model bundle: {error}") from error
