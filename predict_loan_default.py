#!/usr/bin/env python3
"""Loan default prediction - inference only.

Loads the model saved by loan_default_pipeline.py (joblib) and scores new loan
applicants. No training happens here.

Usage:
    python predict_loan_default.py                      (scores new_applicants.csv)
    python predict_loan_default.py new_applicants.csv
    python predict_loan_default.py new_applicants.csv --threshold 0.23 --output scored.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import pandas as pd
import sklearn

LOGGER = logging.getLogger("loan_default_predict")

DEFAULT_MODEL_PATH = Path("outputs/loan_default_model.joblib")
DEFAULT_METADATA_PATH = Path("outputs/model_metadata.json")
DEFAULT_INPUT_PATH = Path("new_applicants.csv")


def load_metadata(path: Path) -> Dict[str, Any]:
    """Read the model card written during training."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Run loan_default_pipeline.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path):
    """Load the saved pipeline (scaler + encoder + network) from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run loan_default_pipeline.py first."
        )
    return joblib.load(path)


def warn_if_version_differs(metadata: Dict[str, Any]) -> None:
    """A model should be loaded with the scikit-learn version that built it."""
    trained_with = metadata.get("scikit_learn_version")
    if trained_with and trained_with != sklearn.__version__:
        LOGGER.warning(
            "Model was trained with scikit-learn %s but %s is installed. "
            "Predictions may be wrong or loading may fail.",
            trained_with,
            sklearn.__version__,
        )


def read_applicants(path: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Read applicants from CSV and check they match what the model expects."""
    if not path.exists():
        raise FileNotFoundError(f"Applicant file not found: {path}")

    data = pd.read_csv(path)
    features = metadata["features"]

    missing_columns = [c for c in features if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    applicants = data[features]  # also puts the columns in the order the model expects

    empty_rows = applicants.index[applicants.isnull().any(axis=1)].tolist()
    if empty_rows:
        raise ValueError(f"Rows with missing values (0-based row numbers): {empty_rows}")

    for column, allowed in metadata["categories"].items():
        unknown = sorted(set(applicants[column]) - set(allowed))
        if unknown:
            raise ValueError(f"Column '{column}' has unknown values {unknown}. Allowed: {allowed}")

    return applicants


def score_applicants(model, applicants: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add default probability and a Low/High risk label to each applicant."""
    probability = model.predict_proba(applicants)[:, 1]
    result = applicants.copy()
    result["DefaultProbability"] = probability.round(3)
    result["Prediction"] = (probability >= threshold).astype(int)
    result["RiskLabel"] = result["Prediction"].map({0: "Low risk", 1: "High risk"})
    return result


def threshold_value(value: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_loan_default.py",
        description="Score new loan applicants with the saved model (no retraining).",
    )
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT_PATH,
                        help=f"CSV file with the applicant columns (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="joblib model file")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH,
                        help="model metadata JSON")
    parser.add_argument("--threshold", type=threshold_value, default=None,
                        help="probability at or above which an applicant is High risk "
                             "(default: value stored with the model)")
    parser.add_argument("--output", type=Path, default=None, help="save scored applicants to CSV")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    try:
        metadata = load_metadata(args.metadata)
        warn_if_version_differs(metadata)
        model = load_model(args.model)
        applicants = read_applicants(args.input, metadata)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1

    threshold = args.threshold if args.threshold is not None else metadata["decision_threshold"]
    result = score_applicants(model, applicants, threshold)

    LOGGER.info("Scored %d applicant(s) using threshold %.2f", len(result), threshold)
    print(result[["DefaultProbability", "Prediction", "RiskLabel"]].to_string())

    if args.output:
        result.to_csv(args.output, index=False)
        LOGGER.info("Saved scored applicants to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
