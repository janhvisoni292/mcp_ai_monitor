import pandas as pd
import numpy as np
import os


# Paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LOG_FILE = os.path.join(
    DATA_DIR,
    "predictions.csv"
)

REF_FILE = os.path.join(
    MODELS_DIR,
    "reference_data.csv"
)


def get_recent_predictions(limit: int = 50) -> list:
    """Returns the last N prediction records as a list of dictionaries."""

    if not os.path.exists(LOG_FILE):
        return []

    try:
        df = pd.read_csv(LOG_FILE)

        return df.tail(limit).to_dict(
            orient="records"
        )

    except Exception as e:
        return [
            {
                "error": str(e)
            }
        ]


def get_ref_data_stats() -> dict:
    """Returns mean and std for each feature from the reference/training data."""

    if not os.path.exists(REF_FILE):
        return {
            "error": "Reference data not found."
        }

    df = pd.read_csv(REF_FILE)

    stats = {}

    for col in df.columns:

        if col == "target":
            continue

        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }

    return stats


def get_current_model_health() -> dict:
    """Calculates average confidence and accuracy from recent logs."""

    recents = get_recent_predictions(
        limit=100
    )

    if not recents:
        return {
            "status": "No data"
        }

    df = pd.DataFrame(recents)

    avg_conf = float(
        df["confidence"].mean()
    )

    df_gt = df.dropna(
        subset=["ground_truth"]
    )

    accuracy = None

    if not df_gt.empty:

        correct = df_gt[
            df_gt["prediction"]
            == df_gt["ground_truth"].astype(int)
        ]

        accuracy = float(
            len(correct) / len(df_gt)
        )

    return {
        "samples_analyzed": len(df),
        "avg_confidence": avg_conf,
        "estimated_accuracy": accuracy
    }


def check_feature_drift(
    recent_window: int = 30
) -> dict:
    """
    Compares live data means vs reference means
    using a Z-score approach.

    If the difference is > 2 standard deviations,
    it's flagged as DRIFT.
    """

    ref_stats = get_ref_data_stats()

    if "error" in ref_stats:
        return ref_stats

    recents = get_recent_predictions(
        limit=recent_window
    )

    if not recents:
        return {
            "status": "Not enough data"
        }

    df_live = pd.DataFrame(recents)

    drift_report = {}

    for feature in [
        "f0",
        "f1",
        "f2",
        "f3",
        "f4"
    ]:

        if feature not in df_live.columns:
            continue

        live_mean = df_live[feature].mean()

        ref_mean = ref_stats[feature]["mean"]

        ref_std = ref_stats[feature]["std"]

        if ref_std == 0:
            ref_std = 0.001

        deviation = (
            abs(live_mean - ref_mean)
            / ref_std
        )

        drift_report[feature] = {
            "drift_score": float(deviation),
            "is_drifting": bool(
                deviation > 2.0
            )
        }

    return drift_report