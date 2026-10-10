
import sys
import os
import numpy as np
import pandas as pd
import joblib

# Use absolute paths because we are inside a package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models directory is sibling to this script
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Import logger directly since it is in the same folder
try:
    import logger

except ImportError:
    # If run from root, we might need mcp_ai_monitor.logger
    try:
        from mcp_ai_monitor import logger

    except ImportError:
        # Fallback: add current dir to path
        sys.path.append(BASE_DIR)
        import logger


def generate_mock_traffic(scenario="mixed", clear_logs=False):
    """
    Generates traffic based on the selected scenario.

    Parameters:
        scenario: 'healthy', 'risky', or 'mixed'
        clear_logs: If True, wipes the CSV before writing.
    """

    print(f"Generating traffic for scenario: {scenario}...")

    # Clear logs if requested
    if clear_logs and os.path.exists(logger.LOG_FILE):
        os.remove(logger.LOG_FILE)
        logger.initialize_log()

    # Load models
    healthy_path = os.path.join(MODELS_DIR, "healthy_model.pkl")
    degraded_path = os.path.join(MODELS_DIR, "degraded_model.pkl")

    if not os.path.exists(healthy_path) or not os.path.exists(degraded_path):
        print(f"CRITICAL: Models not found at {MODELS_DIR}!")
        return

    model_healthy = joblib.load(healthy_path)
    model_degraded = joblib.load(degraded_path)

    # Helper function to log predictions
    def log_batch(data, model, version, ground_truth=None):
        for row in data:
            prob = model.predict_proba([row])[0][1]
            pred = int(prob > 0.5)

            logger.log_prediction(
                row,
                pred,
                prob,
                version,
                ground_truth
            )

    # =========================
    # SCENARIO 1: HEALTHY
    # =========================
    if scenario == "healthy":

        # Generate 60 healthy samples
        data = np.random.normal(
            loc=0.5,
            scale=0.1,
            size=(60, 5)
        )

        log_batch(data, model_healthy, "healthy_v1")

        print("Logged 60 healthy predictions.")

    # =========================
    # SCENARIO 2: RISKY
    # =========================
    elif scenario == "risky":

        # Drifted Data (30 samples)
        drift_data = np.random.normal(
            loc=0.5,
            scale=0.1,
            size=(30, 5)
        )

        drift_data[:, 0] += 0.3

        log_batch(
            drift_data,
            model_healthy,
            "healthy_v1"
        )

        # Biased / Degraded Model (30 samples)
        bias_data = np.random.normal(
            loc=0.5,
            scale=0.1,
            size=(30, 5)
        )

        log_batch(
            bias_data,
            model_degraded,
            "degraded_v1",
            ground_truth=1
        )

        print("Logged 60 risky/drifted predictions.")

    # =========================
    # SCENARIO 3: MIXED
    # =========================
    else:

        # Normal Data
        normal_data = np.random.normal(
            loc=0.5,
            scale=0.1,
            size=(50, 5)
        )

        log_batch(
            normal_data,
            model_healthy,
            "healthy_v1"
        )

        # Drift Data
        drift_data = np.random.normal(
            loc=0.5,
            scale=0.1,
            size=(30, 5)
        )

        drift_data[:, 0] += 0.3

        log_batch(
            drift_data,
            model_healthy,
            "healthy_v1"
        )

        # Bias Data
        bias_data = np.random.normal(
            loc=0.5,
            scale=0.1,
            size=(20, 5)
        )

        log_batch(
            bias_data,
            model_degraded,
            "degraded_v1",
            ground_truth=0
        )

        print("Logged mixed sequence.")

    print(f"Traffic generation complete for {scenario}.")


if __name__ == "__main__":
    generate_mock_traffic()
