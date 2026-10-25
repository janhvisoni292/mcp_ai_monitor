import sys
import os
import numpy as np
import pandas as pd
import joblib

# Use absolute paths because we are inside a package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models dir is sibling to this script
MODELS_DIR = os.path.join(BASE_DIR, 'models')

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

def generate_mock_traffic(scenario='mixed', clear_logs=False):
    """
    Generates traffic based on the selected scenario.
    scenario: 'healthy', 'risky', or 'mixed' (default)
    clear_logs: If True, wipes the CSV before writing.
    """
    print(f"Generating traffic for scenario: {scenario}...")
    
    # Clear logs if requested
    if clear_logs and os.path.exists(logger.LOG_FILE):
        os.remove(logger.LOG_FILE)
        logger.initialize_log()
    
    # Load models
    healthy_path = os.path.join(MODELS_DIR, 'healthy_model.pkl')
    degraded_path = os.path.join(MODELS_DIR, 'degraded_model.pkl')
    
    if not os.path.exists(healthy_path) or not os.path.exists(degraded_path):
        print(f"CRITICAL: Models not found at {MODELS_DIR}!")
        return

    model_healthy = joblib.load(healthy_path)
    model_degraded = joblib.load(degraded_path)
    
    # Helper to log a batch
    def log_batch(data, model, version, ground_truth=None):
        for row in data:
            prob = model.predict_proba([row])[0][1]
            pred = int(prob > 0.5)
            logger.log_prediction(row, pred, prob, version, ground_truth)

    # SCENARIO 1: HEALTHY (matching the reference data)
    ref_file = os.path.join(MODELS_DIR, 'reference_data.csv')
    if os.path.exists(ref_file):
        df_ref = pd.read_csv(ref_file)
        # Drop target/healthy column to get features
        target_names = ['target', 'healthy', 'label', 'class', 'output', 'y']
        feat_cols = [c for c in df_ref.columns if c.lower() not in target_names]
        df_feats = df_ref[feat_cols]
    else:
        # Fallback to dummy data
        df_feats = pd.DataFrame(np.random.normal(0.5, 0.1, size=(100, 5)))

    if scenario == 'healthy':
        # Sample 60 rows from the reference features
        data = df_feats.sample(n=min(60, len(df_feats)), replace=True).values
        log_batch(data, model_healthy, 'healthy_v1')
        print(f"Logged {len(data)} healthy predictions using reference distribution.")

    # SCENARIO 2: RISKY (Drift + Bias)
    elif scenario == 'risky':
        # 1. Drifted Data (30 samples) - We shift the values to simulate drift
        drift_raw = df_feats.sample(n=min(30, len(df_feats)), replace=True).values
        # Shift the first feature significantly
        drift_data = drift_raw.copy()
        drift_data[:, 0] *= 2.0 
        log_batch(drift_data, model_healthy, 'healthy_v1') # Input drift
        
        # 2. Biased/Degraded Model (30 samples)
        bias_data = df_feats.sample(n=min(30, len(df_feats)), replace=True).values
        # Use the degraded model
        log_batch(bias_data, model_degraded, 'degraded_v1')
        print("Logged 60 risky/drifted predictions.")

    # SCENARIO 3: MIXED
    else:
        # Mix of normal and drifted
        normal_data = df_feats.sample(n=min(50, len(df_feats)), replace=True).values
        log_batch(normal_data, model_healthy, 'healthy_v1')
        
        drift_raw = df_feats.sample(n=min(30, len(df_feats)), replace=True).values
        drift_data = drift_raw.copy()
        drift_data[:, 0] *= 2.0
        log_batch(drift_data, model_healthy, 'healthy_v1')
        print("Logged mixed sequence.")

    print(f"Traffic generation complete for {scenario}.")

if __name__ == "__main__":
    generate_mock_traffic()
