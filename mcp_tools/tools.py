import pandas as pd
import numpy as np
import os

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Parent of mcp_tools is mcp_ai_monitor
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_FILE = os.path.join(DATA_DIR, 'predictions.csv')
REF_FILE = os.path.join(MODELS_DIR, 'reference_data.csv')

def get_recent_predictions(limit: int = 50) -> list:
    """
    MCP Tool: Fetches the most recent predictions made by the system.
    Returns a list of dictionaries.
    """
    if not os.path.exists(LOG_FILE):
        return []
    
    try:
        df = pd.read_csv(LOG_FILE)
        # return last 'limit' rows as a list of dicts
        return df.tail(limit).to_dict(orient='records')
    except Exception as e:
        return [{"error": str(e)}]

def get_ref_data_stats() -> dict:
    """
    MCP Tool: Returns statistics (mean, std) of the 'Healthy' training data.
    Used by the agent to detect if current data is drifting away from this baseline.
    """
    if not os.path.exists(REF_FILE):
        return {"error": "Reference data not found. Train healthy model first."}
    
    df = pd.read_csv(REF_FILE)
    stats = {}
    target_names = ['target', 'healthy', 'label', 'class', 'output', 'y']
    for col in df.columns:
        if col.lower() in target_names: continue
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    return stats

def get_current_model_health() -> dict:
    """
    MCP Tool: Calculates basic health metrics from recent logs.
    Returns average confidence and accuracy (if ground truth is available).
    """
    recents = get_recent_predictions(limit=100)
    if not recents:
        return {"status": "No data"}
    
    df = pd.DataFrame(recents)
    
    # Calculate Average Confidence
    avg_conf = float(df['confidence'].mean())
    
    # Calculate Accuracy if Ground Truth exists (it might be NaN for live data)
    # We filter rows where ground_truth is not null/NaN
    df_gt = df.dropna(subset=['ground_truth'])
    accuracy = None
    if not df_gt.empty:
        # Convert ground_truth to int for comparison
        correct = df_gt[df_gt['prediction'] == df_gt['ground_truth'].astype(int)]
        accuracy = float(len(correct) / len(df_gt))
        
    return {
        "samples_analyzed": len(df),
        "avg_confidence": avg_conf,
        "estimated_accuracy": accuracy
    }

def check_feature_drift(recent_window: int = 30) -> dict:
    """
    MCP Tool: Compares recent live traffic mean vs reference mean.
    Returns a dictionary flagging which features are drifting.
    """
    # 1. Get Reference Stats
    ref_stats = get_ref_data_stats()
    if "error" in ref_stats: return ref_stats
    
    # 2. Get Live Data
    recents = get_recent_predictions(limit=recent_window)
    if not recents: return {"status": "Not enough data"}
    df_live = pd.DataFrame(recents)
    
    drift_report = {}
    
    # 3. Compare (Simple Z-Score like check)
    # If live_mean is more than 2 std devs away from ref_mean -> DRIFT
    for feature in ['f0', 'f1', 'f2', 'f3', 'f4']:
        if feature not in df_live.columns: continue
        
        live_mean = df_live[feature].mean()
        ref_mean = ref_stats[feature]['mean']
        ref_std = ref_stats[feature]['std']
        
        # Calculate deviation
        # Avoid division by zero
        if ref_std == 0: ref_std = 0.001
        
        deviation = abs(live_mean - ref_mean) / ref_std
        
        drift_report[feature] = {
            "drift_score": float(deviation),
            "is_drifting": bool(deviation > 2.0) # Threshold
        }
        
    return drift_report
