import csv
import os
from datetime import datetime

#Dynamic paths - always relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(DATA_DIR, 'predictions.csv')

# Column names for our log
COLUMNS = ['timestamp', 'f0', 'f1', 'f2', 'f3', 'f4',
           'prediction', 'confidence', 'model_version', 'ground_truth']

def initialize_log():
    """Creates a fresh CSV file with the correct column headers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

def log_prediction(features, prediction, confidence, model_version, ground_truth=None):
    """Appends one prediction row to the CSV log."""
    # If file doesn't exist, create it first
    if not os.path.exists(LOG_FILE):
        initialize_log()

    row = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction,
        'confidence': round(confidence, 4),
        'model_version': model_version,
        'ground_truth': ground_truth
    }

    # Log each feature (f0, f1, f2, ...)
    for i, val in enumerate(features[:5]):  # max 5 features for simplicity
        row[f'f{i}'] = round(float(val), 4)

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)
