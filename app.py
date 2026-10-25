import sys
import os
import json
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash


# Ensure we can see sibling modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from agent.self_audit_agent import SelfAuditingAgent
from rag.gemini_explainer import explain_report
from mcp_tools import tools
from generate_traffic import generate_mock_traffic
import logger # Import logger to overwrite logs

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app.secret_key = 'secretkey'  #secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False #to suppress a warning from SQLAlchemy
db = SQLAlchemy(app) #initialize the database

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(255))

# Database initialization with app context
with app.app_context(): 
    db.create_all()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        #validations
        if not name or len(name.strip())<2:
            flash('Name must be at least 2 characters long.', 'error')
            return redirect(url_for('register'))
        
        if not email or '@' not in email:
            flash('Please enter a valid email address.', 'error')
            return redirect(url_for('register'))
        
        #password must be at least 8 characters long and a combination of letters and numbers and special characters
        if len(password)<8 or not any(char.isdigit() for char in password)\
              or not any(char.isalpha() for char in password) or not any(not char.isalnum()\
                                                                          for char in password):
            flash('Password must be at least 8 characters long and contain letters, \
                  numbers, and special characters.', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        
        #check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please log in.', 'error')
            return redirect(url_for('register'))
        
        #create new user
        hashed_password = generate_password_hash(password)
        new_user = User(
            name=name.strip(),
            email=email.strip(),
            password=hashed_password
        )
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('register'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        print(f"DEBUG: Found user: {user.name if user else 'None'}")
        if user and check_password_hash(user.password, password):
            print("DEBUG: Password match successful")
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash('Login successful!', 'success')
            return redirect(url_for('upload'))
        else:
            print("DEBUG: Password match failed or user NOT found")
            flash('Invalid email or password.', 'error')
    return render_template('login.html')
# --- LOGOUT ROUTE ---
@app.route('/logout')
def logout():
    session.clear() # This safely destroys the user's login session
    flash('System session terminated securely.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])   
def dashboard():
    # 1. Get Configuration from UI (if POST)
    sensitivity = request.args.get('sensitivity', 'Medium')
    
    # 2. Run Audit with Config
    agent = SelfAuditingAgent(sensitivity=sensitivity)
    report = agent.run_audit()
    
    # 2. Get AI Explanation
    # This calls Gemini API. If no key, it returns a warning.
    explanation = explain_report(report)
    
    # 3. Get History for Chart
    recent_preds = tools.get_recent_predictions(limit=50)
    chart_data = {
        "labels": [i for i in range(len(recent_preds))], # Simple 1..50 index
        "confidence": [p.get('confidence', 0) for p in recent_preds],
        "predictions": [p.get('prediction', 0) for p in recent_preds]
    }
    
    # 3. Render Dashboard
    return render_template(
        'dashboard.html', 
        report=report, 
        explanation=explanation, 
        chart_data=chart_data
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    
    # HANDLE POST
    if 'model_file' not in request.files or 'data_file' not in request.files:
        return "Missing files", 400
        
    model_file = request.files['model_file']
    data_file = request.files['data_file']
    
    if model_file.filename == '' or data_file.filename == '':
        return "No selected file", 400

    # 1. Save Files
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(data_file.filename))
    model_file.save(model_path)
    data_file.save(data_path)
    
    # 2. Process Files (The Automation Logic)
    try:
        # Load Model
        model = joblib.load(model_path)
        
        # Load Data
        print(f"DEBUG: Loading CSV from {data_path}")
        df = pd.read_csv(data_path)
        print(f"DEBUG: CSV loaded. Shape: {df.shape}, Columns: {list(df.columns)}")

        if df.empty:
            return "Error: The uploaded CSV file is empty.", 400
        
        # 1. Identify target column
        target_names = ['target', 'healthy', 'label', 'class', 'output', 'y', 'status']
        actual_target_col = next((c for c in df.columns if c.lower() in target_names), None)
        print(f"DEBUG: Detected target column: {actual_target_col}")

        # 2. Identify feature columns
        feature_cols = [c for c in df.columns if c != actual_target_col]
        X = df[feature_cols]
        print(f"DEBUG: Feature columns: {feature_cols}")
        
        # 3. FEATURE ALIGNMENT LOGIC
        n_expected = None
        if hasattr(model, "n_features_in_"):
            n_expected = model.n_features_in_
            print(f"DEBUG: Model expects {n_expected} features.")
        
        if hasattr(model, "feature_names_in_"):
            expected_names = list(model.feature_names_in_)
            print(f"DEBUG: Model expects specific names: {expected_names}")
            if all(name in df.columns for name in expected_names):
                X = df[expected_names]
            elif n_expected:
                # Force slice to match feature count
                X = X.iloc[:, :n_expected]
        elif n_expected:
            X = X.iloc[:, :n_expected]

        # Ensure we have the right number of features
        if n_expected and X.shape[1] < n_expected:
            return f"Error: Dataset has only {X.shape[1]} features, but model expects {n_expected}.", 400

        # 4. Generate Predictions
        X_input = X.values
        print(f"DEBUG: Running predictions on input shape {X_input.shape}")

        # Wipe old log for fresh analysis
        if os.path.exists(logger.LOG_FILE):
             os.remove(logger.LOG_FILE)
        logger.initialize_log()
              
        # Run Prediction
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)
            if probs.shape[1] >= 2:
                confidences = probs[:, 1]
            else:
                confidences = probs[:, 0] # Fallback for single-class probes
            predictions = model.predict(X_input)
        else:
            predictions = model.predict(X_input)
            confidences = [1.0] * len(predictions)
            
        print(f"DEBUG: Generated {len(predictions)} predictions.")

        # 5. Log to CSV
        targets = df[actual_target_col].values if actual_target_col else [None] * len(predictions)
        
        for i, row_values in enumerate(X_input):
            try:
                logger.log_prediction(
                    features=row_values,
                    prediction=int(predictions[i]), 
                    confidence=float(confidences[i]), 
                    model_version='uploaded_model_v1',
                    ground_truth=int(targets[i]) if (targets[i] is not None and not pd.isna(targets[i])) else None
                )
            except Exception as log_err:
                print(f"DEBUG: Error logging row {i}: {log_err}")
                continue
            
        # 6. UPDATE BASELINE
        ref_path = os.path.join(MODELS_DIR, 'reference_data.csv')
        df_ref = X.head(100).copy() # Use X (features only)
        # Standardize column names for the tools (f0, f1...)
        df_ref.columns = [f'f{j}' for j in range(len(df_ref.columns))]
        if actual_target_col:
            df_ref['target'] = df[actual_target_col].head(100).values
        df_ref.to_csv(ref_path, index=False)
        print(f"DEBUG: Updated baseline reference at {ref_path}")
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing files: {str(e)}", 500

@app.route('/simulate/<scenario>')
def simulate(scenario):
    # Trigger generation logic
    # We clear logs to make the impact immediate and obvious
    generate_mock_traffic(scenario=scenario, clear_logs=True)
    return redirect(url_for('dashboard'))

@app.route('/repair')
def repair():
    agent = SelfAuditingAgent()
    result = agent.attempt_repair()
    
    if result['success']:
        # To show the fix, we should "swap" the active model logic to the new one or 
        # just regenerate healthy traffic to prove it works.
        # For demo: We generate healthy traffic immediately to simulate "After-Repair" state
        generate_mock_traffic(scenario='healthy', clear_logs=True)
        return redirect(url_for('dashboard', repaired='true'))
    else:
        return "Repair Failed", 500

@app.route('/download/<file_type>')
def download_artifact(file_type):
    # Determine path
    # We use absolute paths derived from base
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    if file_type == 'model':
        path = os.path.join(MODELS_DIR, 'repaired_model.pkl')
        filename = "repaired_agent_model.pkl"
    elif file_type == 'data':
        path = os.path.join(MODELS_DIR, 'repaired_data.csv')
        filename = "cleaned_training_data.csv"
    else:
        return "Invalid file type", 400
        
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=filename)
    else:
        return "File not found (Run repair first)", 404

if __name__ == '__main__':
    app.run(debug=True)