
# ============================================================
# IMPORTS
# ============================================================

import sys
import os
import json

import pandas as pd
import joblib
import numpy as np

from flask import (
    Flask,
    render_template,
    flash,
    redirect,
    url_for,
    request,
    session,
    send_file
)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)
from werkzeug.utils import secure_filename


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from agent.self_audit_agent import SelfAuditingAgent
from rag.gemini_explainer import explain_report
from mcp_tools import tools
from generate_traffic import generate_mock_traffic
import logger


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# APP CONFIGURATION
# ============================================================

app.secret_key = 'secretkey'


# Upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(
    BASE_DIR,
    'uploads'
)

os.makedirs(
    app.config['UPLOAD_FOLDER'],
    exist_ok=True
)


# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# ============================================================
# DATABASE
# ============================================================

db = SQLAlchemy(app)


# ============================================================
# USER MODEL
# ============================================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )


# ============================================================
# CREATE DATABASE
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# ROUTE 1: HOME PAGE
# ============================================================

@app.route('/')
def home():

    return render_template(
        'home.html'
    )


# ============================================================
# ROUTE 2: REGISTER
# ============================================================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        name = request.form.get(
            'name',
            ''
        ).strip()

        email = request.form.get(
            'email',
            ''
        ).strip().lower()

        password = request.form.get(
            'password',
            ''
        )

        confirm_password = request.form.get(
            'confirm_password',
            ''
        )


        # ----------------------------------------------------
        # NAME VALIDATION
        # ----------------------------------------------------

        if not name or len(name) < 2:

            flash(
                'Name must be at least 2 characters long.',
                'error'
            )

            return redirect(
                url_for('register')
            )


        # ----------------------------------------------------
        # EMAIL VALIDATION
        # ----------------------------------------------------

        if not email or '@' not in email:

            flash(
                'Please enter a valid email address.',
                'error'
            )

            return redirect(
                url_for('register')
            )


        # ----------------------------------------------------
        # PASSWORD VALIDATION
        # ----------------------------------------------------

        if (
            len(password) < 8
            or not any(char.isdigit() for char in password)
            or not any(char.isalpha() for char in password)
            or not any(not char.isalnum() for char in password)
        ):

            flash(
                'Password must be at least 8 characters long '
                'and contain letters, numbers, and special characters.',
                'error'
            )

            return redirect(
                url_for('register')
            )


        # ----------------------------------------------------
        # CONFIRM PASSWORD
        # ----------------------------------------------------

        if password != confirm_password:

            flash(
                'Passwords do not match.',
                'error'
            )

            return redirect(
                url_for('register')
            )


        # ----------------------------------------------------
        # CHECK EXISTING USER
        # ----------------------------------------------------

        existing_user = User.query.filter_by(
            email=email
        ).first()


        if existing_user:

            flash(
                'Email already registered. Please log in.',
                'error'
            )

            return redirect(
                url_for('register')
            )


        # ----------------------------------------------------
        # CREATE USER
        # ----------------------------------------------------

        hashed_password = generate_password_hash(
            password
        )


        new_user = User(
            name=name,
            email=email,
            password=hashed_password
        )


        try:

            db.session.add(
                new_user
            )

            db.session.commit()


            flash(
                'Registration successful! Please log in.',
                'success'
            )


            return redirect(
                url_for('login')
            )


        except Exception as e:

            db.session.rollback()

            print(
                "REGISTRATION DATABASE ERROR:",
                repr(e)
            )


            flash(
                'An error occurred during registration.',
                'error'
            )

            return redirect(
                url_for('register')
            )


    return render_template(
        'register.html'
    )


# ============================================================
# ROUTE 3: LOGIN
# ============================================================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form.get(
            'email',
            ''
        ).strip().lower()

        password = request.form.get(
            'password',
            ''
        )


        user = User.query.filter_by(
            email=email
        ).first()


        # ----------------------------------------------------
        # CHECK LOGIN
        # ----------------------------------------------------

        if user and check_password_hash(
            user.password,
            password
        ):

            session['user_id'] = user.id

            session['user_name'] = user.name


            flash(
                'Login successful!',
                'success'
            )


            # Login → Upload page
            return redirect(
                url_for('upload')
            )


        else:

            flash(
                'Invalid email or password.',
                'error'
            )


    return render_template(
        'login.html'
    )


# ============================================================
# ROUTE 4: DASHBOARD / SELF-AUDIT
# ============================================================

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )


    # --------------------------------------------------------
    # USER NAME
    # --------------------------------------------------------

    name = session.get(
        'user_name'
    )


    # --------------------------------------------------------
    # SENSITIVITY
    # --------------------------------------------------------

    sensitivity = request.args.get(
        'sensitivity',
        'Medium'
    )


    # --------------------------------------------------------
    # SELF AUDITING AGENT
    # --------------------------------------------------------

    try:

        agent = SelfAuditingAgent(
            sensitivity=sensitivity
        )

        report = agent.run_audit()

        explanation = explain_report(
            report
        )

    except Exception as e:

        print(
            "AUDIT ERROR:",
            repr(e)
        )

        report = {}

        explanation = (
            "Unable to generate audit report."
        )


    # --------------------------------------------------------
    # RECENT PREDICTIONS
    # --------------------------------------------------------

    try:

        recent_preds = tools.get_recent_predictions(
            limit=50
        )

    except Exception as e:

        print(
            "PREDICTION LOG ERROR:",
            repr(e)
        )

        recent_preds = []


    # --------------------------------------------------------
    # CHART DATA
    # --------------------------------------------------------

    chart_data = {

        "labels": list(
            range(
                len(recent_preds)
            )
        ),

        "confidence": [
            p.get(
                'confidence',
                0
            )
            for p in recent_preds
        ]
    }


    # --------------------------------------------------------
    # RENDER DASHBOARD
    # --------------------------------------------------------

    return render_template(
        'dashboard.html',
        name=name,
        report=report,
        explanation=explanation,
        chart_data=chart_data
    )


# ============================================================
# ROUTE 5: UPLOAD PAGE
# ============================================================

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )


    # --------------------------------------------------------
    # USER NAME
    # --------------------------------------------------------

    name = session.get(
        'user_name'
    )


    # --------------------------------------------------------
    # GET REQUEST
    # --------------------------------------------------------

    if request.method == 'GET':

        return render_template(
            'upload.html',
            name=name
        )


    # --------------------------------------------------------
    # GET FILES
    # --------------------------------------------------------

    model_file = request.files.get(
        'model_file'
    )

    data_file = request.files.get(
        'data_file'
    )


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not model_file or not data_file:

        flash(
            'Please upload both model and dataset files.',
            'error'
        )

        return redirect(
            url_for('upload')
        )


    if model_file.filename == '' or data_file.filename == '':

        flash(
            'Please select both files.',
            'error'
        )

        return redirect(
            url_for('upload')
        )


    # --------------------------------------------------------
    # SECURE FILENAMES
    # --------------------------------------------------------

    model_filename = secure_filename(
        model_file.filename
    )

    data_filename = secure_filename(
        data_file.filename
    )


    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    model_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        model_filename
    )

    data_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        data_filename
    )


    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    model_file.save(
        model_path
    )

    data_file.save(
        data_path
    )


    # --------------------------------------------------------
    # PROCESS MODEL + DATA
    # --------------------------------------------------------

    try:

        # Load model
        model = joblib.load(
            model_path
        )


        # Load CSV
        df = pd.read_csv(
            data_path
        )


        # ----------------------------------------------------
        # FIND FEATURES
        # ----------------------------------------------------

        feature_cols = [
            c
            for c in df.columns
            if c.lower() != 'target'
        ]


        if not feature_cols:

            raise ValueError(
                'No feature columns found in dataset.'
            )


        X = df[
            feature_cols
        ]


        # ----------------------------------------------------
        # INITIALIZE LOGGER
        # ----------------------------------------------------

        if os.path.exists(
            logger.LOG_FILE
        ):

            os.remove(
                logger.LOG_FILE
            )


        logger.initialize_log()


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        if hasattr(
            model,
            'predict_proba'
        ):

            probs = model.predict_proba(
                X
            )

        else:

            probs = None


        predictions = model.predict(
            X
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if probs is not None:

            # Binary classification
            if probs.shape[1] == 2:

                confidences = probs[:, 1]

            else:

                confidences = np.max(
                    probs,
                    axis=1
                )

        else:

            confidences = [
                1.0
            ] * len(predictions)


        # ----------------------------------------------------
        # LOG PREDICTIONS
        # ----------------------------------------------------

        for i, row in enumerate(
            X.values
        ):

            try:

                prediction_value = int(
                    predictions[i]
                )

            except (ValueError, TypeError):

                prediction_value = predictions[i]


            logger.log_prediction(

                row,

                prediction_value,

                float(
                    confidences[i]
                ),

                'uploaded_model_v1'
            )


        flash(
            'Model and dataset uploaded successfully!',
            'success'
        )


        # After upload → dashboard
        return redirect(
            url_for('dashboard')
        )


    except Exception as e:

        print(
            "UPLOAD ERROR:",
            repr(e)
        )


        flash(
            f'Error while processing files: {str(e)}',
            'error'
        )


        return redirect(
            url_for('upload')
        )


# ============================================================
# ROUTE 6: SIMULATE SCENARIO
# ============================================================

@app.route('/simulate/<scenario>')
def simulate(scenario):

    # Login check
    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )


    try:

        generate_mock_traffic(
            scenario=scenario,
            clear_logs=True
        )

        return redirect(
            url_for('dashboard')
        )


    except Exception as e:

        print(
            "SIMULATION ERROR:",
            repr(e)
        )

        return (
            f"Simulation Error: {str(e)}",
            500
        )


# ============================================================
# ROUTE 7: AUTO-REPAIR
# ============================================================

@app.route('/repair')
def repair():

    # Login check
    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )


    try:

        agent = SelfAuditingAgent()

        result = agent.attempt_repair()


        if result.get(
            'success',
            False
        ):

            generate_mock_traffic(
                scenario='healthy',
                clear_logs=True
            )


            return redirect(
                url_for(
                    'dashboard',
                    repaired='true'
                )
            )


        return (
            'Repair Failed',
            500
        )


    except Exception as e:

        print(
            "REPAIR ERROR:",
            repr(e)
        )

        return (
            f"Repair Error: {str(e)}",
            500
        )


# ============================================================
# ROUTE 8: DOWNLOAD ARTIFACTS
# ============================================================

@app.route('/download/<file_type>')
def download_artifact(file_type):

    # Login check
    if 'user_id' not in session:

        return redirect(
            url_for('login')
        )


    MODELS_DIR = os.path.join(
        BASE_DIR,
        'models'
    )


    paths = {

        'model': (
            os.path.join(
                MODELS_DIR,
                'repaired_model.pkl'
            ),
            'repaired_model.pkl'
        ),

        'data': (
            os.path.join(
                MODELS_DIR,
                'repaired_data.csv'
            ),
            'cleaned_data.csv'
        )
    }


    # --------------------------------------------------------
    # CHECK TYPE
    # --------------------------------------------------------

    if file_type not in paths:

        return (
            'Invalid type',
            400
        )


    path, filename = paths[
        file_type
    ]


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    if os.path.exists(path):

        return send_file(
            path,
            as_attachment=True,
            download_name=filename
        )


    return (
        'File not found. Run repair first.',
        404
    )


# ============================================================
# ROUTE 9: LOGOUT
# ============================================================

@app.route('/logout')
def logout():

    session.clear()


    flash(
        'You have been logged out.',
        'success'
    )


    return redirect(
        url_for('login')
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == '__main__':

    app.run(
        debug=True
    )
