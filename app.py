from flask import Flask, request, render_template, redirect, url_for, session, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import numpy as np
import joblib
import os

# Load environment variables from .env file
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
model_path  = os.path.join(BASE_DIR, 'cnn_ecg_model.h5')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# ── Load CNN model and scaler safely ──────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    model  = load_model(model_path)
    scaler = joblib.load(scaler_path)
    MODEL_LOADED = True
except Exception as e:
    print(f"[ERROR] Could not load model or scaler: {e}")
    model  = None
    scaler = None
    MODEL_LOADED = False

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)

# CSRF Protection
csrf = CSRFProtect(app)

# ── Flask-Login setup ─────────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── Database setup ────────────────────────────────────────────────────────────
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── ECG Input Range Constants (clinically plausible values) ───────────────────
ECG_RANGES = {
    'rr_interval':  (0.3,  2.0),   # seconds
    'p_wave':       (0.05, 0.25),  # seconds
    'qrs_complex':  (0.05, 0.30),  # seconds
    't_wave':       (0.05, 0.40),  # seconds
    'qt_interval':  (0.20, 0.70),  # seconds
}

# ── User Model ────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id          = db.Column(db.String(80), primary_key=True)   # username as PK
    password    = db.Column(db.String(200), nullable=False)
    answer_1    = db.Column(db.String(200), nullable=False)   # hashed
    answer_2    = db.Column(db.String(200), nullable=False)   # hashed

    @property
    def is_active(self):
        return True

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

# ── ECG Prediction ────────────────────────────────────────────────────────────
# IMPORTANT — Feature remapping
# During training (train_model.py), p_wave and t_wave were computed as
# normalised energy ratios relative to the R-peak amplitude, NOT raw durations:
#   p_wave = clip(0.08 + 0.05 * p_energy, 0.05, 0.25)
#   t_wave = clip(0.10 + 0.08 * t_energy, 0.05, 0.40)
# Qt_interval was always derived via Bazett: 0.39 * sqrt(rr).
# The scaler was fitted on these computed values, so we must remap user-entered
# durations into the same space before transforming — otherwise the model gets
# completely out-of-distribution input and gives garbage predictions.

def _remap_features(ecg_values):
    """
    Remap the 5 user-entered durations (seconds) into the feature space
    that was used during training, so the scaler produces correct z-scores.

    Parameters (all seconds):
        ecg_values = [rr_interval, p_wave, qrs_complex, t_wave, qt_interval]

    Returns a numpy array with the same ordering but values aligned to the
    training distribution.
    """
    rr, p_dur, qrs, t_dur, _qt_user = ecg_values

    # P-wave: convert entered duration to an energy-ratio estimate.
    # Normal P ≈ 0.10 s → p_energy ≈ 0.40 → p_wave_feat ≈ 0.10
    # We approximate p_energy as the ratio of p_duration to rr.
    p_energy = float(np.clip(p_dur / max(rr, 0.3), 0.0, 5.0))
    p_feat   = float(np.clip(0.08 + 0.05 * p_energy, 0.05, 0.25))

    # T-wave: same approach with its scaling constant from train_model.py
    t_energy = float(np.clip(t_dur / max(rr, 0.3), 0.0, 5.0))
    t_feat   = float(np.clip(0.10 + 0.08 * t_energy, 0.05, 0.40))

    # QT: re-derive using Bazett correction (exactly as in training)
    qt_feat = float(np.clip(0.39 * np.sqrt(max(rr, 0.3)), 0.20, 0.70))

    return np.array([[rr, p_feat, qrs, t_feat, qt_feat]], dtype=np.float32)


def _clinical_rules(ecg_values):
    """
    Apply standard clinical thresholds to detect obvious arrhythmia or normal.
    Returns  1 (arrhythmia vote), -1 (normal vote), or 0 (uncertain).

    Criteria:
        Arrhythmia: RR < 0.60 s (HR > 100 bpm, tachycardia)
                    RR > 1.20 s (HR < 50 bpm, bradycardia)
                    QRS > 0.12 s (wide complex — bundle branch block / VT)
                    QTc > 0.45 s (prolonged QT)
        Normal:     0.60 ≤ RR ≤ 1.20  AND  QRS ≤ 0.12  AND  QTc ≤ 0.44
    """
    rr, _p, qrs, _t, _qt = ecg_values
    qtc = 0.39 * np.sqrt(max(rr, 0.3))   # Bazett corrected QT

    arrhythmia_flags = [
        rr  < 0.60,   # tachycardia
        rr  > 1.20,   # bradycardia
        qrs > 0.12,   # wide QRS
        qtc > 0.45,   # long QTc
    ]
    normal_flags = [
        0.60 <= rr <= 1.20,
        qrs <= 0.12,
        qtc <= 0.44,
    ]

    if any(arrhythmia_flags):
        return 1    # vote: arrhythmia
    if all(normal_flags):
        return -1   # vote: normal
    return 0        # uncertain — defer to model


def predict_ecg(ecg_values):
    """
    Predict arrhythmia status.
    1. Remap user-entered features to match the training distribution.
    2. Run the CNN model.
    3. Apply clinical rule checks as a second opinion.
    Returns 'Arrhythmia' or 'Normal'.
    """
    # ── Clinical rules vote (independent of model) ────────────────────────
    rules_vote = _clinical_rules(ecg_values)

    # ── Model vote ───────────────────────────────────────────────────────
    features = _remap_features(ecg_values)          # shape (1, 5)
    scaled   = scaler.transform(features)
    prob     = float(model.predict(scaled, verbose=0)[0][0])  # sigmoid output
    model_vote = 1 if prob > 0.5 else -1

    # ── Combine votes ────────────────────────────────────────────────────
    # Clinical rules take full priority when they are certain.
    # The model is only the sole decider when rules are uncertain (0).
    if rules_vote == 1:
        return "Arrhythmia"   # rules clearly say arrhythmia
    if rules_vote == -1:
        return "Normal"        # rules clearly say normal — trust them
    # rules_vote == 0 (uncertain) — defer entirely to model
    return "Arrhythmia" if prob > 0.5 else "Normal"

def validate_ecg_inputs(data):
    """
    Extract and validate the 5 ECG fields.
    Returns (ecg_values list, error string or None).
    """
    fields = ['rr_interval', 'p_wave', 'qrs_complex', 't_wave', 'qt_interval']
    ecg_values = []
    for field in fields:
        raw = data.get(field)
        if raw is None or str(raw).strip() == '':
            return None, f"Missing value for {field.replace('_', ' ').title()}."
        try:
            value = float(raw)
        except (ValueError, TypeError):
            return None, f"Invalid value for {field.replace('_', ' ').title()}. Must be a number."
        low, high = ECG_RANGES[field]
        if not (low <= value <= high):
            return None, (
                f"{field.replace('_', ' ').title()} must be between {low} and {high} seconds. "
                f"You entered: {value}"
            )
        ecg_values.append(value)
    return ecg_values, None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    if not session.get('questions_answered'):
        return redirect(url_for('questions'))
    return render_template('index.html')


@app.route('/questions', methods=['GET', 'POST'])
@login_required
def questions():
    if request.method == 'POST':
        symptoms = {
            'chest_pain':           request.form.get('chest_pain', '0'),
            'shortness_of_breath':  request.form.get('shortness_of_breath', '0'),
            'fatigue':              request.form.get('fatigue', '0'),
            'dizziness':            request.form.get('dizziness', '0'),
            'palpitations':         request.form.get('palpitations', '0'),
        }
        session['symptoms']           = symptoms
        session['questions_answered'] = True
        return redirect(url_for('index'))
    return render_template('questions.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model is not available. Please contact the administrator.'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received.'}), 400

        ecg_values, err = validate_ecg_inputs(data)
        if err:
            return jsonify({'error': err}), 400

        ecg_result = predict_ecg(ecg_values)
        symptoms   = session.get('symptoms', {})
        yes_count  = list(symptoms.values()).count('1')

        # ── Build suggestion ──────────────────────────────────────────────────
        if ecg_result == 'Arrhythmia' and yes_count >= 3:
            suggestion = "High risk detected. ECG shows arrhythmia and multiple symptoms are present. Seek medical attention immediately."
        elif ecg_result == 'Arrhythmia' and yes_count > 0:
            suggestion = "ECG indicates arrhythmia with some symptoms present. Consult a cardiologist promptly."
        elif ecg_result == 'Arrhythmia':
            suggestion = "ECG indicates arrhythmia. Visit a physician promptly even if you feel no symptoms."
        elif yes_count >= 3:
            suggestion = "ECG appears normal but multiple symptoms are present. Monitor closely and consult a doctor."
        elif yes_count == 2:
            suggestion = "ECG is normal. Some symptoms reported — monitor and consult a doctor if they persist."
        elif yes_count == 1:
            suggestion = "ECG is normal. Mild symptom reported. Maintain a healthy lifestyle."
        else:
            suggestion = "ECG is normal and no symptoms reported. You appear to be healthy."

        cure        = "" if ecg_result == 'Normal' else "Consult a cardiologist for a personalised treatment plan."
        precautions = (
            "Maintain a healthy lifestyle, exercise regularly, and attend routine check-ups."
            if ecg_result == 'Normal'
            else "Stay active, avoid smoking, limit alcohol, manage stress, and follow your doctor's advice."
        )

        # Store result in session and redirect (avoids document.write)
        session['result'] = {
            'ecg_result':  ecg_result,
            'cure':        cure,
            'precautions': precautions,
            'suggestion':  suggestion,
        }
        return jsonify({'redirect': url_for('result')})

    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/result')
@login_required
def result():
    data = session.pop('result', None)
    if not data:
        return redirect(url_for('index'))
    return render_template(
        'result.html',
        ecg_result  = data['ecg_result'],
        cure        = data['cure'],
        precautions = data['precautions'],
        suggestion  = data['suggestion'],
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user     = User.query.get(username)

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('questions_answered', None)
    session.pop('symptoms', None)
    session.pop('result', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        password  = request.form.get('password', '')
        answer_1  = request.form.get('answer_1', '').strip().lower()
        answer_2  = request.form.get('answer_2', '').strip().lower()

        if not username or not password or not answer_1 or not answer_2:
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('register'))

        if User.query.get(username):
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))

        new_user = User(
            id       = username,
            password = generate_password_hash(password, method='pbkdf2:sha256'),
            answer_1 = generate_password_hash(answer_1, method='pbkdf2:sha256'),
            answer_2 = generate_password_hash(answer_2, method='pbkdf2:sha256'),
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        answer_1 = request.form.get('answer_1', '').strip().lower()
        answer_2 = request.form.get('answer_2', '').strip().lower()
        user     = User.query.get(username)

        if (user
                and check_password_hash(user.answer_1, answer_1)
                and check_password_hash(user.answer_2, answer_2)):
            session['reset_user'] = username
            return redirect(url_for('reset_password', username=username))

        flash('Incorrect answers or username not found.', 'error')
    return render_template('forgot_password.html')


@app.route('/reset_password/<username>', methods=['GET', 'POST'])
def reset_password(username):
    # Guard: only allow if the user arrived via forgot_password
    if session.get('reset_user') != username:
        flash('Unauthorised access to password reset.', 'error')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password     = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if len(new_password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif new_password != confirm_password:
            flash('Passwords do not match.', 'error')
        else:
            user = User.query.get(username)
            if user:
                user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
                db.session.commit()
                session.pop('reset_user', None)
                flash('Password reset successfully. Please log in.', 'success')
                return redirect(url_for('login'))
            flash('User not found.', 'error')
    return render_template('reset_password.html', username=username)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Set FLASK_DEBUG=false in .env to disable debug in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug_mode)
