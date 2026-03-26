import numpy as np
import pandas as pd
import os
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, QuizAttempt
from stable_baselines3 import PPO
from src.quiz_env import QuizEnv
from datetime import datetime, timedelta, UTC


app = Flask(__name__)
MAX_QUESTIONS = 20

app.config['SECRET_KEY'] = 'supersecretkey'
database_url = os.environ.get("DATABASE_URL")

if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

PPO_MODEL_PATH = "ppo_quiz_final"

def load_ppo_model():
    print("PPO RL Production Model: loading...")
    try:
        model = PPO.load(PPO_MODEL_PATH)
        print("PPO RL Production Model: active (100K timesteps)")
        return model
    except Exception as e:
        print("PPO RL Production Model: failed to load, using rule-based fallback")
        print(e)
        return None

ppa_model = load_ppo_model()
print("ppa_model =", ppa_model)

print("Loading offline RandomForest model...")
model = joblib.load("rf_model.pkl")
print("RandomForest model loaded successfully.")

print("Loading SHAP explainer...")
explainer = shap.TreeExplainer(model)
print("SHAP explainer loaded successfully.")

reward_types = ['Points', 'Badge', 'Progress Bar', 'Leaderboard']


# XAI
def explain_reward(reward, score, confidence):
    if reward == 'Leaderboard':
        return (
            "You achieved a very high score. "
            "The model predicts competitive rewards work best for high performers. "
            f"(Model confidence: {confidence})"
        )
    if reward == 'Badge':
        return (
            "Strong performance detected. "
            "Achievement-based rewards reinforce mastery and motivation."
        )
    if reward == 'Progress Bar':
        return (
            "Moderate performance observed. "
            "Progress-based feedback helps maintain engagement and learning momentum."
        )
    return (
        "Lower performance detected. "
        "Participation rewards encourage continued effort without pressure."
    )


# Main Quiz API
@app.route('/quiz', methods=['POST'])
@login_required
def take_quiz():
    data = request.json

    score = float(data['completion_rate'])
    time_spent = float(data['session_duration'])

    # Fetching previous attempts from DB
    attempts = QuizAttempt.query.filter_by(
        user_id=current_user.id
    ).order_by(QuizAttempt.id).all()

    questions_seen = len(attempts)

    last_reward = 0.0
    if attempts:
        last_reward = reward_types.index(attempts[-1].reward_type)

    avg_confidence = 0.5
    if attempts:
        recent_scores = [a.score for a in attempts[-3:]]
        avg_confidence = np.mean(recent_scores)

    # ML Prediction
    prev_engagement = avg_confidence

    practice_accuracy = score
    practice_frequency = min(1.0, len(attempts) / 10)
    recency = 1.0 if not attempts else 0.5

    features_df = pd.DataFrame([{
        'score': score,
        'time_spent': time_spent,
        'prev_engagement': prev_engagement,
        'practice_accuracy': practice_accuracy,
        'practice_frequency': practice_frequency,
        'recency': recency
    }])

    ml_probs = model.predict_proba(features_df)[0]
    confidence = round(float(np.max(ml_probs)), 2)

    ml_pred = model.predict(features_df)[0]
    feature_names = features_df.columns.tolist()

    try:
        ml_class_pos = list(model.classes_).index(ml_pred)

        shap_raw = explainer.shap_values(features_df)

        if isinstance(shap_raw, list):
            class_shap = shap_raw[ml_class_pos][0]
        else:
            shap_arr = np.array(shap_raw)

            if shap_arr.ndim == 3:
                if shap_arr.shape[0] == 1 and shap_arr.shape[2] == len(model.classes_):
                    class_shap = shap_arr[0, :, ml_class_pos]
                elif shap_arr.shape[0] == len(model.classes_):
                    class_shap = shap_arr[ml_class_pos, 0, :]
                else:
                    class_shap = np.zeros(len(feature_names))
            elif shap_arr.ndim == 2:
                class_shap = shap_arr[0]
            else:
                class_shap = np.zeros(len(feature_names))

        shap_explanation = {
            feature_names[i]: round(float(class_shap[i]), 3)
            for i in range(len(feature_names))
        }

        top_shap = dict(
            sorted(
                shap_explanation.items(),
                key=lambda item: abs(item[1]),
                reverse=True
            )[:3]
        )

        print("SHAP TOP FEATURES:", top_shap)

    except Exception as e:
        print("SHAP explanation failed:", e)
        top_shap = {}

    # PPO State
    obs = np.array([
        score,
        last_reward,
        min(float(questions_seen), 20.0),
        avg_confidence
    ], dtype=np.float32).reshape(1, -1)

    print(f"DEBUG: state={obs}")

    if ppa_model is None:
        if score >= 0.9:
            reward_idx = 3
        elif score >= 0.75:
            reward_idx = 1
        elif score >= 0.6:
            reward_idx = 2
        else:
            reward_idx = 0
    else:
        action, _ = ppa_model.predict(obs, deterministic=True)
        reward_idx = int(action[0])

    reward_name = reward_types[reward_idx]
    engagement_boost = round(score * 1.3, 2)

    base_explanation = explain_reward(reward_name, score, confidence)

    if top_shap:
        shap_readable_names = {
            "score": "quiz score",
            "time_spent": "time spent",
            "prev_engagement": "previous engagement",
            "practice_accuracy": "practice accuracy",
            "practice_frequency": "practice frequency",
            "recency": "recency"
        }

        shap_parts = []
        for feature, value in top_shap.items():
            label = shap_readable_names.get(feature, feature.replace("_", " "))
            direction = "increased" if value > 0 else "reduced"
            shap_parts.append(f"{label} ({direction} influence: {abs(value):.3f})")

        shap_text = "; ".join(shap_parts)
        full_explanation = f"{base_explanation} Key factors influencing this reward were: {shap_text}."
    else:
        full_explanation = base_explanation

    new_attempt = QuizAttempt(
        score=score,
        time_spent=time_spent,
        engagement_boost=engagement_boost,
        reward_type=reward_name,
        user_id=current_user.id
    )

    db.session.add(new_attempt)
    db.session.commit()

    return jsonify({
        'reward': reward_name,
        'explanation': full_explanation,
        'engagement_boost': engagement_boost,
        'static_score': score,
        'confidence': confidence,
        'history_length': questions_seen + 1,
        'shap': top_shap,
    })

@app.route('/register', methods=['GET', 'POST'])
def register():
    def render_register_with_data(clear_fields=None):
        clear_fields = clear_fields or []

        form_data = {
            "first_name": request.form.get("first_name", "").strip(),
            "last_name": request.form.get("last_name", "").strip(),
            "email": request.form.get("email", "").strip().lower(),
            "username": request.form.get("username", "").strip(),
            "password": "",
            "confirm_password": ""
        }

        for field in clear_fields:
            if field in form_data:
                form_data[field] = ""

        return render_template('register.html', form_data=form_data)

    if request.method == 'POST':
        first_name = request.form['first_name'].strip()
        last_name = request.form['last_name'].strip()
        email = request.form['email'].strip().lower()
        username = request.form['username'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if any(char.isspace() for char in username):
            flash("Username cannot contain spaces.", "error")
            return render_register_with_data(clear_fields=["username"])

        if len(first_name) < 2:
            flash("First name must be at least 2 characters long.", "error")
            return render_register_with_data(clear_fields=["first_name"])

        if len(last_name) < 2:
            flash("Last name must be at least 2 characters long.", "error")
            return render_register_with_data(clear_fields=["last_name"])

        if len(username) < 4:
            flash("Username must be at least 4 characters long.", "error")
            return render_register_with_data(clear_fields=["username"])

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_register_with_data(clear_fields=["password", "confirm_password"])

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_register_with_data(clear_fields=["password", "confirm_password"])

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return render_register_with_data(clear_fields=["username"])

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return render_register_with_data(clear_fields=["email"])

        hashed_pw = generate_password_hash(password)

        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            username=username,
            password=hashed_pw
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html', form_data={})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        if any(char.isspace() for char in username):
            flash("Username cannot contain spaces.", "error")
            return render_template('login.html', form_data={"username": ""})

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))

        flash("Invalid credentials. Please try again.", "error")
        return render_template('login.html', form_data={"username": username})

    return render_template('login.html', form_data={})

@app.route('/profile')
@login_required
def profile():
    attempts = QuizAttempt.query.filter_by(
        user_id=current_user.id
    ).order_by(QuizAttempt.id).all()

    total_attempts = len(attempts)

    avg_score = round(
        sum(a.score for a in attempts) / total_attempts, 2
    ) if total_attempts > 0 else 0

    scores = [a.score for a in attempts]
    engagement = [a.engagement_boost for a in attempts]

    return render_template(
        'profile.html',
        user=current_user,
        total_attempts=total_attempts,
        avg_score=avg_score,
        attempts=attempts,
        scores=scores,
        engagement=engagement
    )

@app.route('/leaderboard')
@login_required
def leaderboard():

    users = User.query.filter_by(role="student").all()
    leaderboard_data = []

    for user in users:
        attempts = QuizAttempt.query.filter_by(user_id=user.id).all()

        if attempts:
            avg_score = sum(a.score for a in attempts) / len(attempts)
            total_engagement = sum(a.engagement_boost for a in attempts)

            leaderboard_data.append({
                "username": user.username,
                "avg_score": round(avg_score, 2),
                "engagement": round(total_engagement, 2),
                "attempts": len(attempts)
            })

    leaderboard_data.sort(key=lambda x: x["engagement"], reverse=True)

    return render_template("leaderboard.html", leaderboard=leaderboard_data)

@app.route("/progress")
@login_required
def progress():

    attempts = QuizAttempt.query.filter_by(
        user_id=current_user.id
    ).all()

    attempts_count = len(attempts)

    total_engagement = sum(a.engagement_boost for a in attempts)

    progress_percent = min(attempts_count * 10, 100)

    attempt_dates = sorted(
        {a.created_at.date() for a in attempts},
        reverse=True
    )

    streak = 0
    today = datetime.now(UTC).date()

    for i, d in enumerate(attempt_dates):
        expected_day = today - timedelta(days=i)

        if d == expected_day:
            streak += 1
        else:
            break

    return render_template(
        "progress.html",
        attempts=attempts_count,
        engagement=round(total_engagement, 2),
        progress=progress_percent,
        streak=streak
    )

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        flash("Access denied: Admins only.")
        return redirect(url_for('dashboard'))

    users = User.query.filter_by(role="student").all()
    attempts = QuizAttempt.query.all()

    total_users = len(users)
    total_attempts = len(attempts)

    avg_engagement = 0
    if attempts:
        avg_engagement = round(
            sum(a.engagement_boost for a in attempts) / total_attempts, 2
        )

    reward_counts = {
        "Points": 0,
        "Badge": 0,
        "Progress Bar": 0,
        "Leaderboard": 0
    }

    for a in attempts:
        reward_counts[a.reward_type] += 1

    engagement_by_day = {}

    for a in attempts:
        day = a.created_at.strftime("%Y-%m-%d")

        if day not in engagement_by_day:
            engagement_by_day[day] = 0

        engagement_by_day[day] = round(
            engagement_by_day.get(day, 0) + a.engagement_boost, 2
        )

    engagement_by_day = dict(sorted(engagement_by_day.items()))

    return render_template(
        "admin.html",
        total_users=total_users,
        total_attempts=total_attempts,
        avg_engagement=avg_engagement,
        reward_counts=reward_counts,
        engagement_trend=engagement_by_day  
    )


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/start')
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for('admin_dashboard'))

    return render_template('quiz.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('welcome'))

with app.app_context():
    db.create_all()

    print("Database driver:", db.engine.url.drivername)
    print("Database URL:", db.engine.url)

    admin_user = User.query.filter_by(username="admin").first()
    if not admin_user:
        admin_user = User(
            first_name="System",
            last_name="Admin",
            email="admin@adaptivequiz.com",
            username="admin",
            password=generate_password_hash("Admin123!"),
            role="admin"
        )
        db.session.add(admin_user)
        db.session.commit()
        print("Default admin user created.")
        
if __name__ == '__main__':
    app.run(debug=True)
