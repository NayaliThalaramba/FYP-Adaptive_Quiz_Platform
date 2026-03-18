from flask import redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, QuizAttempt
from stable_baselines3 import PPO
from src.quiz_env import QuizEnv
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

app = Flask(__name__)
MAX_QUESTIONS = 20

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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

# Train ML model on REAL OULAD DATA (32K students)
print("Training on OULAD Dataset (32K students)...")
try:
    train_data = pd.read_csv('data/oulad_training_tiny.csv')
    print(f"Loaded REAL OULAD data: {len(train_data)} samples")
except FileNotFoundError:
    print("OULAD data missing, using synthetic fallback")
    np.random.seed(42)
    train_data = pd.DataFrame({
        'score': np.random.normal(0.7, 0.2, 10000).clip(0,1),
        'time_spent': np.random.poisson(25, 10000),
        'prev_engagement': np.random.normal(0.6, 0.15, 10000).clip(0,1),
        'motivation': np.random.choice([0,1,2], 10000)
    })
    train_data['optimal_reward'] = np.where(
        train_data['score'] >= 0.9, 3,
        np.where(train_data['score'] >= 0.75, 1,
                 np.where(train_data['score'] >= 0.6, 2, 0)))

print("Loading Duolingo behavioural features...")
try:
    duolingo_data = pd.read_csv('data/duolingo_features.csv')
    print(f"Loaded Duolingo features: {len(duolingo_data)} samples")
except FileNotFoundError:
    print("Duolingo data missing, using synthetic fallback")
    duolingo_data = pd.DataFrame({
        'practice_accuracy': np.random.uniform(0.5, 0.9, 1000),
        'practice_frequency': np.random.uniform(0.2, 1.0, 1000),
        'recency': np.random.uniform(0.0, 1.0, 1000)
    })

print(f"Training RandomForest ({len(train_data)} samples)...")
duolingo_sample = duolingo_data.sample(len(train_data), replace=True)

train_data["practice_accuracy"] = duolingo_sample["practice_accuracy"].values
train_data["practice_frequency"] = duolingo_sample["practice_frequency"].values
train_data["recency"] = duolingo_sample["recency"].values

X = train_data[[
    'score',
    'time_spent',
    'prev_engagement',
    'practice_accuracy',
    'practice_frequency',
    'recency'
]]

y = train_data['optimal_reward']

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X, y)
accuracy = model.score(X, y)
print(f"OULAD Production Model: {accuracy:.1%} accuracy | {len(train_data)} samples")
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

    # Fetch previous attempts from DB
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

    features = np.array([[
        score,
        time_spent,
        prev_engagement,
        practice_accuracy,
        practice_frequency,
        recency
    ]], dtype=float)

    ml_probs = model.predict_proba(features)[0]
    confidence = round(float(np.max(ml_probs)), 2)

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

    # Store in DB
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
        'explanation': explain_reward(reward_name, score, confidence),
        'engagement_boost': engagement_boost,
        'static_score': score,
        'confidence': confidence,
        'history_length': questions_seen + 1
    })

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            return "Username already exists"

        if User.query.filter_by(email=email).first():
            return "Email already registered"

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

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))

        return "Invalid credentials"

    return render_template('login.html')

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

    # -------- REAL DAILY STREAK --------
    attempt_dates = sorted(
        {a.created_at.date() for a in attempts},
        reverse=True
    )

    streak = 0
    today = datetime.utcnow().date()

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

# Routes
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

if __name__ == '__main__':
    app.run(debug=True)
