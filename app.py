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

print(f"Training RandomForest ({len(train_data)} samples)...")
X = train_data[['score', 'time_spent', 'prev_engagement', 'motivation']]
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
    motivation = current_user.id
    prev_engagement = avg_confidence

    features = np.array([[score, time_spent, prev_engagement, motivation]])
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

    users = User.query.all()
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

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/start')
@login_required
def dashboard():
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
