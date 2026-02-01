from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import os

app = Flask(__name__)

# PERSISTENT STUDENT HISTORY
HISTORY_FILE = 'student_history.json'
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        student_history = json.load(f)
else:
    student_history = {str(i): [] for i in range(3)}

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(student_history, f)

# TRAIN RANDOM FOREST MODEL
np.random.seed(42)

train_data = pd.DataFrame({
    'score': np.random.normal(0.7, 0.2, 10000).clip(0, 1),
    'time_spent': np.random.poisson(25, 10000),
    'prev_engagement': np.random.normal(0.6, 0.15, 10000).clip(0, 1),
    'motivation': np.random.choice([0, 1, 2], 10000)
})

train_data['optimal_reward'] = np.where(
    train_data['score'] >= 0.9, 3,
    np.where(train_data['score'] >= 0.75, 1,
             np.where(train_data['score'] >= 0.6, 2, 0))
)

X = train_data[['score', 'time_spent', 'prev_engagement', 'motivation']]
y = train_data['optimal_reward']

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X, y)


# STUDENTS
students_df = pd.DataFrame({
    'student_id': [0, 1, 2],
    'motivation_type': ['competitive', 'progress', 'social']
})

reward_types = ['⭐ Points', '🏅 Badge', '📈 Progress Bar', '👑 Leaderboard']

# EXPLAINABLE AI
def explain_reward(reward, score):
    if reward == '👑 Leaderboard':
        return "Excellent performance detected. Competitive rewards maximize motivation at high achievement levels."
    if reward == '🏅 Badge':
        return "Strong quiz performance rewarded with achievement-based incentives."
    if reward == '📈 Progress Bar':
        return "Moderate performance benefits from progress visualization to sustain engagement."
    return "Participation-based rewards help encourage continued learning."


# MAIN QUIZ ROUTE

@app.route('/quiz', methods=['POST'])
def take_quiz():
    data = request.json
    student_id = str(data['student_id'])
    score = float(data['completion_rate'])
    time_spent = float(data['session_duration'])

    motivation = int(student_id)  # 0,1,2

    history = student_history[student_id]
    prev_engagement = history[-1][2] if history else 0.6

    # ML Prediction
    features = np.array([[score, time_spent, prev_engagement, motivation]])
    ml_reward_idx = model.predict(features)[0]

    # RULE-BASED ADAPTIVE 
    if score >= 0.9:
        reward_idx = 3
    elif score >= 0.75:
        reward_idx = 1
    elif score >= 0.6:
        reward_idx = 2
    else:
        reward_idx = 0

    reward_name = reward_types[reward_idx]

    
    engagement_boost = round(score * 1.3, 2)
    student_history[student_id].append([score, time_spent, engagement_boost])
    save_history()

    return jsonify({
        'reward': reward_name,
        'explanation': explain_reward(reward_name, score),
        'engagement_boost': engagement_boost,
        'static_score': score,
        'confidence': round(np.max(model.predict_proba(features)), 2),
        'history_length': len(student_history[student_id])
    })


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/start')
def dashboard():
    return render_template('quiz.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
