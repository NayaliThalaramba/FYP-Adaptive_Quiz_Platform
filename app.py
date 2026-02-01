from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import os

app = Flask(__name__)

# PERSISTENT HISTORY (saved to file)
HISTORY_FILE = 'student_history.json'
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        student_history = json.load(f)
else:
    student_history = {str(i): [] for i in range(3)}  # "0", "1", "2" as strings

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(student_history, f)

print("🚀 Training ML Model...")

# 10K training data
np.random.seed(42)
train_data = pd.DataFrame({
    'score': np.random.normal(0.7, 0.2, 10000).clip(0,1),
    'time_spent': np.random.poisson(25, 10000),
    'prev_engagement': np.random.normal(0.6, 0.15, 10000).clip(0,1),
    'motivation': np.random.choice([0,1,2], 10000)
})

# SCORE-DRIVEN REWARDS
train_data['optimal_reward'] = np.where(
    (train_data['score'] > 0.85) & (train_data['motivation'] == 0), 3,  # High competitive → Leaderboard
    np.where(train_data['motivation'] == 1, 2,                           # Progress → Progress Bar
             np.where(train_data['score'] > 0.75, 1, 0)))               # Social: Badge/Points

X = train_data[['score', 'time_spent', 'prev_engagement', 'motivation']]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, train_data['optimal_reward'])
print("✅ Model trained!")

students_df = pd.DataFrame({
    'student_id': [0,1,2],
    'motivation_type': ['competitive', 'progress', 'social']
})

def explain_reward(reward, score, motivation):
    if reward == '👑 Leaderboard':
        return "High performance combined with competitive motivation triggered leaderboard-based rewards."
    elif reward == '📈 Progress Bar':
        return "Consistent progress patterns indicate progress-based motivation."
    elif reward == '🏅 Badge':
        return "Achievement-based rewards were selected due to strong quiz completion."
    else:
        return "Point-based rewards were selected to reinforce participation and engagement."


@app.route('/quiz', methods=['POST'])
def take_quiz():
    data = request.json
    student_id = str(data['student_id'])  # String key for JSON
    completion_rate = data['completion_rate']
    session_duration = data['session_duration']
    
    student = students_df[students_df['student_id'] == int(student_id)].iloc[0]
    motivation = students_df[students_df['student_id'] == int(student_id)].index[0]
    
    # REAL STUDENT HISTORY
    history = student_history[student_id]
    prev_engagement = history[-1][2] if history else 0.6
    
    # ML PREDICTS
    features = np.array([[completion_rate, session_duration, prev_engagement, motivation]])
    reward_idx = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    
    # UPDATE HISTORY
    current_engagement = completion_rate * 1.3
    student_history[student_id].append([completion_rate, session_duration, current_engagement])
    save_history()
    
    reward_types = ['⭐ Points', '🏅 Badge', '📈 Progress Bar', '👑 Leaderboard']
    
    reward_name = reward_types[reward_idx]

    return jsonify({
        'reward': reward_name,
        'explanation': explain_reward(reward_name, completion_rate, motivation),
        'engagement_boost': current_engagement,
        'static_score': completion_rate,
        'confidence': confidence,
        'history_length': len(student_history[student_id])
    })


@app.route('/history/<student_id>')
def get_history(student_id):
    return jsonify(student_history[str(student_id)][:5])  # Last 5 quizzes

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/start')
def dashboard():
    return render_template('quiz.html')  

if __name__ == '__main__':
    app.run(debug=True, port=5000)
