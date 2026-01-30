from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# STUDENT HISTORY (NEW - tracks each student's past performance!)
student_history = {0: [], 1: [], 2: []}  # Alice=0, Bob=1, Charlie=2

print("🚀 Training ML Model...")

# Generate 10K OULAD-style training data
np.random.seed(42)
train_data = pd.DataFrame({
    'score': np.random.normal(0.7, 0.2, 10000).clip(0,1),
    'time_spent': np.random.poisson(25, 10000),
    'prev_engagement': np.random.normal(0.6, 0.15, 10000).clip(0,1),
    'motivation': np.random.choice([0,1,2], 10000, p=[0.4,0.35,0.25])
})

# Real target: optimal reward type (score-dependent!)
train_data['optimal_reward'] = np.where(
    (train_data['motivation']==0) & (train_data['score']>0.8), 3,  # Competitive+high score → Leaderboard
    np.where(train_data['motivation']==1, 2,                          # Progress → Progress Bar
             np.where(train_data['score']>0.7, 1, 0)))               # Social → Badge/Points by score

# TRAIN MODEL
X = train_data[['score', 'time_spent', 'prev_engagement', 'motivation']]
y = train_data['optimal_reward']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained! Test Accuracy: {accuracy:.1%}")

# 1000 Fake students
students_df = pd.DataFrame({
    'student_id': range(1000),
    'name': [f'Student_{i}' for i in range(1000)],
    'motivation_type': np.random.choice(['competitive', 'progress', 'social'], 1000)
})

def pick_adaptive_reward(student_id, completion_rate, session_duration):
    student = students_df[students_df['student_id'] == student_id].iloc[0]
    motivation_map = {'competitive': 0, 'progress': 1, 'social': 2}
    motivation = motivation_map[student['motivation_type']]
    
    # USE REAL STUDENT HISTORY (not hardcoded 0.6!)
    history = student_history[student_id]
    prev_engagement = history[-1][2] if history else 0.6  # Last engagement OR default
    
    # ML PREDICTION with REAL data
    features = np.array([[completion_rate, session_duration, prev_engagement, motivation]])
    reward_idx = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    
    # UPDATE STUDENT HISTORY (learns from this quiz!)
    current_engagement = completion_rate * 1.3
    student_history[student_id].append([completion_rate, session_duration, current_engagement])
    
    # Keep only last 5 quizzes per student (memory efficient)
    if len(student_history[student_id]) > 5:
        student_history[student_id] = student_history[student_id][-5:]
    
    reward_types = ['⭐ Points', '🏅 Badge', '📈 Progress Bar', '👑 Leaderboard']
    reward = reward_types[reward_idx]
    
    return {
        'reward': reward,
        'explanation': f"**ML Model** predicted {reward} for {student['motivation_type']} learner (confidence: {confidence:.0%})",
        'engagement_boost': current_engagement,
        'static_score': completion_rate,
        'model_used': True,
        'confidence': confidence,
        'history_length': len(student_history[student_id])
    }

@app.route('/')
def dashboard():
    return '''
    <!DOCTYPE html>
    <html><head><title>Adaptive Quiz</title></head>
    <body>
        <h1>🧠 Adaptive ML Quiz Platform LIVE!</h1>
        <iframe src="https://fyp-adaptive-quiz-platform.onrender.com/quiz.html" width="100%" height="800px"></iframe>
    </body>
    </html>
    '''  # Inline HTML for now

@app.route('/quiz', methods=['POST'])
def take_quiz():
    data = request.json
    result = pick_adaptive_reward(data['student_id'], data['completion_rate'], data['session_duration'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
