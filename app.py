from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)


print("🚀 Training ML Model...")


np.random.seed(42)
train_data = pd.DataFrame({
    'score': np.random.normal(0.7, 0.2, 10000).clip(0,1),
    'time_spent': np.random.poisson(25, 10000),
    'prev_engagement': np.random.normal(0.6, 0.15, 10000).clip(0,1),
    'motivation': np.random.choice([0,1,2], 10000, p=[0.4,0.35,0.25])
})


train_data['optimal_reward'] = np.where(
    (train_data['motivation']==0) & (train_data['score']>0.8), 3,
    np.where(train_data['motivation']==1, 2,
             np.random.choice([0,1,2,3], p=[0.3,0.4,0.2,0.1])))


X = train_data[['score', 'time_spent', 'prev_engagement', 'motivation']]
y = train_data['optimal_reward']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained! Test Accuracy: {accuracy:.1%}")


students_df = pd.DataFrame({
    'student_id': range(1000),
    'name': [f'Student_{i}' for i in range(1000)],
    'motivation_type': np.random.choice(['competitive', 'progress', 'social'], 1000)
})


def pick_adaptive_reward(student_id, completion_rate, session_duration):
    student = students_df[students_df['student_id'] == student_id].iloc[0]
    motivation_map = {'competitive': 0, 'progress': 1, 'social': 2}
    motivation = motivation_map[student['motivation_type']]
    
    
    features = np.array([[completion_rate, session_duration, 0.6, motivation]])
    reward_idx = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    
    reward_types = ['⭐ Points', '🏅 Badge', '📈 Progress Bar', '👑 Leaderboard']
    reward = reward_types[reward_idx]
    
    return {
        'reward': reward,
        'explanation': f"**ML Model** predicted {reward} for {student['motivation_type']} learner (confidence: {confidence:.0%})",
        'engagement_boost': completion_rate * 1.3,
        'static_score': completion_rate,
        'model_used': True,
        'confidence': confidence
    }

@app.route('/')
def dashboard():
    return render_template('quiz.html')

@app.route('/quiz', methods=['POST'])
def take_quiz():
    data = request.json
    result = pick_adaptive_reward(data['student_id'], data['completion_rate'], data['session_duration'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
