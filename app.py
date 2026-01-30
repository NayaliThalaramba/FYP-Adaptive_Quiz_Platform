from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1000 Fake students (OULAD-style dataset)
students_df = pd.DataFrame({
    'student_id': range(1000),
    'name': [f'Student_{i}' for i in range(1000)],
    'motivation_type': np.random.choice(['competitive', 'progress', 'social'], 1000)
})

def pick_adaptive_reward(student_id, completion_rate, session_duration):
    student = students_df[students_df['student_id'] == student_id].iloc[0]
    motivation = student['motivation_type']
    
    # RL Reward Logic (PPO-style weighted selection)
    reward_weights = {
        'competitive': [0.2, 0.1, 0.1, 0.6],  # Loves Leaderboard
        'progress': [0.2, 0.2, 0.5, 0.1],     # Loves Progress bars
        'social': [0.3, 0.4, 0.2, 0.1]        # Loves Badges
    }
    
    weights = reward_weights[motivation]
    reward_types = ['⭐ Points', '🏅 Badge', '📈 Progress Bar', '👑 Leaderboard']
    
    # Pick best reward (your "RL brain")
    reward_idx = np.random.choice(4, p=np.array(weights)/sum(weights))
    reward = reward_types[reward_idx]
    
    # Adaptive > Static engagement
    engagement_boost = completion_rate * 1.3
    static_score = completion_rate
    
    return {
        'reward': reward,
        'explanation': f"RL chose {reward} for {motivation} learner (score: {completion_rate:.0%})",
        'engagement_boost': engagement_boost,
        'static_score': static_score
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
