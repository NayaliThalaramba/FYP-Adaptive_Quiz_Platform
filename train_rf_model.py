import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

print("Training offline RandomForest model...")

# Loading OULAD data
try:
    train_data = pd.read_csv('data/oulad_training_tiny.csv')
    print(f"Loaded REAL OULAD data: {len(train_data)} samples")
except FileNotFoundError:
    print("OULAD data missing, using synthetic fallback")
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

# Loading Duolingo data
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

duolingo_sample = duolingo_data.sample(len(train_data), replace=True, random_state=42)

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
print(f"RandomForest trained. Accuracy: {accuracy:.1%}")

joblib.dump(model, "rf_model.pkl")
print("Saved trained model as rf_model.pkl")