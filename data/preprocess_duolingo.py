import pandas as pd
import numpy as np

print("Processing Duolingo Learning Dataset...")

# Loading only part of the dataset to avoid memory crash
df = pd.read_csv("data/learning_traces.13m.csv", nrows=50000)

print("Loaded rows:", len(df))

df["practice_accuracy"] = df["session_correct"] / df["session_seen"]
df["practice_accuracy"] = df["practice_accuracy"].fillna(0)

df["practice_frequency"] = df["history_seen"]
df["recency"] = df["delta"]

ml_data = df[[
    "practice_accuracy",
    "practice_frequency",
    "recency"
]]

# Normalizing values
ml_data = (ml_data - ml_data.min()) / (ml_data.max() - ml_data.min())

ml_data.to_csv("data/duolingo_features.csv", index=False)

print("Saved duolingo_features.csv")