import gymnasium as gym
import numpy as np
from gymnasium import spaces

class QuizEnv(gym.Env):
    """
    Gymnasium environment for adaptive quiz difficulty.
    State: [score, difficulty, questions_seen, avg_confidence]
    Actions: 0=easy, 1=medium, 2=hard
    Reward: correct_answer + bonus for staying in suitable difficulty range.
    """
    def __init__(self, max_questions=20):
        super().__init__()
        self.max_questions = max_questions

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 2.0, max_questions, 1.0]),
            dtype=np.float32,
        )

        # action space: 0=easy, 1=medium, 2=hard
        self.action_space = spaces.Discrete(3)

        self.reset_state()

    def reset_state(self):
        self.score = 0.0
        self.difficulty = 1.0  # start at medium (1)
        self.questions_seen = 0
        self.avg_confidence = 0.5
        self.done = False

        self.norm_score = self.score / 100.0
        self.norm_questions = self.questions_seen / self.max_questions
        self.state = np.array([
            self.norm_score,
            self.difficulty,
            self.questions_seen,
            self.avg_confidence,
        ], dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.reset_state()
        return self.state, {}

    def step(self, action):
        # action: 0=easy, 1=medium, 2=hard
        if self.questions_seen >= self.max_questions:
            self.done = True

        # simulate correct/incorrect based on difficulty and current level
        difficulty_value = float(action)  # 0,1,2

        # harder questions: more reward if you get it right, but more risk
        p_correct = 0.9 - 0.3 * difficulty_value / 2.0
        p_correct = np.clip(p_correct, 0.3, 0.95)

        correct = np.random.random() < p_correct

        if correct:
            self.score += 1.0
            # strongly encourage extreme difficulty choice
            if difficulty_value == 0.0:
                reward = 3.0   # heavy bonus for easy
            elif difficulty_value == 2.0:
                reward = 3.0   # heavy bonus for hard
            else:
                reward = 1.0   # only base for medium
        else:
            reward = -0.2  # keep penalty

        self.difficulty = float(action)
        self.questions_seen += 1
        self.avg_confidence = 0.5 + 0.3 * (self.score / (self.questions_seen + 1)) - 0.15 * difficulty_value

        # normalize state
        self.norm_score = self.score / 100.0
        self.norm_questions = self.questions_seen / self.max_questions
        self.state = np.array([
            self.norm_score,
            difficulty_value,
            self.questions_seen,
            self.avg_confidence,
        ], dtype=np.float32)

        truncated = False
        terminated = self.questions_seen >= self.max_questions
        self.done = terminated

        info = {"score": self.score, "difficulty": difficulty_value}

        return self.state, reward, terminated, truncated, info

    def get_difficulty_text(self, difficulty):
        if difficulty < 0.5:
            return "easy"
        elif difficulty < 1.5:
            return "medium"
        else:
            return "hard"
