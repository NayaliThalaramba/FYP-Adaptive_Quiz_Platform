import gymnasium as gym
import numpy as np
from gymnasium import spaces

class QuizEnv(gym.Env):
    """
    Gymnasium environment for adaptive gamification reward optimization.
    State: [engagement_score, last_reward_type, questions_seen, avg_performance]
    Actions:
    0 = Points
    1 = Badge
    2 = Progress Bar
    3 = Leaderboard
    Reward: Engagement improvement after reward allocation.
    """
    def __init__(self, max_questions=20):
        super().__init__()
        self.max_questions = max_questions

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 3.0, max_questions, 1.0]),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)

        self.reset_state()

    def reset_state(self):
        self.norm_score = np.random.uniform(0.2, 0.9)
        self.last_reward = float(np.random.choice([0, 1, 2, 3]))
        self.questions_seen = np.random.randint(0, self.max_questions)
        self.avg_confidence = np.clip(
            0.5 + 0.3 * self.norm_score - 0.1 * self.last_reward,
            0.0, 1.0
        )

        self.done = False

        self.state = np.array([
            self.norm_score,
            self.last_reward,
            self.questions_seen,
            self.avg_confidence,
        ], dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.reset_state()
        return self.state, {}

    def step(self, action):

        if self.questions_seen >= self.max_questions:
            self.done = True

        engagement_sensitivity = np.random.choice([0.3, 0.6, 0.9])

        if self.norm_score < 0.4:
            reward_effect = {
                0: 0.06,  
                1: 0.03,
                2: 0.02,
                3: -0.01  
            }

        elif self.norm_score < 0.7:
            reward_effect = {
                0: 0.03,
                1: 0.07,  
                2: 0.05,
                3: 0.01
            }

        else:
            reward_effect = {
                0: 0.01,
                1: 0.03,
                2: 0.05,
                3: 0.08  
            }

        old_score = self.norm_score

        delta = reward_effect[int(action)] * engagement_sensitivity
        self.norm_score = np.clip(self.norm_score + delta, 0.0, 1.0)

        reward = (self.norm_score - old_score) * 10.0
        self.last_reward = float(action)
        self.questions_seen += 1
        self.avg_confidence = np.clip(
            0.5 + 0.4 * self.norm_score - 0.1 * self.last_reward,
            0.0, 1.0
        )

        self.state = np.array([
            self.norm_score,
            self.last_reward,
            self.questions_seen,
            self.avg_confidence,
        ], dtype=np.float32)

        terminated = self.questions_seen >= self.max_questions
        truncated = False

        info = {"engagement": self.norm_score}

        return self.state, reward, terminated, truncated, info

    def get_difficulty_text(self, difficulty):
        if difficulty < 0.5:
            return "easy"
        elif difficulty < 1.5:
            return "medium"
        else:
            return "hard"
