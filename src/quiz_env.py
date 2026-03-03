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

        # action space: 0=easy, 1=medium, 2=hard
        self.action_space = spaces.Discrete(4)

        self.reset_state()

    def reset_state(self):
        # Randomize initial engagement to match real app distribution
        self.norm_score = np.random.uniform(0.2, 0.9)

        # Random previous reward
        self.last_reward = float(np.random.choice([0, 1, 2, 3]))

        # Random progress in quiz
        self.questions_seen = np.random.randint(0, self.max_questions)

        # Confidence derived from engagement
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

        # Simulated user engagement sensitivity
        engagement_sensitivity = np.random.choice([0.3, 0.6, 0.9])

        # State-dependent reward shaping
        if self.norm_score < 0.4:
            # Low engagement students
            reward_effect = {
                0: 0.06,   # Points help low performers
                1: 0.03,
                2: 0.02,
                3: -0.01   # Leaderboard discourages weak students
            }

        elif self.norm_score < 0.7:
            # Medium engagement students
            reward_effect = {
                0: 0.03,
                1: 0.07,   # Badge best here
                2: 0.05,
                3: 0.01
            }

        else:
            # High engagement students
            reward_effect = {
                0: 0.01,
                1: 0.03,
                2: 0.05,
                3: 0.08   # Leaderboard best for strong students
            }

        old_score = self.norm_score

        delta = reward_effect[int(action)] * engagement_sensitivity
        self.norm_score = np.clip(self.norm_score + delta, 0.0, 1.0)

        # reward = improvement in engagement
        reward = (self.norm_score - old_score) * 10.0

        # update state tracking
        self.last_reward = float(action)
        self.questions_seen += 1

        # simulate slight confidence change
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
