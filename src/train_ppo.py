import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.quiz_env import QuizEnv

def train_ppo():
    
    env = make_vec_env(lambda: QuizEnv(max_questions=20), n_envs=1)

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
    )

    # train
    model.learn(total_timesteps=100000)
    model.save("ppo_quiz_final")

    print("PPO model trained and saved as 'ppo_quiz_final.zip'")

if __name__ == "__main__":
    train_ppo()
