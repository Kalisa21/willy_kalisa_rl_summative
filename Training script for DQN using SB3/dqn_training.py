from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv
import os

def train_dqn_model(total_timesteps=100_000, save_path="models/dqn/dqn_model"):
    # Create environment with flattened observations for DQN
    def make_env():
        env = LegalHelpEnv(render_mode=None)
        env = FlattenObservation(env)  # DQN requires flat obs
        env = Monitor(env)  # for logging
        return env

    env = DummyVecEnv([make_env])

    # Instantiate model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        gamma=0.95,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./tensorboard_logs/dqn/"
    )

    print("📦 Starting DQN training...")
    model.learn(total_timesteps=total_timesteps)
    print("✅ DQN training complete!")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    train_dqn_model()
