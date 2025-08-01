from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv
import os

def train_ppo_model(total_timesteps=100_000, save_path="models/pg/ppo_model"):
    # Create and wrap environment
    def make_env():
        env = LegalHelpEnv(render_mode=None)
        env = FlattenObservation(env)  # PPO works well with flat obs
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # Initialize PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_logs/ppo/"
    )

    print("🚀 Starting PPO training...")
    model.learn(total_timesteps=total_timesteps)
    print("✅ PPO training complete!")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"💾 PPO model saved to {save_path}")

if __name__ == "__main__":
    train_ppo_model()
