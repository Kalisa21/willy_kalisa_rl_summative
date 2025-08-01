from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv
import os

def train_a2c_model(total_timesteps=100_000, save_path="models/pg/a2c_model"):
    # Create and wrap environment
    def make_env():
        env = LegalHelpEnv(render_mode=None)
        env = FlattenObservation(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # Initialize A2C model
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,
        gamma=0.99,
        n_steps=5,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_logs/a2c/"
    )

    print("🚀 Starting A2C training...")
    model.learn(total_timesteps=total_timesteps)
    print("✅ A2C training complete!")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"💾 A2C model saved to {save_path}")

if __name__ == "__main__":
    train_a2c_model()
