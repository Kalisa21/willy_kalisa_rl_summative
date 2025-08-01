# ppo_training.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv

# Create necessary directories
os.makedirs("models/ppo", exist_ok=True)
os.makedirs("logs/ppo", exist_ok=True)

# Wrap environment
env = FlattenObservation(Monitor(LegalHelpEnv(), filename="logs/ppo/monitor.csv"))
eval_env = FlattenObservation(Monitor(LegalHelpEnv()))

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/ppo/best_model",
    log_path="logs/ppo/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# PPO Agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./logs/ppo/"
)

# Train the agent
model.learn(total_timesteps=50000, callback=eval_callback)
model.save("models/ppo/final_model")
print("✅ PPO training complete.")
