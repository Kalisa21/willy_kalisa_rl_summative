# dqn_training.py

import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv

# Create directories
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

# Wrap environment
env = FlattenObservation(Monitor(LegalHelpEnv(), filename="logs/dqn/monitor.csv"))
eval_env = FlattenObservation(Monitor(LegalHelpEnv()))

# Eval callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/dqn/best_model",
    log_path="logs/dqn/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# DQN Agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./logs/dqn/"
)

# Train
model.learn(total_timesteps=30000, callback=eval_callback)
model.save("models/dqn/final_model")
print("✅ DQN training complete.")
