# a2c_training.py

import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv

# Create necessary directories
os.makedirs("models/a2c", exist_ok=True)
os.makedirs("logs/a2c", exist_ok=True)

# Wrap environment
env = FlattenObservation(Monitor(LegalHelpEnv(), filename="logs/a2c/monitor.csv"))
eval_env = FlattenObservation(Monitor(LegalHelpEnv()))

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/a2c/best_model",
    log_path="logs/a2c/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# A2C Agent
model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.25,
    max_grad_norm=0.5,
    use_rms_prop=True,
    rms_prop_eps=1e-5,
    verbose=1,
    tensorboard_log="./logs/a2c/"
)

# Train the agent
model.learn(total_timesteps=50000, callback=eval_callback)
model.save("models/a2c/final_model")
print("✅ A2C training complete.")
