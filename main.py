import argparse
import os
import time
import numpy as np
import imageio
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN, PPO, A2C
from reinforce_training import PolicyNetwork
from custom_env import LegalHelpEnv

def run_random_agent(episodes=3, render_mode="human", save_gif=False):
    env = FlattenObservation(LegalHelpEnv(render_mode=render_mode))
    frames = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        print(f"\n🎲 Episode {ep+1} — Random Agent")

        while not done:
            if render_mode == "rgb_array":
                frames.append(env.render())
            elif render_mode == "human":
                env.render()
                time.sleep(0.3)

            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"🎯 Reward: {total_reward:.2f}")

    env.close()

    if save_gif and frames:
        os.makedirs("gifs", exist_ok=True)
        imageio.mimsave("gifs/random_agent.gif", frames, fps=4)
        print("📸 GIF saved to gifs/random_agent.gif")

def evaluate_model(algo, model_path, episodes=5, render_mode="human"):
    env = FlattenObservation(LegalHelpEnv(render_mode=render_mode))

    if algo.lower() == "dqn":
        model = DQN.load(model_path)
    elif algo.lower() == "ppo":
        model = PPO.load(model_path)
    elif algo.lower() == "a2c":
        model = A2C.load(model_path)
    elif algo.lower() == "reinforce":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        model = PolicyNetwork(obs_dim, act_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise ValueError("Unsupported algorithm!")

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0

        print(f"\n📦 Evaluating {algo.upper()} — Episode {ep+1}")

        while not done:
            if algo.lower() == "reinforce":
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    probs = model(obs_tensor)
                    action = torch.argmax(probs).item()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if render_mode == "human":
                env.render()
                time.sleep(0.3)

        print(f"🏁 Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "random"], required=True)
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], help="Algorithm to train/eval")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    parser.add_argument("--save_gif", action="store_true")

    args = parser.parse_args()

    if args.mode == "random":
        run_random_agent(episodes=args.episodes, render_mode=args.render, save_gif=args.save_gif)

    elif args.mode == "eval":
        if not args.model_path or not args.algo:
            raise ValueError("Both --algo and --model_path are required for evaluation.")
        evaluate_model(args.algo, args.model_path, episodes=args.episodes, render_mode=args.render)

    elif args.mode == "train":
        if not args.algo:
            raise ValueError("You must specify --algo for training.")
        if args.algo == "dqn":
            from dqn_training import train_dqn_model
            train_dqn_model()
        elif args.algo == "ppo":
            from ppo_training import train_ppo_model
            train_ppo_model()
        elif args.algo == "a2c":
            from a2c_training import train_a2c_model
            train_a2c_model()
        elif args.algo == "reinforce":
            from reinforce_training import train_reinforce
            train_reinforce()
