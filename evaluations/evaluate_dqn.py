from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv
import time

def evaluate_model(model_path="models/dqn/dqn_model", episodes=5, render=True):
    # Load environment
    def make_env():
        env = LegalHelpEnv(render_mode="human" if render else None)
        return FlattenObservation(env)

    env = make_env()
    model = DQN.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        print(f"\n🎬 Episode {ep + 1}")
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if render:
                env.render()
                time.sleep(0.3)

        print(f"🏁 Episode {ep + 1} finished in {step} steps | Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate_model()
