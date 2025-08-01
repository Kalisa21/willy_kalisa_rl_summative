from stable_baselines3 import A2C
from gymnasium.wrappers import FlattenObservation
from custom_env import LegalHelpEnv
import time

def evaluate_model(model_path="models/pg/a2c_model", episodes=5, render=True):
    # Create evaluation environment
    def make_env():
        env = LegalHelpEnv(render_mode="human" if render else None)
        return FlattenObservation(env)

    env = make_env()
    model = A2C.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\n🎬 A2C Episode {ep + 1}")

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
