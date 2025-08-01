from custom_env import LegalHelpEnv
import time

def test_environment():
    env = LegalHelpEnv(render_mode="human")
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)

    total_reward = 0
    done = False

    print("\n--- Starting random action loop ---")

    for step in range(20):
        if done:
            print("Episode done.")
            break

        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step+1} | Action: {action} | Reward: {reward:.2f} | Total: {total_reward:.2f} | Visited: {info['visited']}")
        
        env.render()
        time.sleep(0.5)

    env.close()
    print("Finished test.")

if __name__ == "__main__":
    test_environment()
