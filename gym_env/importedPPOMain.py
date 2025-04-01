import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Function to plot learning curve
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == "__main__":
    # Create and wrap the environment
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])  # ✅ Proper VecEnv wrapping
    env = VecNormalize(env, norm_obs=True, norm_reward=False)  # ✅ Normalize observations

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1,batch_size=64,n_epochs=4,learning_rate=0.0003)

    # Training parameters
    n_games = 1000
    figure_file = "plots/cartpole.png"
    best_score = -float("inf")
    score_history = []

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()  # ✅ VecEnv reset (no `info` returned)

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)  # ✅ Gymnasium step API
            score += reward

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            # model.save("ppo_cartpole")  # ✅ Save model
            # env.save("ppo_cartpole_env.pkl")  # ✅ Save VecNormalize stats

        print(f"Episode {i}, Score: {score}, Avg Score: {avg_score}")

    # env.close()  # ✅ Ensure env is closed before saving
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
