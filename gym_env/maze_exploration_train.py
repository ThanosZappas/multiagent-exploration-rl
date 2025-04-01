import os
import datetime

import gymnasium as gym
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, A2C, SAC
import policies

import maze_exploration_env


def run_q(episodes, is_training=True, render_mode=None):
    env = gym.make('maze-exploration-v0', render_mode=render_mode)

    if is_training:
        # Q Table dimensions: [row, col, actions]
        q = np.zeros((env.unwrapped.grid_rows, env.unwrapped.grid_columns, env.action_space.n))
    else:
        with open('models/v0_maze-exploration.pkl', 'rb') as model:
            q = pickle.load(model)

    learning_rate = 0.9  # alpha
    discount_factor = 0.9  # gamma
    epsilon = 0.2

    steps_per_episode = np.zeros(episodes)

    for i in range(episodes):
        if render_mode == 'human':
            print(f'Episode {i}')
        state, _ = env.reset()
        terminated = False
        step_count = 0

        while not terminated:
            # Epsilon-greedy action selection
            if is_training and random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_index = tuple(state)
                action = np.argmax(q[state_index])

            new_state, reward, terminated, _, _ = env.step(action)

            # Q-learning update:
            state_index = tuple(state)
            new_state_index = tuple(new_state)
            q[state_index + (action,)] = q[state_index + (action,)] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state_index]) - q[state_index + (action,)]
            )

            state = new_state
            step_count += 1

        steps_per_episode[i] = step_count
        # Decay epsilon linearly
        # epsilon = max(epsilon / episodes, 0.2)

    if is_training:
        with open('models/v0_maze-exploration.pkl', 'wb') as model:
            pickle.dump(q, model)

    # Graph steps
    sum_steps = np.zeros(episodes)
    for t in range(episodes):
        sum_steps[t] = np.mean(steps_per_episode[max(0, t - 100):(t + 1)])  # Average steps per 100 episodes
    plt.plot(sum_steps)
    plt.savefig('logs/v0_training.png')

    env.close()
    return q, steps_per_episode


# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3_ppo():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = 'cpu'
    env = gym.make('maze-exploration-v0')

    # Use Proximal Policy (PPO) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_dir}/{time}"
    model = PPO(policies.MlpPolicy, env, verbose=1, device=device, tensorboard_log=log_dir, ent_coef=0.001)

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 1000
    iters = 0
    for iters in range(200):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # train
        model.save(f"{model_dir}/PPO_{time}/ppo_{TIMESTEPS * iters}")  # Save a trained model every TIMESTEPS
        # iters += 1
    # model.save("models/test_model")  # Save a trained model
    vec_env = model.get_env()
    observation = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation)
        observation, rewards, done, _, info = env.step(action)
        env.render()
    env.close()


def train_sb3_a2c():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = 'cpu'
    env = gym.make('maze-exploration-v0')

    # Use Proximal Policy (PPO) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_dir}/{time}"
    model = A2C(policies.MlpPolicy, env, verbose=1, device=device, tensorboard_log=log_dir, ent_coef=0.001)

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 1000
    iters = 0
    for iters in range(200):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # train
        model.save(f"{model_dir}/A2C_{time}/a2c_{TIMESTEPS * iters}")  # Save a trained model every TIMESTEPS
        # iters += 1
    # model.save("models/test_model")  # Save a trained model
    vec_env = model.get_env()
    observation = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation)
        observation, rewards, done, _, info = env.step(action)
        env.render()
    env.close()


# Example call:
if __name__ == "__main__":
    # run_q(episodes=500, is_training=True, render_mode=None)
    # train_sb3_ppo()  # Train using StableBaseline3
    train_sb3_a2c()  # Train using StableBaseline3
    # predict_sb3_model()  # Test using StableBaseline3
