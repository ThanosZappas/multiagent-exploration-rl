import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
import maze_exploration_env 
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from simple_cnn import Simple1ChannelCNN
import datetime
import os

grid_map = [
    [1, 1, 1, 1,1,1,1,1,1,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 1,0,0,0,0,0,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 1,0,0,1,1,1,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 1,0,0,0,0,0,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 0,0,0,1,0,0,1],
    [1, 1, 1, 1,1,1,1,1,1,1],
    ]


# Setup directories
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = f"models/PPO_{time}"
log_dir = "logs/ppo_maze_exploration"
log_dir = f"{log_dir}/{time}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# Device selection
device = "mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu"

#Create the environments
train_env = gym.make("maze-exploration-v1", grid_map=grid_map, range_gs = False)
eval_env = gym.make("maze-exploration-v1", grid_map=grid_map, range_gs = False)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    # best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=5000,             
    n_eval_episodes=10,
    deterministic=True,        
    render=False
)

# Model setup
policy_kwargs = dict(
    features_extractor_class=Simple1ChannelCNN,
    features_extractor_kwargs=dict(features_dim=64)
)

model = PPO(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        # ent_coef=0.005,
        # gamma=0.99,
        # n_steps=128,
        tensorboard_log=log_dir,
        device=device
        )

# Train the model
model.learn(total_timesteps=300000, progress_bar=True,reset_num_timesteps=False,callback=eval_callback)  # Perform a training iteration

# Load the model
# model = PPO.load("models/PPO_20250507-152612/best_model.zip", env=train_env)

# # Run a single evaluation episode
# obs, _ = train_env.reset()
# done = False
# total_reward = 0

# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = train_env.step(action)
#     done = terminated or truncated
#     total_reward += reward
#     train_env.render()

# print(f"Episode finished with reward: {total_reward}")
# train_env.close()
    