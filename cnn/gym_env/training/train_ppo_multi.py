import os
import datetime
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from simple_cnn import Simple1ChannelCNN
import maze_exploration_env


def make_env(grid_map, range_gs = False):
    def _init():
        env = gym.make("maze-exploration-v1",grid_map=grid_map, range_gs = range_gs)
        return Monitor(env)
    return _init


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


def train_sb3_ppo():
    # Setup directories
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f"models/PPO_{time}"
    log_dir = "logs/ppo_maze_exploration"
    log_dir = f"{log_dir}/{time}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Device selection
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # Create training environment (10 parallel agents)
    train_env = SubprocVecEnv([make_env(grid_map=grid_map) for _ in range(10)])

    # Create evaluation environment (single agent with rendering)
    eval_env = DummyVecEnv([lambda: Monitor(gym.make("maze-exploration-v1",grid_map=grid_map, range_gs = False))])

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq= max(10000 // 10, 1),             # Evaluate every 10k steps
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
        ent_coef=0.05,
        gamma=0.995,
        n_steps=128,
        tensorboard_log=log_dir,
        device=device
    )

    # Continuous training (extend timesteps as needed)
    model.learn(
        total_timesteps=int(1e6),  # 1 million steps; increase for continuous training
        callback=eval_callback,
        reset_num_timesteps=False,
        progress_bar=True
    )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train_sb3_ppo()
