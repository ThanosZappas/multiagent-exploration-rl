import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN,PPO
# from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
import maze_exploration_env 
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from simple_cnn import Simple1ChannelCNN
from advanced_cnn import CCNFeatureExtractor
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

CHANNELS = 3  # Number of channels in the input (1 for grayscale images)


# Setup directories
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = f"models/DQN_{time}"
log_dir = "logs/DQN_maze_exploration"
log_dir = f"{log_dir}/{time}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# Device selection
device = "mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu"

#Create the environments
train_env = gym.make("maze-exploration-v1", grid_map=grid_map, range_gs = False, max_steps=500, channels= CHANNELS)
eval_env = gym.make("maze-exploration-v1", grid_map=grid_map, range_gs = False, max_steps=500, channels= CHANNELS)


class CustomEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        # Access evaluation episode info
        if self.locals.get("dones") is not None:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done and info.get("is_success", False):  # or use your own flag
                    print(f"Eval: Maze fully explored at step {info.get('steps', 'N/A')}")
        # # Print evaluation summary at eval_freq
        # if self.n_calls % self.eval_freq == 0:
        #     print(f"Eval: Step {self.num_timesteps}, mean_reward={self.last_mean_reward:.3f}")
        return result
    
# Evaluation callback
eval_callback = CustomEvalCallback(
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
advanced_policy_kwargs = dict(
    features_extractor_class=CCNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256)
)

# model = DQN(
#         "CnnPolicy",
#         train_env,
#         policy_kwargs=advanced_policy_kwargs,
#         verbose=1,
#         # ent_coef=0.005,
#         # gamma=0.99,
#         # n_steps=128,
#         tensorboard_log=log_dir,
#         device=device
#         )

# Model with parameters from the paper
model = DQN(
    "CnnPolicy",
    train_env,
    policy_kwargs=advanced_policy_kwargs,
    verbose=1,
    gamma=0.9,
    learning_rate=0.0001,
    batch_size=32,  # Add batch size parameter
    buffer_size=100000,  # Reasonable buffer size for DQN
    learning_starts=1000,  # Start learning after gathering some experience
    tensorboard_log=log_dir,
    device=device
)

# Train the model
model.learn(total_timesteps=500000, progress_bar=True,reset_num_timesteps=False,callback=eval_callback)  # Perform a training iteration

# Load the model
# model = DQN.load("models/DQN_20250507-152612/best_model.zip", env=train_env)

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
