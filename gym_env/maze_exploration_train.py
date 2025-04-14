import os
import datetime

import gymnasium as gym
import numpy as np
from regex import P
from sqlalchemy import TIME
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import maze_exploration_env
ACTION_MAP = {
    0: 'UP_LEFT',  # Move diagonally up and left
    1: 'UP',       # Move up
    2: 'UP_RIGHT', # Move diagonally up and right
    3: 'LEFT',     # Move left
    4: 'STAY',     # Stay in place
    5: 'RIGHT',    # Move right
    6: 'DOWN_LEFT', # Move diagonally down and left
    7: 'DOWN',     # Move down
    8: 'DOWN_RIGHT' # Move diagonally down and right
}

# Train using StableBaseline3 with PPO
def train_sb3_ppo():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = 'cpu'
    
    # Create the environment
    env = gym.make('maze-exploration-v0')
    
    # Setup for logging
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_dir}/{time}"
    
    # Create the model with default MLP policy
    # Note: SB3 will automatically handle dictionary observations with its default policy
    model = PPO("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)

    # Training loop
    TIMESTEPS = 10000
    for iters in range(50):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        
        # # Save model periodically
        # if iters % 10 == 0:
        #     save_path = f"{model_dir}/PPO_{time}"
        #     os.makedirs(save_path, exist_ok=True)
        #     model.save(f"{save_path}/ppo_{TIMESTEPS * (iters+1)}")

    # Final save
    # model.save(f"{model_dir}/PPO_{time}/final_model")
    
    # Test the trained model
    observation, info = env.reset()  # Use the existing env instance
    done = False
    truncated = False
    step = 1
    while not (done or truncated):
        action, _states = model.predict(observation)
        action = int(action.item())  # Add this line to convert numpy array to integer
        observation, rewards, done, truncated, info = env.step(action)
        print('\n------------------------------------------------------------------------------------------------------------------------------\n')
        print(f"Step: {step}, Action: {ACTION_MAP.get(action)}, Reward: {rewards} \n")
        print("Game Status:\n")
        env.render()
        np.set_printoptions(formatter={'int': lambda x: f'{x:2d}'})
        print("Agent`s View: \n")
        for row in observation.get('grid'):
            print(' '.join(f'{x:>3}' for x in row))
        step += 1
        if done:
            print("Episode finished")
            break
        if truncated:
            print("Episode truncated")
            break
    env.close()

def predict_sb3_model(model_path):
    # Load the model
    env = gym.make('maze-exploration-v0', render_mode='human')
    model = PPO.load(model_path, env=env)
    
    # Test the model
    observation, _ = env.reset()  # Note the unpacking for the info
    done = False
    truncated = False
    
    while not (done or truncated):
        action, _states = model.predict(observation)
        observation, rewards, done, truncated, info = env.step(action)
        env.render()
    
    env.close()

if __name__ == "__main__":
    train_sb3_ppo()  # Train using StableBaseline3
    
    # Uncomment to test a trained model
    # model_path = "models/PPO_20250409-000000/final_model.zip"  # Update with your model path
    # predict_sb3_model(model_path)
