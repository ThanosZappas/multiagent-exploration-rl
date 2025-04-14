import os
import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import maze_exploration_env

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
    ACTUAL_TIMESTEPS = int(TIMESTEPS/2)
    for iters in range(50):
        model.learn(total_timesteps=ACTUAL_TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
        
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
    
    while not (done or truncated):
        action, _states = model.predict(observation)
        action = int(action.item())  # Add this line to convert numpy array to integer
        observation, rewards, done, truncated, info = env.step(action)
        print("agent view: \n", observation.get('grid'))    
        env.render()
    
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
