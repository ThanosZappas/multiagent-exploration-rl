import os
import datetime

import gymnasium as gym
from stable_baselines3 import PPO
import policies

import maze_exploration_env


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
    model = PPO(policies.MlpPolicy, env, verbose=1, device=device, tensorboard_log=log_dir, ent_coef=0.01)

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 1000
    iters = 0
    for iters in range(500):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # train
        # model.save(f"{model_dir}/PPO_{time}/ppo_{TIMESTEPS * iters}")  # Save a trained model every TIMESTEPS
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
def predict_sb3_model():
    model = PPO.load("models/PPO_20250401-224617/ppo_426000.zip",
                     env=gym.make('maze-exploration-v0', render_mode='human'))
    # env = gym.make('maze-exploration-v0')
    env = model.get_env()
    observation = env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation)
        observation, rewards, done, info = env.step(action)
        env.render()


if __name__ == "__main__":
    train_sb3_ppo()  # Train using StableBaseline3
    # predict_sb3_model()  # Test using StableBaseline3
