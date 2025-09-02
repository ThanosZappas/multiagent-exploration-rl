from collections import deque
import gymnasium as gym
import numpy as np
import datetime
import torch as th
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
from neural_networks.simple_cnn import Simple1ChannelCNN
from neural_networks.advanced_cnn import CCNFeatureExtractor as CNN
from environment.maze_exploration_env import MazeExplorationEnv

# Setup directories
time = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
base_model_dir = f"models/PPO_{time}"
base_log_dir = f"logs/PPO_{time}"
os.makedirs(base_model_dir, exist_ok=True)
os.makedirs(base_log_dir, exist_ok=True)
CHANNELS = 4  # Change to 1 for single channel CNN

# Curriculum configuration
CURRICULUM_CONFIGURATION = {
    # 1: {"timesteps": 10000000, "rows": 6, "columns": 6, "maze_density" : 0.85, "max_steps": 30}
    # ,
    # 2: {"timesteps": 5000000, "rows": 8, "columns": 8, "maze_density" : 0.85, "max_steps": 100}
    # ,
    1: {"timesteps": 20000000, "rows": 10, "columns": 10, "maze_density" : 0.85, "max_steps": 250}
    # ,
    # 4: {"timesteps": 10000000, "rows": 14, "columns": 14, "maze_density" : 0.85, "max_steps": 450}
}

def make_env(difficulty_level=1):
    """Create environment with specified difficulty level"""
    def _init():
        env = gym.make("maze-exploration-v1",rows=CURRICULUM_CONFIGURATION[difficulty_level].get("rows"), columns=CURRICULUM_CONFIGURATION[difficulty_level].get("columns"),maze_density=CURRICULUM_CONFIGURATION[difficulty_level].get("maze_density"), max_steps=CURRICULUM_CONFIGURATION[difficulty_level].get("max_steps"), channels=CHANNELS, 
                      difficulty_level=difficulty_level)
        # Add 'coverage' to the info keywords to be logged by the Monitor
        return Monitor(env, info_keywords=("coverage",))
    return _init

def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: The initial learning rate.
    :param end_value: The final learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return end_value + (initial_value - end_value) * progress_remaining

    return func

class MetricsEvalCallback(BaseCallback):
    """
    A custom callback that logs training and evaluation metrics by directly
    accessing episode info.
    It logs:
    - rollout/success_rate (training)
    - rollout/mean_ep_coverage (training)
    - eval/success_rate (evaluation)
    - eval/mean_coverage (evaluation)
    """
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, verbose: int = 0):
        super(MetricsEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        # Use a deque to store recent training coverages for a rolling average
        self.recent_train_coverages = deque(maxlen=100)

    def _on_step(self) -> bool:
        # --- Log training metrics ---
        # Check if any episodes ended in the training environment
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # The Monitor wrapper puts final episode info in `info['episode']`
                info = self.locals["infos"][i]
                if "episode" in info and "coverage" in info["episode"]:
                    coverage = info["episode"]["coverage"]
                    self.recent_train_coverages.append(coverage)

        # Periodically log the rolling average of training metrics
        if self.n_calls % 1024 == 0 and self.recent_train_coverages:
            coverages = list(self.recent_train_coverages)
            success_rate = np.mean([c >= 1.0 for c in coverages])
            mean_coverage = np.mean(coverages)
            self.logger.record("rollout/success_rate", success_rate)
            self.logger.record("rollout/mean_ep_coverage", mean_coverage)

        # --- Log evaluation metrics ---
        # if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
        #     self._run_evaluation()

        return True

    def _run_evaluation(self) -> None:
        """
        Manually run evaluation and log metrics.
        """
        all_coverages = []
        all_rewards = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, infos = self.eval_env.step(action)
                done = terminated[0] or truncated[0]
                episode_reward += reward[0]
                
                if done:
                    # The info dict from the VecEnv contains the final info from the Monitor
                    final_info = infos[0]
                    if "episode" in final_info:
                        all_coverages.append(final_info["episode"]["coverage"])
            all_rewards.append(episode_reward)

        if all_coverages:
            mean_coverage = np.mean(all_coverages)
            success_rate = np.mean([c >= 1.0 for c in all_coverages])
            self.logger.record("eval/mean_coverage", mean_coverage)
            self.logger.record("eval/success_rate", success_rate)
        
        if all_rewards:
            self.logger.record("eval/mean_reward", np.mean(all_rewards))
        
        self.logger.dump(self.num_timesteps)


def transfer_weights(new_model, old_model_path, device):
    """
    Transfers the feature extractor weights from a saved model to a new model.
    This is useful when the observation space changes, as the policy/value networks
    will have different shapes, but the convolutional features are still valuable.
    """
    print(f"Loading feature extractor weights from: {old_model_path}")
    
    # Load parameters from the old model's zip file
    _, params, _ = load_from_zip_file(old_model_path, device=device)
    
    # Extract the state dictionaries
    old_state_dict = params['policy']
    new_state_dict = new_model.policy.state_dict()

    # Transfer weights only for the feature extractor
    for name, param in old_state_dict.items():
        if name.startswith('features_extractor'):
            if name in new_state_dict and new_state_dict[name].shape == param.shape:
                new_state_dict[name].copy_(param)
                print(f"  - Transferred layer: {name}")
            else:
                print(f"  - Skipped layer {name}: shape mismatch or not found.")
    
    # Load the modified state dict into the new model.
    # This will load the feature extractor weights and keep the new randomly
    # initialized weights for the policy and value networks.
    new_model.policy.load_state_dict(new_state_dict)
    print("Feature extractor weight transfer complete.")


def train_curriculum():
    """Train agent using curriculum learning across difficulty levels"""
    
    
    # Device selection
    device = "mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    
    # Model setup
    policy_kwargs = dict(
        features_extractor_class=CNN,
        features_extractor_kwargs=dict(features_dim=256)
    )
    
    model = None
    previous_model_path = None
    
    # Train through curriculum levels
    for level, configuration in CURRICULUM_CONFIGURATION.items():
        print(f"\n{'='*60}")
        print(f"Training Level {level}")
        print(f"{'='*60}")
        
        # Create environments for this level
        train_env = DummyVecEnv([make_env(difficulty_level=level)])
        eval_env = DummyVecEnv([make_env(difficulty_level=level)])
        # Setup logging for this level
        level_log_dir = f"{base_log_dir}/level_{level}"
        os.makedirs(level_log_dir, exist_ok=True)
        
        # Combined callback for training and evaluation metrics
        metrics_callback = MetricsEvalCallback(
            eval_env=eval_env,
            eval_freq=20000,
            n_eval_episodes=5,
        )

        # Use a separate EvalCallback just for saving the best model
        eval_callback_saver = EvalCallback(
            eval_env,
            best_model_save_path=f"{base_model_dir}/level_{level}",
            log_path=level_log_dir,
            eval_freq=20000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        # Combine callbacks
        callback = CallbackList([metrics_callback, eval_callback_saver])
        
        # For each level, we create a new model.
        # If a previous model exists, we transfer its learned weights.
        if CHANNELS == 1:
            lr_schedule = linear_schedule(0.0003, 0.0001)
            model = PPO(
                "CnnPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                ent_coef=0.001,
                gamma=0.99,
                n_steps=512,
                clip_range=0.2,
                learning_rate=lr_schedule,
                batch_size=256,
                n_epochs=4,
                tensorboard_log=base_log_dir,
                device=device
            )
        elif CHANNELS == 4:
            lr_schedule = linear_schedule(0.0002, 0.0001)
            model = PPO(
                    "CnnPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    ent_coef=0.0015,
                    gamma=0.99,
                    n_steps=512,
                    clip_range=0.2,
                    learning_rate=lr_schedule,
                    batch_size=512,
                    n_epochs=8,
                    tensorboard_log=base_log_dir,
                    device=device
            )

        # If we have a model from a previous level, transfer its weights
        if previous_model_path:
            transfer_weights(model, previous_model_path, device)
           
        # Train for this level
        tb_log_name = f"level_{level}"
        print(f"Training for {configuration['timesteps']} timesteps...")
        print(f"TensorBoard logs will be saved to: {base_log_dir}/{tb_log_name}")
        model.learn(
            total_timesteps=configuration['timesteps'],
            progress_bar=True,
            reset_num_timesteps=False,
            callback=callback,
            tb_log_name=tb_log_name
        )
        
        # Save model after each level and set path for the next iteration
        current_model_path = f"{base_model_dir}/level_{level}_final.zip"
        model.save(current_model_path)
        previous_model_path = current_model_path
        print(f"Level {level} completed and saved to {current_model_path}!")
        
        # Clean up environments
        train_env.close()
        eval_env.close()
    
    print(f"\n{'='*60}")
    print("Curriculum training completed!")
    print(f"Models saved in: {base_model_dir}")
    print(f"Logs saved in: {base_log_dir}")
    print(f"\nTo view TensorBoard logs, run:")
    print(f"tensorboard --logdir={base_log_dir}")
    print(f"Then open: http://localhost:6006")
    print(f"{'='*60}")
    
    return model, base_model_dir

def evaluate_model(model_path, difficulty_level=5, configuration=None, episodes=10):
    """Evaluate a trained model on specified difficulty level"""
    
    # Create environment
    env = gym.make("maze-exploration-v1", rows=configuration.get("rows"), columns=configuration.get("columns"),maze_density=configuration.get("maze_density"),max_steps=configuration.get("max_steps"),channels=CHANNELS, 
                      difficulty_level=difficulty_level)
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    total_rewards = []
    coverages = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()  # Render the environment
            
            if done:
                coverage = info.get("coverage", 0)
                coverages.append(coverage)
                total_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Coverage = {coverage:.2%}")
    
    env.close()
    
    print(f"\nEvaluation Results (Level {difficulty_level}):")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Coverage: {np.mean(coverages):.2%} ± {np.std(coverages):.2%}")
    print(f"Success Rate (100% coverage): {np.mean([c >= 1.0 for c in coverages]):.2%}")

if __name__ == "__main__":
    # Train using curriculum learning
    final_model, model_dir = train_curriculum()
    model_dir = base_model_dir  
    
    # Evaluate the final model on the hardest level
    # print("\nEvaluating final model on Level 5...")
    # evaluate_model(f"{model_dir}/level_5_final.zip", difficulty_level=train_curiculum)
    
    # Optional: Evaluate on all levels to see generalization
    print("\nEvaluating generalization across all levels...")
    for level, configuration in CURRICULUM_CONFIGURATION.items():
        print(f"\nLevel {level} evaluation:")
        model_to_eval_path = f"{model_dir}/level_{level}_final.zip"
        evaluate_model(model_to_eval_path, difficulty_level=level, configuration=configuration, episodes=5)