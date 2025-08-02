import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import datetime
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from neural_networks.simple_cnn import Simple1ChannelCNN
from neural_networks.advanced_cnn import CCNFeatureExtractor as CNN
from environment.maze_exploration_env import MazeExplorationEnv


def make_env(difficulty_level=1):
    """Create environment with specified difficulty level"""
    def _init():
        env = gym.make("maze-exploration-v1", 
                      difficulty_level=difficulty_level)
        return Monitor(env)
    return _init

def train_curriculum():
    """Train agent using curriculum learning across difficulty levels"""
    
    # Default grid map (can be None to use auto-generated 10x10)
    grid_map = None
    
    # Setup directories
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_model_dir = f"models/PPO_Curriculum_{time}"
    base_log_dir = f"logs/ppo_curriculum_{time}"
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Device selection
    device = "mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Curriculum configuration
    curriculum_config = {
        1: {"timesteps": 50000, "description": "No obstacles, fixed target"},
        2: {"timesteps": 75000, "description": "No obstacles, random target"},
        3: {"timesteps": 100000, "description": "10% obstacles, random target, random start"},
        4: {"timesteps": 125000, "description": "30% obstacles, near unexplored target"},
        5: {"timesteps": 150000, "description": "40% obstacles, near unexplored target"}
    }
    
    # Model setup
    policy_kwargs = dict(
        features_extractor_class=CNN,
        features_extractor_kwargs=dict(features_dim=256)
    )
    
    model = None
    
    # Train through curriculum levels
    for level, config in curriculum_config.items():
        print(f"\n{'='*60}")
        print(f"Training Level {level}: {config['description']}")
        print(f"{'='*60}")
        
        # Create environments for this level
        train_env = DummyVecEnv([make_env(difficulty_level=level)])
        eval_env = DummyVecEnv([make_env(difficulty_level=level)])
        
        # Setup logging for this level
        level_log_dir = f"{base_log_dir}/level_{level}"
        os.makedirs(level_log_dir, exist_ok=True)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{base_model_dir}/level_{level}",
            log_path=level_log_dir,
            eval_freq=5000,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )
        
        if model is None:
            # Create new model for first level
            model = PPO(
                "CnnPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                ent_coef=0.005,
                gamma=0.99,
                n_steps=128,
                tensorboard_log=level_log_dir,
                device=device
            )
        else:
            # Update environment for existing model
            model.set_env(train_env)
            # Update tensorboard log directory
            model.tensorboard_log = level_log_dir
        
        # Train for this level
        print(f"Training for {config['timesteps']} timesteps...")
        model.learn(
            total_timesteps=config['timesteps'],
            progress_bar=True,
            reset_num_timesteps=False,
            callback=eval_callback,
            tb_log_name=f"level_{level}"
        )
        
        # Save model after each level
        model.save(f"{base_model_dir}/level_{level}_final")
        print(f"Level {level} completed and saved!")
        
        # Clean up environments
        train_env.close()
        eval_env.close()
    
    print(f"\n{'='*60}")
    print("Curriculum training completed!")
    print(f"Models saved in: {base_model_dir}")
    print(f"Logs saved in: {base_log_dir}")
    print(f"{'='*60}")
    
    return model, base_model_dir

def evaluate_model(model_path, difficulty_level=5, episodes=10):
    """Evaluate a trained model on specified difficulty level"""
    
    # Create environment
    env = gym.make("maze-exploration-v1", 
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
    # final_model, model_dir = train_curriculum()
    model_dir = "models/PPO_Curriculum_20250802-180645"  
    
    # Evaluate the final model on the hardest level
    print("\nEvaluating final model on Level 5...")
    evaluate_model(f"{model_dir}/level_5_final.zip", difficulty_level=5)
    
    # Optional: Evaluate on all levels to see generalization
    print("\nEvaluating generalization across all levels...")
    for level in range(1, 6):
        print(f"\nLevel {level} evaluation:")
        evaluate_model(f"{model_dir}/level_5_final.zip", difficulty_level=level, episodes=5)
