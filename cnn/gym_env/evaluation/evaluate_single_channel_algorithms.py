import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C
import json
from datetime import datetime

# Add the parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.maze_exploration_env import MazeExplorationEnv

def evaluate_model(model, env, num_episodes=100, success_threshold=1.0):
    """
    Evaluate a model over a number of episodes and return the results
    Args:
        model: The RL model to evaluate
        env: The environment to evaluate in
        num_episodes: Number of episodes to evaluate
        success_threshold: The exploration percentage threshold to consider an episode successful
    """
    episode_rewards = []
    episode_lengths = []
    coverages = []
    successes = []
    successful_episode_lengths = []  # New list for successful episode lengths only
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        coverage = info.get('coverage', 0)
        is_success = coverage >= success_threshold  # Convert to percentage
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        coverages.append(coverage)
        successes.append(is_success)
        
        if is_success:
            successful_episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1} episodes")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'coverages': coverages,
        'successes': successes,
        'successful_lengths': successful_episode_lengths
    }

def main():
    # Create output directory if it doesn't exist
    output_dir = "gym_env/evaluation/final_prediction_logs/SingleChannel"
    os.makedirs(output_dir, exist_ok=True)
    
    # Models to evaluate
    models_info = [
        {
            'name': 'DQN',
            'path': 'gym_env/models/final/DQN_SingleChannel_Final/level_1_final.zip',
            'class': DQN
        },
        {
            'name': 'PPO',
            'path': 'gym_env/models/final/PPO_SingleChannel_Final/level_1_final.zip',
            'class': PPO
        },
        {
            'name': 'A2C',
            'path': 'gym_env/models/final/A2C_SingleChannel_Final/level_1_final.zip',
            'class': A2C
        }
    ]
    
    # Evaluation parameters
    env_params = {
        'rows': 10,
        'columns': 10,
        'maze_density': 0.85,
        'max_steps': 250,
        'channels': 1,  # Single channel
        'difficulty_level': 1
    }
    
    # Create environment
    env = MazeExplorationEnv(**env_params)
    
    # Evaluate each model
    for model_info in models_info:
        print(f"\nEvaluating {model_info['name']}...")
        
        try:
            # Load the model
            model = model_info['class'].load(model_info['path'])
            
            # Run evaluation
            results = evaluate_model(model, env)
            
            # Calculate key metrics
            mean_coverage = np.mean(results['coverages'])
            std_coverage = np.std(results['coverages'])
            success_rate = np.mean(results['successes']) * 100  # Convert to percentage
            
            # Calculate episode length statistics only for successful episodes
            if results['successful_lengths']:
                mean_success_episode_length = np.mean(results['successful_lengths'])
                std_success_episode_length = np.std(results['successful_lengths'])
            else:
                mean_success_episode_length = 0
                std_success_episode_length = 0
            
            # Save results
            output_file = os.path.join(output_dir, f"{model_info['name']}.json")
            
            # Prepare results for saving
            save_data = {
                'model_name': model_info['name'],
                'environment_params': env_params,
                'num_episodes': 100,
                # Key metrics for thesis
                'mean_coverage': mean_coverage,
                'std_coverage': std_coverage,
                'success_rate': success_rate,
                'mean_success_episode_length': mean_success_episode_length,
                'std_success_episode_length': std_success_episode_length,
                # Raw data for additional analysis
                'episode_lengths': results['lengths'],
                'coverages': results['coverages'],
                'successful_lengths': results['successful_lengths']
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=4)
                
            print(f"Results saved to {output_file}")
            print(f"Mean Coverage: {mean_coverage:.2f}% ± {std_coverage:.2f}%")
            print(f"Success Rate: {success_rate:.2f}%")
            if results['successful_lengths']:
                print(f"Mean Success Episode Length: {mean_success_episode_length:.2f} ± {std_success_episode_length:.2f} steps")
            
        except Exception as e:
            print(f"Error evaluating {model_info['name']}: {str(e)}")

if __name__ == "__main__":
    main()