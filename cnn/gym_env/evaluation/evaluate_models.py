import os
import sys
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
import json
import gymnasium as gym

# Add the parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.maze_exploration_env import MazeExplorationEnv

def evaluate_model(model_path, model_class, difficulty_level=5, configuration=None, channels=4, episodes=1000):
    """Evaluate a trained model on specified difficulty level"""
    
    # Create environment
    env = gym.make("maze-exploration-v1", rows=configuration.get("rows"), columns=configuration.get("columns"),
                   maze_density=configuration.get("maze_density"), max_steps=configuration.get("max_steps"),
                   channels=channels, difficulty_level=difficulty_level)
    
    # Load model with the appropriate algorithm class
    model = model_class.load(model_path, env=env)
    
    total_rewards = []
    coverages = []
    episode_lengths = []  # Add this line
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0  # Add this line
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1  # Add this line
            
            if done:
                coverage = info.get("coverage", 0)
                coverages.append(coverage)
                total_rewards.append(total_reward)
                episode_lengths.append(steps)  # Add this line
                print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Coverage = {coverage:.2%}")
    
    # Calculate statistics
    successful_episodes = [length for length, coverage in zip(episode_lengths, coverages) if coverage >= 1.0]
    mean_success_length = np.mean(successful_episodes) if successful_episodes else 0
    std_success_length = np.std(successful_episodes) if successful_episodes else 0
    
    env.close()
    
    # Return only the specified metrics
    return {
        "mean_coverage": np.mean(coverages),
        "std_coverage": np.std(coverages),
        "success_rate": np.mean([c >= 1.0 for c in coverages]) * 100,
        "mean_success_episode_length": mean_success_length,
        "std_success_episode_length": std_success_length
    }


def main():
    configuration={"rows":10, "columns":10, "maze_density":0.85, "max_steps":250}
    
    # Mapping of model paths and their respective channels
    models_config = [
        {"path": "gym_env/models/final/DQN_SingleChannel_Final/level_1_final.zip", "channels": 1, "model_name": "DQN", "model_class": DQN},
        {"path": "gym_env/models/final/PPO_SingleChannel_Final/level_1_final.zip", "channels": 1, "model_name": "PPO", "model_class": PPO},
        {"path": "gym_env/models/final/A2C_SingleChannel_Final/level_1_final.zip", "channels": 1, "model_name": "A2C", "model_class": A2C},
        {"path": "gym_env/models/final/DQN_MultiChannel_Final/level_1_final.zip", "channels": 4, "model_name": "DQN", "model_class": DQN},
        {"path": "gym_env/models/final/PPO_MultiChannel_Final/level_1_final.zip", "channels": 4, "model_name": "PPO", "model_class": PPO},
        {"path": "gym_env/models/final/A2C_MultiChannel_Final/level_1_final.zip", "channels": 4, "model_name": "A2C", "model_class": A2C}
    ]


    results = {}
    for model_config in models_config:
        model_path = model_config["path"]
        channels = model_config["channels"]
        model_class = model_config["model_class"]
        model_name = model_config["model_name"]
        print(f"\nEvaluating model: {model_path}")
        try:
            result = evaluate_model(
                model_path=model_path, 
                model_class=model_class,
                difficulty_level=1, 
                configuration=configuration, 
                channels=channels, 
                episodes=1000
            )
            if result:
                results[model_path] = result
                #save as json to file
                output_dir = "gym_env/evaluation/final_prediction_logs/SingleChannel" if channels == 1 else "gym_env/evaluation/final_prediction_logs/MultiChannel"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{model_name}.json")
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=4)
                print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Failed to evaluate {model_path}: {e}")


if __name__ == "__main__":
    main()
