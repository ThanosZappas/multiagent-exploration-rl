import gymnasium as gym
import numpy as np
from environment.maze_exploration_env import MazeExplorationEnv

def test_curriculum_environment():
    """Test the curriculum environment across different difficulty levels"""
    
    print("Testing Curriculum-Based Maze Exploration Environment")
    print("=" * 60)
    
    for level in range(1, 6):
        print(f"\nTesting Difficulty Level {level}")
        print("-" * 30)
        
        # Create environment
        env = gym.make('maze-exploration-v1', difficulty_level=level)
        base_env = env.unwrapped
        
        # Get configuration info
        config = base_env.current_config
        print(f"Configuration: {config}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Agent position: {base_env.agent_position}")
        print(f"Target position: {base_env.target_position}")
        print(f"Lives: {base_env.current_lives}")
        
        # Take a few random steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode ended at step {step + 1}")
                print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
                break
        
        coverage = base_env.calculate_coverage()
        print(f"Coverage after {step + 1} steps: {coverage:.2%}")
        print(f"Total reward: {total_reward:.2f}")
        
        # Test obstacle generation
        obstacles = np.sum(base_env.grid_map == 1)
        total_cells = base_env.num_rows * base_env.num_cols
        obstacle_ratio = obstacles / total_cells
        print(f"Obstacles: {obstacles}/{total_cells} ({obstacle_ratio:.2%})")
        
        env.close()
    
    print("\n" + "=" * 60)
    print("All difficulty levels tested successfully!")

def visualize_level_comparison():
    """Visualize grid maps for different difficulty levels"""
    print("\nGrid Maps for Different Difficulty Levels:")
    print("=" * 60)
    
    for level in range(1, 6):
        print(f"\nLevel {level} Grid Map:")
        env = gym.make('maze-exploration-v1', difficulty_level=level)
        base_env = env.unwrapped
        obs, _ = env.reset()
        
        print("Grid Map (1=obstacle, 0=free):")
        for row in base_env.grid_map:
            print(" ".join([str(int(cell)) for cell in row]))
        
        config = base_env.current_config
        print(f"Config: Maze Density={config['maze_density']}, "
              f"Target Mode={config['target_mode']}, "
              f"Random Start={config['random_start']}")
        
        env.close()

if __name__ == "__main__":
    # Test basic functionality
    test_curriculum_environment()
    
    # Visualize different levels
    visualize_level_comparison()
