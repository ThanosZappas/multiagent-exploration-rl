from math import e
import re
from turtle import distance, update
# from arrow import get
import gymnasium as gym
from gymnasium import make, spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from matplotlib.pyplot import grid
from torch import rand

import maze_exploration_agent as agent
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='maze-exploration-v0',  # call it whatever you want
    entry_point='maze_exploration_env:MazeExplorationEnv', # module_name:class_name
)


class MazeExplorationEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 1}
    def __init__(self, grid_rows=10, grid_columns=10, render_mode=None):
        self.random_mazes = False # Set to True for random maze generation
        self.grid_rows = grid_rows
        self.grid_columns = grid_columns
        self.agent_view = np.zeros((self.grid_rows, self.grid_columns), dtype=np.int32)
        self.render_mode = render_mode
        self.max_distance = np.sqrt(np.square(self.grid_rows) + np.square(self.grid_columns))  # Maximum possible Euclidean distance
        # Initialize the MazeExploration problem
        self.maze_exploration = agent.MazeExploration(grid_rows=grid_rows, grid_columns=grid_columns,
                                                      obstacle_probability=0.85)
        self.maze_exploration.reset() 
        self.maze = self.maze_exploration.maze
        self.agent_position = self.maze_exploration.agent_position
        self.target_position = self._calculate_new_target()
        self.last_euclidean_distance = 1
        self.action_space = spaces.Discrete(len(self.maze_exploration.ACTION_MAP))

        # Update observation space to include both the grid view and the distances from surrounding cells
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=-1, high=1, shape=(grid_rows, grid_columns), dtype=np.int32),
            'distances': spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)  # 9 cells (3x3 grid including agent position)
        })
      
    def _calculate_new_target(self):
        # Get the agent's current position
        agent_row, agent_column = self.agent_position
        
        # Find the closest valid and not explored position in the maze
        closest_target = None
        min_distance = float('inf')
        if self.maze is None:
            raise ValueError("Maze is not properly initialized or has an unexpected shape.")
        
        for row in range(self.grid_rows):
            for column in range(self.grid_columns):
                # Check if the position is valid and not explored
                if self.maze[row, column] == 0 and self.agent_view[row, column] == 0:
                    # Calculate the Euclidean distance to the agent's position
                    distance = ((row - agent_row)**2 + (column - agent_column)**2)
                    # Check if the distance is less than the minimum distance found so far
                    if distance < min_distance:
                        min_distance = distance
                        closest_target = (row, column)
        return closest_target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.maze_exploration.reset(seed=seed, random_mazes=self.random_mazes)
        self.maze = self.maze_exploration.maze
        self.last_euclidean_distance=1
        self.agent_view = np.zeros((self.grid_rows, self.grid_columns), dtype=np.int32)
        self.agent_position = self.maze_exploration.agent_position
        observation = self._calculate_observation()
        self.target_position = self._calculate_new_target()
        info = self._get_info()
        # if self.render_mode == 'human':
        #     self.render()
        return observation, info

    def _get_info(self):
        info = {"agent_position": self.agent_position, "target_position": self.target_position}
        return info
    

    # Update the agent's view based on the current position
    def _update_agent_view(self):
        agent_row, agent_column = self.agent_position
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = agent_row + i
                col = agent_column + j
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_columns:
                    if self.maze[row, col] == 0:
                        self.agent_view[row, col] = 1  # Mark explorable cells as 1
                    else:
                        self.agent_view[row, col] = -1  # Mark obstacles as -1

        
    def step(self, action):
        # Get initial observation before action
        observation = self._calculate_observation()
        
        # Perform action and get basic reward
        reward, terminated, truncated = self.maze_exploration.perform_action(action)
        
        # Update agent's view if not terminated
        if not (terminated or truncated):
            self._update_agent_view()
            
            # Check if maze is fully explored
            if np.all(abs(self.agent_view) == 1):
                terminated = True
                reward += self.grid_rows  # Big reward for completing exploration
            
            elif self.target_position is not None:
                # Get updated observation after movement
                observation = self._calculate_observation()
                current_distance = observation['distances'][4]
                
                # Calculate distance-based reward
                if current_distance < self.last_euclidean_distance:
                    distance_reward = 0.5 * (self.last_euclidean_distance - current_distance)
                    reward += distance_reward
                else:
                    distance_penalty = 0.2 * (current_distance - self.last_euclidean_distance)
                    reward -= distance_penalty
                
                self.last_euclidean_distance = current_distance
                
                # Check if reached target
                if self.agent_view[self.target_position] == 1:
                    reward += 1
                    self.target_position = self._calculate_new_target()
                

        return observation, reward, terminated, truncated, self._get_info()

    def _calculate_observation(self):
        self.agent_position = self.maze_exploration.agent_position
        agent_row, agent_column = self.agent_position
        
        # Update cumulative agent view (using -1 for obstacles, 1 for explored spaces)
        self._update_agent_view()
        
        # If maze is fully explored
        if np.all(np.abs(self.agent_view) == 1):  # Check both 1 and -1
            self.target_position = None
            return {
                'grid': self.agent_view.astype(np.int32),
                'distances': np.zeros(9, dtype=np.float32)
            }

        # Calculate distances from surrounding cells to target
        if self.target_position is None:
            self.target_position = self._calculate_new_target()

        # If no target position is found, return zeros for distances
        # This can happen if the maze is fully explored
        if self.target_position is None:
            return {
                'grid': self.agent_view.astype(np.int32),
                'distances': np.zeros(9, dtype=np.float32)
            }
        
        target_row, target_column = self.target_position
        surrounding_distances = []
        
        # Calculate distances for 3x3 grid around agent (including agent's position)
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = agent_row + i
                col = agent_column + j
                
                if (0 <= row < self.grid_rows and 
                    0 <= col < self.grid_columns and 
                    self.maze[row, col] == 0):  # If valid and not obstacle
                    # Calculate Euclidean distance from this cell to target
                    dist = np.linalg.norm([target_row - row, target_column - col])
                    dist /= self.max_distance  # Normalize distance
                else:
                    dist = 1.0  # Max distance for obstacles or out-of-bounds
                
                surrounding_distances.append(dist)

        return {
            'grid': self.agent_view.astype(np.int32),
            'distances': np.array(surrounding_distances, dtype=np.float32)
        }
    
    # Gym required function to render environment
    def render(self):
        self.maze_exploration.render()


  
# For unit testing
if __name__ == "__main__":
    env = gym.make('maze-exploration-v0', render_mode='human')
    observation, _ = env.reset()
    for i in range(100):
        rand_action = env.action_space.sample()
        observation, reward, terminated, _, _ = env.step(rand_action)
        if terminated:
            observation, _ = env.reset()
