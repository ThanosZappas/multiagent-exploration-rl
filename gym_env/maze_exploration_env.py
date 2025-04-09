import re
from turtle import distance, update
from arrow import get
import gymnasium as gym
from gymnasium import make, spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from matplotlib.pyplot import grid

import maze_exploration_agent as agent
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='maze-exploration-v0',  # call it whatever you want
    entry_point='maze_exploration_env:MazeExplorationEnv',  # module_name:class_name
)


class MazeExplorationEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, grid_rows=10, grid_columns=10, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_columns = grid_columns
        self.agent_view = np.zeros((self.grid_rows, self.grid_columns), dtype=int)
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

        # Update observation space to include both the grid view and the distance
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(grid_rows, grid_columns), dtype=np.int32),
            'distance': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
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
        self.maze_exploration.reset(seed=seed)
        self.last_euclidean_distance=1
        self.agent_view = np.zeros((self.grid_rows, self.grid_columns), dtype=int)
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
                    self.agent_view[row, col] = 1  # Mark cells AROUND the agent as seen
        return self.agent_view
        
    def step(self, action):
        reward, terminated, truncated = self.maze_exploration.perform_action(action)
        
           
        # Update the agent's view
        self._update_agent_view()
        # Check if the agent has explored the maze
        if np.all(self.agent_view == 1):
            terminated = True
            reward += 10
            observation = {}
        else:
            if self.target_position is not None:
                if self.agent_view[self.target_position] == 1:
                    # The agent has found the target
                    self.target_position = self._calculate_new_target()
                    reward += 1

                observation = self._calculate_observation()

                # else:
                #     # The agent has not found the target
                #     # Add reward for moving closer to the target
                #     current_distance = observation['distance'][0]
                #     reward += (1 - current_distance)  # Reward is higher when distance is smaller
                #     self.last_euclidean_distance = current_distance
        if self.render_mode == 'human':
            print(action)
            self.render()

    
        return observation, reward, terminated, truncated, self._get_info()

    def _calculate_observation(self):
        # Update the agent's position
        self.agent_position = self.maze_exploration.agent_position
        agent_row, agent_column = self.agent_position
        # Create an observation of the maze with the agent's view
        self._update_agent_view()
        # Check if the agent has explored the maze
        if np.all(self.agent_view == 1):
            self.target_position = None
            return {'grid' : self.agent_view, 'distance': np.array([0], dtype=np.float32)}
        
        # Calculate Euclidean distance to the target position
        if self.target_position is None:
            self.target_position = self._calculate_new_target()
            print("Target position is None")
        target_row, target_column = self.target_position # type: ignore
        euclidean_distance = np.linalg.norm([target_row - agent_row, target_column - agent_column])
        euclidean_distance /= self.max_distance  # Scale distance to [0, 1]

        return {'grid' : self.agent_view.astype(np.int32), 'distance': np.array([euclidean_distance], dtype=np.float32)}
    
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
