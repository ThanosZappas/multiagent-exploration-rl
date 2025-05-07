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
import random
from enum import IntEnum
import matplotlib.pyplot as plt
# import grid_map_exploration_agent as agent
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='maze-exploration-v1',  # call it whatever you want
    entry_point='maze_exploration_env:MazeExplorationEnv', # module_name:class_name
)

class MazeExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_map, range_gs = False, max_steps=150):
        super(MazeExplorationEnv, self).__init__()

        self.grid_map = np.array(grid_map)
        self.num_rows, self.num_cols = self.grid_map.shape
        self.agent_position = None
        self.target_position = None
        self.range_gs = range_gs
        self.action_space = spaces.Discrete(8)

        self.agent_view = self._reset_agent_view()
        
        self.observation_space = spaces.Box(low=0, high=4, shape=(1, self.num_rows, self.num_cols), dtype=np.uint8)

        self.fig = None
        self.ax = None
        # self.last_euclidean_distance=1
        self.steps = 0
        self.max_steps = max_steps

        
    def _calculate_observation(self):
        
        self._update_agent_view()
        
        # If maze is fully explored
        if np.all(self.agent_view != 0):  # Check if all cells are explored
            self.target_position = None
            return self._transform_obs(self.agent_view) 
            
        self.agent_view[self.target_position] = MazeElements.TARGET  # Mark target position as 3
            
        return self._transform_obs(self.agent_view)
     

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is None or 'start_position' not in options:
            self.agent_position = (1,2)
            # self.agent_position = self._random_position()
        else:
            self.agent_position = options['start_position']
        
        self.agent_view = self._reset_agent_view()
        self.target_position = self._calculate_new_target()
        

        self.steps = 0

        if self.range_gs:
            _, width = self.grid_map.shape
            self.agent_position = (5,random.randint(5,width-5))
        observation = self._calculate_observation()
        info ={}
        # info = self._get_info()
        return observation, info

    def step(self, action):
        self.steps += 1
        reward = -0.1  # Initialize reward
        terminated = False
        truncated = False

        #  Check max steps
        if self.steps >= self.max_steps:
            truncated = True
            reward = -1.0
            return self._calculate_observation(), reward, terminated, truncated, {}

        # Convert action and calculate new position
        action_row, action_column = self._action_to_direction(action)
        new_row = self.agent_position[0] + action_row
        new_column = self.agent_position[1] + action_column

        # Handle collision
        if self.grid_map[new_row, new_column] == 1:  
            terminated = True
            reward = -1.0
            return self._calculate_observation(), reward, terminated, truncated, {}

        # Move agent if no collision
        self.agent_position = (new_row, new_column)
        
        # Update view 
        self._update_agent_view()
        # Calculate coverage
        coverage = self.calculate_coverage()
        # Check if the agent has fully explored the maze
        if coverage==1: #np.all(self.agent_view) != 0:
            terminated = True
            # print(self.agent_view)
            # print("Coverage: ", coverage)
            # print("Maze fully explored at step: ",self.steps,"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            reward = 100 - self.steps*0.25 #self.num_rows * self.num_cols 
        else: # Check target
            row, col = self.target_position
            if self.agent_view[row, col] in [MazeElements.EXPLORED, MazeElements.OBSTACLE]:
                reward += 1.0
                self.target_position = self._calculate_new_target()
            reward += 0.25 * coverage
        return self._calculate_observation(), reward, terminated, truncated, {}

    # Gym required function to render environment
    def render(self):
        # Render to the console.
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                if np.array_equal((row, column),(self.agent_position)):
                    print(f'{str(GridTile.AGENT):>3}', end=' ')
                elif self.grid_map[row, column] == 1:
                    print(f'{str(GridTile.OBSTACLE):>3}', end=' ')
                elif np.array_equal((row, column), (self.target_position)): #TODO: Fix this not showing target
                    print(f'{str(GridTile.TARGET):>3}', end=' ')
                else:
                    print(f'{str(GridTile.FLOOR):>3}', end=' ')
            print()
        print()
        
        # self.fig_render(self)


    def fig_render(self):
        if self.fig is None or self.ax is None:  # Only create new fig, ax if they don't already exist
            self.fig, self.ax = plt.subplots()

        self.ax.clear()  # Clear existing plot

        # Remainder of your rendering code...
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if np.array_equal((i, j), self.agent_position):
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='red'))
                elif np.array_equal((i, j), self.target_position):
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))
                elif self.grid_map[i, j] == 1:
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
                else:
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='white', fill=False))

        plt.xlim([0, self.num_cols])
        plt.ylim([0, self.num_rows])
        plt.draw()
        plt.pause(0.0001) 
        plt.show(block=False)  # Add this line to update the window automatically

    def _random_position(self):
        valid_positions = np.argwhere(self.grid_map == 0)
        index = np.random.choice(len(valid_positions))
        return valid_positions[index]

    def _calculate_new_target(self): #Closest unexplored cell
        # Get the agent's current position
        agent_row, agent_column = self.agent_position
        
        # Find the closest valid and not explored position in the maze
        closest_target = None
        min_distance = float('inf')
        
        # Only consider inner cells (exclude outer edges)
        for row in range(1, self.num_rows - 1):
            for column in range(1, self.num_cols - 1):
                # Check if the position is unexplored
                if self.agent_view[row, column] == MazeElements.UNEXPLORED:
                    # Calculate the Euclidean distance to the agent's position
                    distance = ((row - agent_row)**2 + (column - agent_column)**2)
                    # Check if the distance is less than the minimum distance found so far
                    if distance < min_distance:
                        min_distance = distance
                        closest_target = (row, column)
                        
        return closest_target
    
    def _get_info(self):
        info = {"agent_position": self.agent_position, "target_position": self.target_position}
        return info
    

    
    def calculate_coverage(self):
        total_cells_except_walls = (self.num_rows - 2) * (self.num_cols - 2)
        walls = self.num_cols * self.num_rows - total_cells_except_walls
        explored_cells = np.sum(self.agent_view != MazeElements.UNEXPLORED) - walls
        coverage = explored_cells / total_cells_except_walls
        return coverage

    def _transform_obs(self,obs):
        observation = np.expand_dims(obs.astype(np.uint8), axis=0)
        return observation

    def _reset_agent_view(self):
        num_rows, num_cols = self.grid_map.shape
        agent_view = np.zeros((num_rows, num_cols), dtype=np.int32)          
        # Set walls to 4
        for row in range(num_rows):
            for col in range(num_cols):
                if row == 0 or col == 0 or row == num_rows - 1 or col == num_cols - 1:
                    agent_view[row, col] = MazeElements.OBSTACLE
        return agent_view
    
   
    # Update the agent's view based on the current position
    def _update_agent_view(self):
        agent_row, agent_column = self.agent_position
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = agent_row + i
                col = agent_column + j
                if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
                    if self.grid_map[row, col] == 0:
                        self.agent_view[row, col] = MazeElements.EXPLORED  # Mark explorable cells as 1
                    else:
                        self.agent_view[row, col] = MazeElements.OBSTACLE  # Mark obstacles as 4
        self.agent_view[agent_row, agent_column] = MazeElements.AGENT   # Mark the agent's position as 2
        

    def _action_to_direction(self, action):
        action_map = {
            0: (-1, -1),  # Top Left
            1: (-1, 0),   # Up
            2: (-1, 1),   # Top Right
            3: (0, -1),   # Left
            4: (0, 1),    # Right
            5: (1, -1),   # Bottom Left
            6: (1, 0),    # Down
            7: (1, 1)     # Bottom Right
        }
        
        if isinstance(action, np.ndarray):
            action = action.item()
        return action_map[action]

class MazeElements(IntEnum):
    UNEXPLORED = 0
    EXPLORED = 1
    AGENT = 2
    TARGET = 3
    OBSTACLE = 4

# The maze is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(IntEnum):
    FLOOR = 0
    AGENT = 2
    TARGET = 3
    OBSTACLE = 4

    # Return the proper string representation of the tile
    # This is used for printing the grid to the console.
    def __str__(self):
        string_array = ['_','', 'X', 'T', '#']
        return string_array[self.value]

if __name__ == "__main__":
    grid_map = [
    [1, 1, 1, 1,1,1,1,1,1,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 1,0,0,0,0,0,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 1,0,0,1,1,1,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 1,0,0,0,0,0,1],
    [1, 1, 0, 0,0,0,0,0,0,1],
    [1, 0, 0, 0,0,0,1,0,0,1],
    [1, 1, 1, 1,1,1,1,1,1,1],
    ]

    env = gym.make('maze-exploration-v1', grid_map=grid_map, range_gs = False)
    observation, _ = env.reset()
    for i in range(2):
        rand_action = env.action_space.sample()
        observation, reward, terminated, _, _ = env.step(rand_action)
        print(f"Action: Observation:\n {observation}\n")
        print(f"Action: {MazeExplorationEnv._action_to_direction(env,action=rand_action)}, \nReward: {reward}\n")
        env.render()
        if terminated:
            observation, _ = env.reset()
