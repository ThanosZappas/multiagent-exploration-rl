import random
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.maze_generator import create_maze
import time

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='maze-exploration-v1',  # call it whatever you want
    entry_point='environment.maze_exploration_env:MazeExplorationEnv', # module_name:class_name
)

class MazeExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, rows = 10, columns = 10, maze_density=0.85, max_steps=250, channels=4, difficulty_level=1):
        super(MazeExplorationEnv, self).__init__()

        self.difficulty_config = {
            1: { 
                "target_mode": "near_unexplored",      
                "agent_start_mode": "fixed"            
            },
            2: {
                "target_mode": "near_unexplored",      
                "agent_start_mode": "fixed'"
            },
            3: {
                "target_mode": "near_unexplored",              
                "agent_start_mode": "fixed'"
            },
            4: {
                "target_mode": "near_unexplored",              
                "agent_start_mode": "fixed'"
            },
            5: {
                "target_mode": "near_unexplored",          
                "agent_start_mode": "fixed'"
            },
        }

        
        self.difficulty_level = difficulty_level
        self.current_config = self.difficulty_config[difficulty_level]
        
        self.grid_map = create_maze(rows, columns, maze_density)
        self.maze_density = maze_density
        self.num_rows, self.num_cols = rows, columns
        self.max_steps = max_steps
        self.channels = channels

      
        # Initialize positions and state
        self.set_agent_position()
        self.coverage_grid = np.zeros_like(self.grid_map)  # Track explored areas
        self.target_position = self._calculate_new_target()
        self.action_space = spaces.Discrete(4)  # Only cardinal directions: up, right, down, left
        self.agent_lives = 2  # Number of lives for obstacle collisions
        self.agent_view = self._reset_agent_view()
        self.current_lives = self.agent_lives
        self.steps = 0

        self.total_free_cells = np.sum(self.grid_map == 0)
        self.explored_free_cells = np.sum((self.coverage_grid == 1) & (self.grid_map == 0))
        self.coverage = self.calculate_coverage()
        
        self.coverage_80_percent_reached = False
        self.coverage_90_percent_reached = False



        # Set observation space based on channels
        if self.channels == 1:
            self.observation_space = spaces.Box(
                low=0, high=4, shape=(1, self.num_rows, self.num_cols), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(4, self.num_rows, self.num_cols), dtype=np.float32
            )

    def _calculate_observation(self):
        self._update_agent_view()
        if self.channels == 1:
            obs = np.expand_dims(self.agent_view.astype(np.float32), axis=0)
        else:
            # Channel 1: Explored mask (1 if explored, else 0)
            explored_mask = (self.agent_view == MazeElements.EXPLORED).astype(np.float32)
            
            # Channel 2: Discovered obstacles (1 if discovered obstacle, else 0)
            obstacle_mask = (self.agent_view == MazeElements.OBSTACLE).astype(np.float32)
            
            # Channel 3: Agent position (1 if agent, else 0)
            agent_mask = np.zeros_like(explored_mask, dtype=np.float32)
            agent_row, agent_col = self.agent_position
            agent_mask[agent_row, agent_col] = 1
            
            # Channel 4: Target Position (1 if target, else 0)
            target_mask = np.zeros_like(explored_mask, dtype=np.float32)
            target_row, target_col = self.target_position
            target_mask[target_row, target_col] = 1
            # Stack all channels together
            obs = np.stack([explored_mask, obstacle_mask, agent_mask, target_mask], axis=0)
        return obs
    
    def set_agent_position(self):
        # Set agent starting position based on difficulty config
        agent_start_mode = self.current_config.get("agent_start_mode", "fixed")
        if agent_start_mode == "fixed":
            self.agent_position = (1, 1)  # Fixed starting position for easier levels
        else:
            self.agent_position = self._random_position()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate new obstacle layout based on difficulty level
        self.grid_map = create_maze(self.num_rows, self.num_cols, self.maze_density)        
        self.set_agent_position()

        # Reset agent view and coverage
        self.agent_view = self._reset_agent_view()
        self.coverage_grid = np.zeros_like(self.grid_map)
        # Not sure if it should exist
        self.coverage = self.calculate_coverage()
        self.total_free_cells = np.sum(self.grid_map == 0)
        self.explored_free_cells = np.sum((self.coverage_grid == 1) & (self.grid_map == 0))
        
        
        # Set target position based on difficulty config
        self.target_position = self._calculate_new_target()
        
        self.steps = 0
        self.current_lives = self.agent_lives
        self.coverage_80_percent_reached = False
        self.coverage_90_percent_reached = False

        observation = self._calculate_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        self.steps += 1
        reward = -0.1  # Small step penalty to encourage efficiency
        terminated = False
        truncated = False

        # Check max steps
        if self.steps >= self.max_steps:
            print(f"MAX STEPS REACHED: {self.steps}/{self.max_steps}")
            truncated = True
            reward = -5.0
            # print("Episode truncated due to max steps reached.")
            # print("Reward:", reward, "Coverage:", self.coverage, "Lives remaining:", self.current_lives)
            return self._calculate_observation(), reward, terminated, truncated, {
                    "coverage": self.coverage,
                    "lives_remaining": self.current_lives,
                    "steps": self.steps}

        # Convert action and calculate new position
        action_row, action_column = self._action_to_direction(action)
        new_row = self.agent_position[0] + action_row
        new_column = self.agent_position[1] + action_column

        # Handle collision with walls or obstacles
        if (new_row < 0 or new_row >= self.num_rows or 
            new_column < 0 or new_column >= self.num_cols or
            self.grid_map[new_row, new_column] == 1):  
            
            # Lose a life instead of immediate termination
            self.current_lives -= 1

            if self.current_lives <= 0:
                terminated = True
                reward = -10.0
                # print("Agent has no lives left. Episode terminated.")
                # print("Reward:", reward, "Coverage:", self.coverage, "Lives remaining:", self.current_lives)
                return self._calculate_observation(), reward, terminated, truncated,  {
                    "coverage": self.coverage,
                    "lives_remaining": self.current_lives,
                    "steps": self.steps}
            else:
                reward -= 5.0
                # Don't move, but continue episode
                return self._calculate_observation(), reward, terminated, truncated, {}

        # Move agent if no collision
        self.agent_position = (new_row, new_column)
        previous_explored_cells = self.explored_free_cells
        # Update coverage grid (mark surrounding cells as explored)
        self._update_coverage_grid()
        
        # Update agent view 
        self._update_agent_view()
        
        # Calculate coverage
        self.coverage = self.calculate_coverage()
        if self.coverage >= 0.8 and self.coverage_80_percent_reached == False:
            self.coverage_80_percent_reached = True
            reward += 7.5
        if self.coverage >= 0.9 and self.coverage_90_percent_reached == False:
            self.coverage_90_percent_reached = True
            reward += 10
        # Check if the agent has fully explored the maze
        if self.coverage >= 1.0:
            print("MAZE FULLY EXPLORED!")
            reward += 100 - (self.steps * 0.1)  # Bonus for completing quickly
            terminated = True
        else:
            # Small reward for new exploration
            # reward += 0.75 * self.coverage
            reward += 0.5 * (self.explored_free_cells - previous_explored_cells)

            # Check if agent is adjacent to or on the target
            agent_r, agent_c = self.agent_position
            target_r, target_c = self.target_position
            if abs(agent_r - target_r) <= 1 and abs(agent_c - target_c) <= 1:
                reward += 7.5  # Target reached 
                self.target_position = self._calculate_new_target()
        
        return self._calculate_observation(), reward, terminated, truncated, {
            "coverage": self.coverage,
            "lives_remaining": self.current_lives,
            "steps": self.steps
        }


    # Gym required function to render environment
    def render(self):
        # Render to the console.
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                if np.array_equal((row, column),(self.agent_position)):
                    print(f'{str(GridTile.AGENT):>3}', end=' ')
                elif self.grid_map[row, column] == 1:
                    print(f'{str(GridTile.OBSTACLE):>3}', end=' ')
                elif np.array_equal((row, column), (self.target_position)):
                    print(f'{str(GridTile.TARGET):>3}', end=' ')
                else:
                    print(f'{str(GridTile.FLOOR):>3}', end=' ')
            print()
        print()
        

    def _random_position(self):
        valid_positions = np.argwhere(self.grid_map == 0)
        index = np.random.choice(len(valid_positions))
        return tuple(valid_positions[index])  # Convert to tuple for consistent comparison


    def _calculate_new_target(self):
        """Calculate new target position based on difficulty configuration"""
        target_mode = self.current_config.get("target_mode", "near_unexplored")
        
        if target_mode == "random":
            return self._random_position()
        elif target_mode == "near_unexplored":
            return self._sample_near_unexplored()
        elif target_mode == "path_based":
            return self._plan_target_path()
        else:
            # Default fallback — random is safest
            return self._random_position()


    
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
        

    def calculate_coverage(self):
        """Calculate coverage as ratio of explored free cells to total free cells"""
        self.explored_free_cells = np.sum((self.coverage_grid == 1) & (self.grid_map == 0))
        if self.total_free_cells == 0:
            return 1.0
        
        coverage = self.explored_free_cells / self.total_free_cells
        return min(coverage, 1.0)  # Ensure coverage doesn't exceed 1.0


    def _update_coverage_grid(self):
        """Update coverage grid based on agent's current position"""
        agent_row, agent_col = self.agent_position
        
        # Mark surrounding 8 cells + current cell as explored
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = agent_row + i
                col = agent_col + j
                if (0 <= row < self.num_rows and 0 <= col < self.num_cols and
                    self.grid_map[row, col] == 0):  # Only mark free cells
                    self.coverage_grid[row, col] = 1


    def _sample_near_unexplored(self, radius=2):
        """Sample a target position near unexplored (0) and explorable (non-obstacle) areas."""
        unexplored = [
            (r, c)
            for r in range(1, self.num_rows - 1)
            for c in range(1, self.num_cols - 1)
            if self.coverage_grid[r, c] == 0 and self.grid_map[r, c] == 0
        ]

        if not unexplored:
            return self._random_position()  # Fallback if everything is explored

        agent_r, agent_c = self.agent_position
        nearby = [
            (r, c) for (r, c) in unexplored
            if abs(r - agent_r) <= radius and abs(c - agent_c) <= radius
        ]

        return random.choice(nearby if nearby else unexplored)


    def _get_info(self): 
        info = {"agent_position": self.agent_position, "target_position": self.target_position}
        return info
    

    def _action_to_direction(self, action):
        action_map = {
            0: (-1, 0),   # Up
            1: (0, 1),    # Right
            2: (1, 0),    # Down
            3: (0, -1)    # Left
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