import random
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from common import maze_generator as maze_generator

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='maze-exploration-v1',  # call it whatever you want
    entry_point='environment.maze_exploration_env:MazeExplorationEnv', # module_name:class_name
)

class MazeExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_map=None, max_steps=300, channels=3, difficulty_level=1):
        super(MazeExplorationEnv, self).__init__()

        # Curriculum configuration
        self.difficulty_config = {
            1: {"maze_density": 0.15, "target_mode": "fixed", "random_start": False},     # Very open, few walls
            2: {"maze_density": 0.35, "target_mode": "random", "random_start": False},    # Some corridors
            3: {"maze_density": 0.55, "target_mode": "random", "random_start": True},     # Moderate complexity
            4: {"maze_density": 0.70, "target_mode": "near_unexplored", "random_start": True},  # Complex paths
            5: {"maze_density": 0.85, "target_mode": "near_unexplored", "random_start": True},  # Dense maze
        }
        
        self.difficulty_level = difficulty_level
        self.current_config = self.difficulty_config[difficulty_level]
        
        # Initialize base grid (10x10 with outer walls)
        if grid_map is None:
            self.base_grid = maze_generator.empty_maze()
            self.base_grid = maze_generator.init()
        else:
            self.base_grid = np.array(grid_map)
                   
        self.grid_map = self.base_grid.copy()
        self.num_rows, self.num_cols = self.grid_map.shape
        
        # Initialize positions and state
        self.agent_position = None
        self.target_position = None
        self.fixed_target_position = (1, 2)  # Fixed target for level 1
        self.action_space = spaces.Discrete(4)  # Only cardinal directions: up, right, down, left
        self.channels = channels
        self.agent_lives = 3  # Number of lives for obstacle collisions
        self.agent_view = self._reset_agent_view()
        self.coverage_grid = np.zeros_like(self.grid_map)  # Track explored areas
        self.fig = None
        self.ax = None
        self.steps = 0
        self.max_steps = max_steps

        # Set observation space based on channels
        if self.channels == 1:
            self.observation_space = spaces.Box(
                low=0, high=4, shape=(1, self.num_rows, self.num_cols), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(3, self.num_rows, self.num_cols), dtype=np.uint8
            )

    def _calculate_observation(self):
        self._update_agent_view()
        if self.channels == 1:
            obs = np.expand_dims(self.agent_view.astype(np.uint8), axis=0)
        else:
            # Channel 0: Explored mask (1 if explored, else 0)
            explored_mask = (self.agent_view == MazeElements.EXPLORED).astype(np.uint8)
            
            # Channel 1: Discovered obstacles (1 if discovered obstacle, else 0)
            obstacle_mask = (self.agent_view == MazeElements.OBSTACLE).astype(np.uint8)
            
            # Channel 2: Agent position (1 if agent, else 0)
            agent_mask = np.zeros_like(explored_mask, dtype=np.uint8)
            agent_row, agent_col = self.agent_position
            agent_mask[agent_row, agent_col] = 1
            
            obs = np.stack([explored_mask, obstacle_mask, agent_mask], axis=0)
        return obs
     

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate new obstacle layout based on difficulty level
        maze_generator.init()
        
        # Set agent starting position based on difficulty config
        if self.current_config["random_start"]:
            self.agent_position = self._random_position()
        else:
            self.agent_position = (1, 2)  # Fixed starting position for easier levels
        
        # Reset agent view and coverage
        self.agent_view = self._reset_agent_view()
        self.coverage_grid = np.zeros_like(self.grid_map)
        
        # Set target position based on difficulty config
        self.target_position = self._calculate_new_target()
        
        self.steps = 0
        self.terminated = False
        self.current_lives = self.agent_lives

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
            truncated = True
            reward = -1.0
            return self._calculate_observation(), reward, terminated, truncated, {}

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
            reward = -1.0
            
            if self.current_lives <= 0:
                terminated = True
                return self._calculate_observation(), reward, terminated, truncated, {}
            else:
                # Don't move, but continue episode
                return self._calculate_observation(), reward, terminated, truncated, {}

        # Move agent if no collision
        old_position = self.agent_position
        self.agent_position = (new_row, new_column)
        
        # Update coverage grid (mark surrounding cells as explored)
        self._update_coverage()
        
        # Update agent view 
        self._update_agent_view()
        
        # Calculate coverage
        coverage = self.calculate_coverage()
        
        # Check if the agent has fully explored the maze
        if coverage >= 1.0:
            terminated = True
            reward = 100 - self.steps * 0.25  # Bonus for completing quickly
        else:
            # Check if agent reached target
            if np.array_equal(self.agent_position, self.target_position):
                reward += 10.0  # Target reached bonus
                self.target_position = self._calculate_new_target()
            
            # Small reward for new exploration
            reward += 0.25 * coverage
            
        return self._calculate_observation(), reward, terminated, truncated, {
            "coverage": coverage,
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
                elif np.array_equal((row, column), (self.target_position)): #TODO: Fix this not showing target
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
        target_mode = self.current_config["target_mode"]
        
        if target_mode == "fixed":
            return self.fixed_target_position
        elif target_mode == "random":
            return self._random_position()
        elif target_mode == "near_unexplored":
            return self._sample_near_unexplored()
        else:
            # Default fallback
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
        # Count total free cells (excluding walls and obstacles)
        total_free_cells = np.sum(self.grid_map == 0)
        
        # Count explored free cells
        explored_free_cells = np.sum((self.coverage_grid == 1) & (self.grid_map == 0))
        
        if total_free_cells == 0:
            return 1.0
        
        coverage = explored_free_cells / total_free_cells
        return min(coverage, 1.0)  # Ensure coverage doesn't exceed 1.0


    def _update_coverage(self):
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
        """Sample target position near unexplored areas"""
        # Find unexplored cells (excluding walls and obstacles)
        unexplored = []
        for row in range(1, self.num_rows - 1):
            for col in range(1, self.num_cols - 1):
                if (self.coverage_grid[row, col] == 0 and 
                    self.grid_map[row, col] == 0):
                    unexplored.append((row, col))
        
        if len(unexplored) == 0:
            # If no unexplored areas, return any valid position
            return self._random_position()

        agent_row, agent_col = self.agent_position
        # Find unexplored cells within radius
        nearby = []
        for row, col in unexplored:
            if abs(row - agent_row) <= radius and abs(col - agent_col) <= radius:
                nearby.append((row, col))
        
        # Return nearby unexplored cell if available, otherwise any unexplored cell
        if nearby:
            return random.choice(nearby)
        else:
            return random.choice(unexplored)

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