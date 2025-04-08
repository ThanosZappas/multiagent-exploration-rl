import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import maze_exploration_agent as agent
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='maze-exploration-v0',  # call it whatever you want
    entry_point='maze_exploration_env:MazeExplorationEnv',  # module_name:class_name
)


# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class MazeExplorationEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, grid_rows=3, grid_columns=3, render_mode=None):

        self.grid_rows = grid_rows
        self.grid_columns = grid_columns
        self.render_mode = render_mode

        # Initialize the MazeExploration problem
        self.maze_exploration = agent.MazeExploration(grid_rows=grid_rows, grid_columns=grid_columns,
                                                      obstacle_probability=0.85)

        self.action_space = spaces.Discrete(len(agent.AgentAction))

        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(3, 3),
            dtype=np.int64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.maze_exploration.reset(seed=seed)
        observation = self.get_observation()
        info = {}
        if self.render_mode == 'human':
            self.render()
        return observation, info

    def step(self, action):
        reward, terminated = self.maze_exploration.perform_action(agent.AgentAction(action))
        observation = self.get_observation()
        info = {}
        if self.render_mode == 'human':
            print(agent.AgentAction(action))
            self.render()
        return observation, reward, terminated, False, info

    def get_observation(self):
        # Initialize a 3x3 array with 1s (representing walls/obstacles)
        observation = np.ones((3, 3), dtype=int)

        # Get the agent's current position
        agent_row, agent_column = self.maze_exploration.agent_position

        # Fill the observation array with the corresponding maze values
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = agent_row + i
                column = agent_column + j
                if 0 <= row < self.grid_rows and 0 <= column < self.grid_columns:
                    observation[i + 1, j + 1] = self.maze_exploration.maze[row, column]

        # Set the agent's current position to 2
        observation[1, 1] = 2

        return observation

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
