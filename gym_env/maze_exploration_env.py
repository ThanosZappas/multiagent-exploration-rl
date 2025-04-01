import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
# import gym_env.maze_exploration_agent as agent

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

    def __init__(self, grid_rows=5, grid_columns=5, render_mode=None):

        self.grid_rows = grid_rows
        self.grid_columns = grid_columns
        self.render_mode = render_mode

        # Initialize the MazeExploration problem
        self.maze_exploration = agent.MazeExploration(grid_rows=grid_rows, grid_columns=grid_columns, obstacle_probability=0.85)

        self.action_space = spaces.Discrete(len(agent.AgentAction))

        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows - 1, self.grid_columns - 1]),
            shape=(2,),
            dtype=np.int64
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # gym requires this call to control randomness and reproduce scenarios.

        # Reset the MazeExploration. Optionally, pass in seed control randomness and reproduce scenarios.
        self.maze_exploration.reset(seed=seed)

        # Construct the observation state:
        observation = np.array(self.maze_exploration.agent_position)

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if self.render_mode == 'human':
            self.render()

        # Return observation and info
        return observation, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Perform action
        reward, terminated = self.maze_exploration.perform_action(agent.AgentAction(action))

        # Construct the observation state:
        observation = np.array(self.maze_exploration.agent_position)

        info = {}
        if self.render_mode == 'human':
            print(agent.AgentAction(action))
            self.render()
        return observation, reward, terminated, False, info

    # Gym required function to render environment
    def render(self):
        self.maze_exploration.render()


# For unit testing
if __name__ == "__main__":
    env = gym.make('maze-exploration-v0', render_mode='human')
    observation, _ = env.reset()
    for i in range(10):
        rand_action = env.action_space.sample()
        observation, reward, terminated, _, _ = env.step(rand_action)
        if terminated:
            observation, _ = env.reset()
