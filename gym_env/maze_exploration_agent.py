"""This module models the problem to be solved. In this very simple example, the problem is to optimize an Agent that
searches a Maze. The Maze is divided into a rectangular grid. The Agent's goal is to explore all the cells of the
maze."""

import random
from enum import Enum

import numpy as np
import pygame
import sys
from os import path


# Maze generation functions
def create_maze(rows, columns, obstacle_probability=0.8):
    # Reduce the input dimensions to generate a maze with corridors.
    rows = int(rows / 2)
    columns = int(columns / 2)

    maze = np.ones((rows * 2 + 1, columns * 2 + 1))
    x, y = (0, 0)
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < rows and ny < columns and maze[2 * nx + 1, 2 * ny + 1] == 1:
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    zero_indices = np.argwhere(maze == 0)
    zero_coords = [tuple(index) for index in zero_indices]

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Randomly remove some walls to add more open space
    for z in zero_coords:
        if random.random() >= obstacle_probability:
            for dx, dy in directions:
                nx, ny = z[0] + dx, z[1] + dy
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
                    maze[nx, ny] = 0

    # Add boundaries
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # Remove potential dead-end crosses to avoid trapping the agent
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            walls = []
            for d in directions:
                ni = i + d[0]
                nj = j + d[1]
                if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1] and maze[ni, nj]:
                    walls.append((ni, nj))
            if len(walls) >= len(directions):
                for coord in walls:
                    maze[coord] = 0

    # Re-add boundaries after removal.
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    return maze


# Actions the Agent is capable of performing i.e. go in a certain direction
class AgentAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


# The Maze is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(Enum):
    FLOOR = 0
    WALL = 1
    AGENT = 2

    # Return the first letter of tile name, for printing to the console.
    def __str__(self):
        string_array = ['_', '#', 'X']
        return string_array[self.value]


class MazeExploration:

    def __init__(self, grid_rows=5, grid_columns=5, obstacle_probability=0.85, fps=1):
        """
        grid_rows, grid_columns: dimensions for maze generation.
        obstacle_probability: used in maze generation (controls obstacle probability).
        fps: rendering frames per second.
        """
        # Save inputs for maze generation. The generated maze will update grid_rows and grid_columns.
        self.input_rows = grid_rows+2
        self.input_columns = grid_columns+2
        self.obstacle_probability = obstacle_probability
        self.fps = fps
        self.last_action = ''
        self._init_pygame()
        self.reset()

    def _init_pygame(self):
        pygame.init()  # initialize pygame
        pygame.display.init()  # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)

        # Define game window size (width, height)
        self.window_size = (self.cell_width * self.input_columns, self.cell_height
                            * self.input_rows + self.action_info_height)
        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size)

        # Load & resize sprites
        file_name = path.join(path.dirname(__file__), "sprites/firefighter.png")
        img = pygame.image.load(file_name)
        self.agent_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/tile_floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/fire_hydrant_with_tile_floor.png")
        img = pygame.image.load(file_name)
        self.obstacle_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/brick2.png")
        img = pygame.image.load(file_name)
        self.wall_img = pygame.transform.scale(img, self.cell_size)

    def reset(self, seed=None):
        # Generate a new maze using create_maze.
        self.maze = create_maze(self.input_rows, self.input_columns, self.obstacle_probability)
        # Update grid dimensions from the generated maze.
        self.grid_rows, self.grid_columns = self.maze.shape

        # Update the window size based on the new dimensions.
        self.window_size = (
            self.cell_width * self.grid_columns, self.cell_height * self.grid_rows + self.action_info_height)
        # self.window_surface = pygame.display.set_mode(self.window_size)

        # Choose a starting free cell for the agent.
        free_cells = np.argwhere(self.maze == 0)
        if seed is not None:
            random.seed(seed)
            self.agent_position = list(random.choice(free_cells))
        else:
            self.agent_position = (1, 1)  # default starting position

        # Initialize visited grid for free cells only.
        self.visited = np.zeros((self.grid_rows, self.grid_columns), dtype=bool)
        self.visited[self.agent_position[0], self.agent_position[1]] = True

    def perform_action(self, agent_action: AgentAction) -> (float, bool):
        self.last_action = agent_action

        # Determine the target position based on the action.
        new_position = list(self.agent_position).copy()
        if agent_action == AgentAction.LEFT:
            new_position[1] -= 1
        elif agent_action == AgentAction.RIGHT:
            new_position[1] += 1
        elif agent_action == AgentAction.UP:
            new_position[0] -= 1
        elif agent_action == AgentAction.DOWN:
            new_position[0] += 1
        reward = -0.1  # Default reward for a step.
        row, column = new_position
        # Check for boundaries.
        if row < 0 or row >= self.grid_rows or column < 0 or column >= self.grid_columns:
            reward += -1.0
            return reward, False  # Out of bounds: ignore action.

        # Check for wall collision.
        if self.maze[row, column] == 1:
            reward += -1.0
            return reward, False  # Hit a wall: do not move.

        # Valid move: update agent position.
        self.agent_position = new_position
        # Reward new free cell visits.
        if not self.visited[row, column]:
            self.visited[row, column] = True
            reward += 2.0  # Small reward for a new cell.


        # Check if all free cells have been visited.
        # Only count cells where maze == 0.
        free_cell_indices = (self.maze == 0)
        if np.all(self.visited[free_cell_indices]):
            reward += float(np.sum(free_cell_indices) * 15)  # Big reward for task completion.
            terminated = True
        else:
            terminated = False

        return reward, terminated

    def render(self):
        # Render to the console.
        for row in range(self.grid_rows):
            for column in range(self.grid_columns):
                if [row, column] == self.agent_position:
                    print(str(GridTile.AGENT), end=' ')
                elif self.maze[row, column] == 1:
                    print(str(GridTile.WALL), end=' ')
                else:
                    print(str(GridTile.FLOOR), end=' ')
            print()
        print()

        self._process_events()

        # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        self.window_surface.fill((255, 255, 255))

        # Draw each cell.
        for row in range(self.grid_rows):
            for column in range(self.grid_columns):
                pos = (column * self.cell_width, row * self.cell_height)
                if self.maze[row, column] == 1:
                    if row == 0 or column == 0 or row == self.grid_rows-1 or column == self.grid_columns-1:
                        self.window_surface.blit(self.wall_img, pos)
                    else:
                        self.window_surface.blit(self.obstacle_img, pos)
                else:
                    self.window_surface.blit(self.floor_img, pos)
                if [row, column] == self.agent_position:
                    self.window_surface.blit(self.agent_img, pos)

        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0, 0, 0), (255, 255, 255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)

        pygame.display.update()
        # Limit frames per second
        self.clock.tick(self.fps)

    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if (event.type == pygame.KEYDOWN):
                # User hit escape
                if (event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

# For unit testing
if __name__ == "__main__":
    mazeExploration = MazeExploration(grid_rows=10, grid_columns=10, obstacle_probability=0.85, fps=1)
    mazeExploration.render()
    for i in range(10):
        # Randomly select an action.
        rand_action = random.choice(list(AgentAction))
        reward, terminated = mazeExploration.perform_action(rand_action)
        print("Reward:", reward, "Terminated:", terminated)
        mazeExploration.render()
        if terminated:
            print("Maze completed!")
            break