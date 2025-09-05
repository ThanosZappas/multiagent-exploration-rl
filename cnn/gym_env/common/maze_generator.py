import numpy as np
import random

def create_maze(rows, cols, obs_prob=0.85):
     # Calculate base dimensions for maze generation
    base_rows = int((rows - 1) / 2)
    base_cols = int((cols - 1) / 2)

    # Create maze of exact size rows x cols
    maze = np.ones((rows, cols))
    maze[1, 1] = 0  # Starting position (1,1) should be free
    
    x, y = (0, 0)
    
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < base_rows and ny < base_cols and maze[2 * nx + 1, 2 * ny + 1] == 1:
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    zero_indices = np.argwhere(maze == 0)
    zero_coords = [tuple(index) for index in zero_indices]

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # adds randomly crosses of free space.
    for z in zero_coords:
        if random.random() >= obs_prob:
            for dx, dy in directions:
                nx, ny = z[0] + dx, z[1] + dy
                maze[nx, ny] = 0

    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # removes crosses (so agents wont be stuck).
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            walls = []
            for d in directions:
                neighbor_i = i + d[0]
                neighbor_j = j + d[1]
                # Check if neighbor is in bounds
                if 0 <= neighbor_i < maze.shape[0] and 0 <= neighbor_j < maze.shape[1] and maze[
                    (neighbor_i, neighbor_j)]:
                    walls.append((neighbor_i, neighbor_j))
            if len(walls) >= len(directions):
                for coord in walls:
                    maze[coord] = 0

    # re-adds the boundaries (after cross removed).
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    maze[1, 1] = 0  # Starting position (1,1) should be free

    return maze

def create_simple_maze_plot(maze, save_path=None, title=None, figsize=(8, 8), save_figure=False):
    """
    Create a simple black and white plot of the maze without any extras.
    
    Args:
        maze (np.array): 2D maze array where 1=wall/obstacle, 0=free cell
        save_path (str, optional): Path to save the PNG file
        title (str, optional): Title for the plot (if None, no title shown)
        figsize (tuple): Figure size (width, height)
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt
    
    save_path = "gym_env/plots/maze_figures/" + title.replace(" ", "_").lower() + ".png" if save_path is None else save_path
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create the maze visualization
    # 0 (free cells) -> white (1.0)
    # 1 (walls/obstacles) -> black (0.0)
    maze_visual = 1 - maze  # Invert: 0->1 (white), 1->0 (black)
    
    # Display the maze - pure black and white
    ax.imshow(maze_visual, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    
    # # Set title only if provided
    # if title:
    #     ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Remove all axes, ticks, and labels for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Make sure the aspect ratio is equal (square cells)
    ax.set_aspect('equal')
    
    # Remove any padding
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    # Save if path provided
    if save_figure:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0)
        print(f"Simple maze plot saved to: {save_path}")
    
    return fig

if __name__ == "__main__":

    fig1 = create_simple_maze_plot(create_maze(10, 10, 0.30),
                           title="10x10 - 15% Obstacles",save_figure=True)
    fig2 = create_simple_maze_plot(create_maze(15, 15, 0.30),
                           title="15x15 - 15% Obstacles)",save_figure=True)
    fig3 = create_simple_maze_plot(create_maze(30, 30, 0.30),
                           title="30x30 - 15% Obstacles)",save_figure=True)
    fig4 = create_simple_maze_plot(create_maze(10, 10, 0.85),
                           title="10x10 - 85% Obstacles",save_figure=True)
    fig5 = create_simple_maze_plot(create_maze(15, 15, 0.85),
                           title="15x15 - 85% Obstacles)",save_figure=True)
    fig6 = create_simple_maze_plot(create_maze(30, 30, 0.85),
                           title="30x30 - 85% Obstacles)",save_figure=True)
    # Show all plots
    import matplotlib.pyplot as plt
    plt.show()
    