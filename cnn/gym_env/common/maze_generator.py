import numpy as np
import random

## TODO: Correct maze creation.

def empty_maze():
    base_grid = np.zeros((10, 10))
    base_grid[0, :] = 1  # Top wall
    base_grid[-1, :] = 1  # Bottom wall
    base_grid[:, 0] = 1  # Left wall
    base_grid[:, -1] = 1  # Right wall
    return base_grid

def init(self):
        """Generate obstacles based on the current difficulty level"""
        maze_density = self.current_config["maze_density"]
        
        # Generate maze with proper obstacle density using create_maze function
        self.grid_map = create_maze(self.num_rows, self.num_cols, maze_density)
        
        # Clear starting area more aggressively for lower levels
        if self.difficulty_level <= 3:  # Be more generous with clearing for first 3 levels
            clear_radius = 4 - self.difficulty_level  # 3 cells for level 1, 2 for level 2, 1 for level 3
            start_pos = (1, 2)  # Fixed starting position
            for i in range(-clear_radius, clear_radius + 1):
                for j in range(-clear_radius, clear_radius + 1):
                    row = start_pos[0] + i
                    col = start_pos[1] + j
                    if (0 < row < self.num_rows - 1 and 
                        0 < col < self.num_cols - 1):  # Don't clear outer walls
                        self.grid_map[row, col] = 0
                        
        # Verify the final grid meets connectivity criteria
        if not self._is_connected(self.grid_map):
            # Try to repair the grid
            self._ensure_connected_grid(self.grid_map)
            
            # If still not connected, fall back to base grid
            if not self._is_connected(self.grid_map):
                print("Warning: Generated maze was not fully connected. Using fallback grid.")
                self.grid_map = self.base_grid.copy()

        return self.grid_map

def create_maze(rows, columns, obstacle_probability=0.85):
    """Generate a maze using DFS with configurable obstacle probability.
    
    Args:
        rows (int): Number of rows in the maze
        columns (int): Number of columns in the maze
        obstacle_probability (float): Controls how many walls are kept:
            - Low (0.15) = More open space (easier)
            - High (0.85) = Dense maze (harder)
    
    Returns:
        np.array: Generated maze with 0 (path) and 1 (wall)
    """
    # ...existing maze creation code...
    inner_rows = (rows - 1) // 2
    inner_cols = (columns - 1) // 2
    
    maze = np.ones((rows, columns))
    
    # Generate maze using DFS
    x, y = (0, 0)
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < inner_rows and 0 <= ny < inner_cols and 
                maze[2 * nx + 1, 2 * ny + 1] == 1):
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    zero_indices = np.argwhere(maze == 0)
    zero_coords = [tuple(index) for index in zero_indices]

    # Randomly remove walls based on probability
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for z in zero_coords:
        if random.random() >= obstacle_probability:
            for dx, dy in directions:
                nx, ny = z[0] + dx, z[1] + dy
                if (0 < nx < rows - 1 and 0 < ny < columns - 1):
                    maze[nx, ny] = 0

    # Ensure boundaries are walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # Clear starting area
    maze[1:3, 1:4] = 0

    return maze

def _ensure_connected_grid(self, grid):
        """Modify grid to ensure all free cells are connected"""
        while not self._is_connected(grid):
            # Find disconnected regions
            start_pos = (1, 2)
            visited = set()
            
            def flood_fill(pos):
                row, col = pos
                if (row < 0 or row >= self.num_rows or 
                    col < 0 or col >= self.num_cols or 
                    grid[row, col] == 1 or 
                    pos in visited):
                    return
                
                visited.add(pos)
                for dr, dc in [(-1,0), (0,1), (1,0), (0,-1)]:  # Up, Right, Down, Left
                    flood_fill((row + dr, col + dc))
            
            flood_fill(start_pos)
            
            # Find disconnected free cells
            disconnected = []
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    if grid[row, col] == 0 and (row, col) not in visited:
                        disconnected.append((row, col))
            
            if not disconnected:
                break
                
            # For each disconnected cell, try to create a path to the main region
            for cell in disconnected:
                row, col = cell
                # Find closest visited cell
                min_dist = float('inf')
                best_path = None
                
                for vrow, vcol in visited:
                    dist = abs(row - vrow) + abs(col - vcol)
                    if dist < min_dist:
                        min_dist = dist
                        # Create direct path
                        path = []
                        curr_row, curr_col = row, col
                        while (curr_row, curr_col) != (vrow, vcol):
                            if curr_row < vrow:
                                curr_row += 1
                            elif curr_row > vrow:
                                curr_row -= 1
                            if curr_col < vcol:
                                curr_col += 1
                            elif curr_col > vcol:
                                curr_col -= 1
                            path.append((curr_row, curr_col))
                        best_path = path
                
                # Create path by removing obstacles
                if best_path:
                    for prow, pcol in best_path:
                        grid[prow, pcol] = 0


def _is_connected(self, grid):
        """Check if all free cells in the grid are connected/reachable"""
        def flood_fill(pos, visited):
            row, col = pos
            if (row < 0 or row >= self.num_rows or 
                col < 0 or col >= self.num_cols or 
                grid[row, col] == 1 or 
                pos in visited):
                return
            
            visited.add(pos)
            # Check cardinal directions
            for dr, dc in [(-1,0), (0,1), (1,0), (0,-1)]:  # Up, Right, Down, Left
                flood_fill((row + dr, col + dc), visited)
        
        # Start flood fill from agent's starting position
        start_pos = (1, 2)  # Default starting position
        visited = set()
        flood_fill(start_pos, visited)
        
        # Count all free cells
        free_cells = set()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if grid[row, col] == 0:
                    free_cells.add((row, col))
        
        # Check if all free cells were visited
        return len(visited) == len(free_cells)

def print_maze(maze1, maze2):
    print("Low obstacle density (0.15):")
    print(np.array2string(maze1, separator=' '))
    print("\nHigh obstacle density (0.85):")
    print(np.array2string(maze2, separator=' '))

if __name__ == "__main__":
    maze1 = create_maze(15, 15, 0.70)
    maze2 = create_maze(15, 15, 0.85)
    print_maze(maze1,maze2)
    