import numpy as np
import random
import time
import matplotlib.pyplot as plt
import optuna
import optuna.visualization


class Agent:
    def __init__(self, start: tuple, real_stage, view_range=2):
        self.x = start[0]
        self.y = start[1]
        self.view_range = view_range
        self.explored_stage = np.full_like(real_stage, -1)
        self.explored_stage[self.x, self.y] = 0
        self.agent_view(real_stage)
        self.q_table = np.zeros((real_stage.shape[0], real_stage.shape[1], 4))  # Q-table for learning
        self.learning_rate = 0.001
        self.discount_factor = 0.8  # gamma, which balances immediate rewards with future rewards
        self.epsilon = 0.15  # Exploration rate

    def agent_view(self, real_stage):
        """ Refreshes the explored map of the agent (sees up, down, left, right). """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < real_stage.shape[0] and 0 <= ny < real_stage.shape[1]:
                self.explored_stage[nx, ny] = real_stage[nx, ny]

    def choose_action(self):
        """ Chooses an action using an epsilon-greedy policy. """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Explore randomly
        return np.argmax(self.q_table[self.x, self.y])  # Exploit best known action

    def move(self, action, maze):
        """ Moves the agent in the chosen direction. """
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nx, ny = self.x + moves[action][0], self.y + moves[action][1]
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
            self.x, self.y = nx, ny
        return (self.x, self.y)

    def update_q_table(self, old_state, action, reward, new_state):
        """ Updates the Q-table using the Q-learning algorithm. """
        old_q = self.q_table[old_state[0], old_state[1], action]
        future_q = np.max(self.q_table[new_state[0], new_state[1]])
        self.q_table[old_state[0], old_state[1], action] = (1 - self.learning_rate) * old_q + self.learning_rate * (
                reward + self.discount_factor * future_q)


def generate_stage(rows: int, cols: int, obs_prob=0.2):
    # generate obstacles with obs_prob probability
    num_obstacles = int(rows * cols * obs_prob)

    stage = np.full((rows, cols), 0)

    # Set 1s at random positions for the specified percentage
    indices = np.random.choice(rows * cols, num_obstacles, replace=False)
    stage.flat[indices] = 1

    return stage


def create_maze(rows, cols, obs_prob=0.8):
    rows = int(rows / 2)
    cols = int(cols / 2)

    maze = np.ones((rows * 2 + 1, cols * 2 + 1))

    x, y = (0, 0)

    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < rows and ny < cols and maze[2 * nx + 1, 2 * ny + 1] == 1:
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
    # draw_maze(maze)

    return maze


def draw_maze(maze, agent=None):
    print("Maximum cells to explore: ", maze.shape[0] * maze.shape[1] - np.sum(maze == 1)
    )
    plt.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    if agent:
        plt.scatter(agent.y, agent.x, color='red', s=100, marker='o')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# def images_to_gif(gif_filename=f"maze_{time.time()}.gif", duration=300, image_folder="tmp_img", gif_folder="utils"):
#     image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') and os.path.isfile(os.path.join(image_folder, f))]
#
#     image_files.sort()
#
#     images = []
#
#     for image_file in image_files:
#         image_path = os.path.join(image_folder, image_file)
#         image = Image.open(image_path)
#         images.append(image)
#
#     gif_filepath = os.path.join(gif_folder, gif_filename)
#     images[0].save(gif_filepath, save_all=True, append_images=images[1:], loop=0, duration=duration)
#
#     for image_file in image_files:
#         os.remove(os.path.join(image_folder, image_file))
#     time.sleep(1)



# def train_agent(maze, agent, episodes=1000):
#     """ Trains the agent using Q-learning for full maze exploration. """
#     total_reward = 0
#     for episode in range(episodes):
#         agent.x, agent.y = (1, 1)  # Reset agent position to start
#         visited = set()
#         for _ in range(1500):  # Increase steps to allow full exploration
#             old_state = (agent.x, agent.y)
#             action = agent.choose_action()
#             new_state = agent.move(action, maze)
#             visited.add(new_state)
#             reward = 1 if new_state not in visited else -0.01  # Reward new exploration
#             agent.update_q_table(old_state, action, reward, new_state)
#             total_reward += reward
#             if len(visited) == (maze.shape[0] * maze.shape[1]) - np.sum(maze == 1):
#                 # print("all cells explored\n")
#                 total_reward += 100
#                 break  # Stop when all cells are explored
#             # agent.epsilon = max(0.05, agent.epsilon * 0.999)  # Ensure some exploration remains
#         if episode % 100 == 0:
#             print(f"Episode {episode}: Total Reward = {total_reward}, Explored Cells = {len(visited)}")
#     return total_reward


def train_agent(maze, agent, episodes=1500):
    """ Trains the agent using Q-learning for full maze exploration. """
    total_reward = 0
    for episode in range(episodes):
        agent.x, agent.y = (1, 1)  # Reset agent position to start
        visited = set()
        recent_positions = []  # Track recent positions to detect loops

        for _ in range(3000):  # More steps to encourage full exploration
            old_state = (agent.x, agent.y)
            action = agent.choose_action()
            new_state = agent.move(action, maze)

            reward = 0  # Initialize reward

            if new_state not in visited:
                reward += 5  # Reward new exploration
                visited.add(new_state)
            else:
                reward -= 1  # Small penalty for revisiting

            # Dynamic reward for coverage (less aggressive scaling)
            coverage = len(visited) / ((maze.shape[0] * maze.shape[1]) - np.sum(maze == 1))
            reward += coverage * 5  # Increased base reward, but not exponential

            # Penalties for staying in the same area too long
            recent_positions.append(new_state)
            if len(recent_positions) > 50:  # Longer tracking for loop detection
                recent_positions.pop(0)
            if recent_positions.count(new_state) > 10:  # More strict loop detection
                reward -= 5  # Higher penalty for loops

            # Collision penalty (if agent moves into a wall)
            if maze[new_state] == 1:
                reward -= 5
                new_state = old_state  # Reset to prevent getting stuck

            # High bonus for reaching 100% exploration
            if coverage >= 1.0:
                reward += 500  # Big incentive for full coverage
                break

            agent.update_q_table(old_state, action, reward, new_state)
            total_reward += reward

        if episode % 100 == 0:
            print(
                f"Episode {episode}: Total Reward = {total_reward}, Explored Cells = {len(visited)}, Coverage = {coverage:.2%}")

    return total_reward


# optimize based on the total reward
def optimize_parameters(maze, agent, trials=10):
    """ Optimizes the hyperparameters of the agent using Optuna. """

    def objective(trial):
        agent.learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.2)
        agent.discount_factor = trial.suggest_float("discount_factor", 0.6, 0.99)
        agent.epsilon = trial.suggest_float("epsilon", 0.05, 0.2)
        reward = train_agent(maze, agent, episodes=200)
        return reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    print("Optimization complete. Best parameters:")
    print(study.best_params)
    plot_optimization(study)
    return study.best_params


def plot_optimization(study):
    """ Plots the optimization results. """
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_contour(study).show()
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.show()


def main():
    start = (1, 1)

    maze = create_maze(rows=20, cols=20, obs_prob=0.85)
    agent = Agent(start, maze)
    draw_maze(maze, agent)
    train_agent(maze, agent, episodes=1500)
    # best_params = optimize_parameters(maze, agent, trials=12)

    # maze = create_maze(15, 15, 0.15)
    # agent = Agent(start, maze)
    # draw_maze(maze, agent)
    # train_agent(maze, agent, episodes=2000)
    # print("Training complete. Agent should now explore the entire maze efficiently.")
    # print("Learned Q-table:")
    # print(agent.q_table)


if __name__ == "__main__":
    main()