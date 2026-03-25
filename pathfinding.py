"""
AI Pathfinding Project: Repeated A* and Adaptive A* Implementation

This module implements several pathfinding strategies for navigating a gridworld
when the agent does not know all blocked cells ahead of time.

Algorithms included:
- Repeated Forward A*
- Repeated Backward A*
- Adaptive A*

The code also supports:
- Random gridworld generation
- Saving/loading environments
- Running experiments across many environments
- Visualizing environments and solutions
"""

# Numerical array operations for representing the grid and doing matrix-style work
import numpy as np

# Priority queue used by A* to always expand the most promising state first
import heapq

# Plotting utilities for saving images of the gridworld
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Runtime measurement for experiments
import time

# Random choices for maze generation, tie-breaking behavior, and reproducibility
import random

# Useful for default dictionary behavior if needed in extensions of this project
from collections import defaultdict

# Used to save and load entire GridWorld objects to files
import pickle

# Filesystem utilities for creating folders and working with paths
import os


class GridWorld:
    """Represents a gridworld environment with blocked and unblocked cells."""
    # This class stores the environment itself.
    # Each cell is either:
    #   0 -> unblocked
    #   1 -> blocked
    #
    # It also stores the start and goal positions used by the agent.

    def __init__(self, size=51):
        # Size of the grid (for a 51x51 world by default)
        self.size = size

        # Create an empty grid of all zeros first.
        # This is just an initial placeholder; maze generation will overwrite it.
        self.grid = np.zeros((size, size), dtype=int)

        # Start and goal are assigned later after the maze is generated.
        self.start = None
        self.goal = None

    def generate_maze_dfs(self, block_probability=0.3, seed=None):
        """Generate maze using depth-first search with random tie-breaking."""
        # This function creates a random maze-like environment.
        #
        # Main idea:
        # 1. Treat every cell as initially blocked/unvisited.
        # 2. Use a DFS-style process to visit cells.
        # 3. When a new cell is discovered, randomly decide whether it stays blocked
        #    or becomes unblocked.
        # 4. After the grid is built, choose valid start and goal cells.

        # If a seed is provided, use it so the same environment can be reproduced.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Start by marking every cell as blocked.
        # This means the generator will "open up" cells as it explores.
        self.grid = np.ones((self.size, self.size), dtype=int)

        # Track whether each cell has been visited by the maze generator.
        visited = np.zeros((self.size, self.size), dtype=bool)

        # Choose a random starting cell for DFS.
        start_x = random.randint(0, self.size - 1)
        start_y = random.randint(0, self.size - 1)

        # DFS uses a stack. The last item is the current cell being explored.
        stack = [(start_x, start_y)]

        # Mark the starting cell as visited and force it to be unblocked.
        visited[start_x, start_y] = True
        self.grid[start_x, start_y] = 0

        # 4-connected movement: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Keep going until every cell has been visited by the generator.
        while np.sum(visited) < self.size * self.size:
            # If the DFS stack becomes empty, that means the current branch is done.
            # In that case, pick a random unvisited cell and start a new branch.
            if len(stack) == 0:
                unvisited = np.argwhere(~visited)
                if len(unvisited) == 0:
                    break

                idx = random.randint(0, len(unvisited) - 1)
                current = tuple(unvisited[idx])

                stack.append(current)
                visited[current[0], current[1]] = True

                # When restarting DFS from a fresh cell, force that cell to be open.
                self.grid[current[0], current[1]] = 0
                continue

            # Look at the current DFS cell without removing it yet.
            current = stack[-1]

            # Collect all unvisited neighbors of the current cell.
            neighbors = []
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy

                # Only include cells inside the grid that have not been visited yet.
                if 0 <= nx < self.size and 0 <= ny < self.size and not visited[nx, ny]:
                    neighbors.append((nx, ny))

            # If there are no unvisited neighbors, this is a dead end.
            # Backtrack by popping the stack.
            if len(neighbors) == 0:
                stack.pop()
            else:
                # Otherwise choose one random unvisited neighbor to continue DFS.
                next_cell = neighbors[random.randint(0, len(neighbors) - 1)]
                visited[next_cell[0], next_cell[1]] = True

                # Randomly decide whether this visited cell is blocked or unblocked.
                # If it is unblocked, we keep exploring from it by pushing it to the stack.
                # If it is blocked, it stays out of the DFS expansion path.
                if random.random() < block_probability:
                    self.grid[next_cell[0], next_cell[1]] = 1
                else:
                    self.grid[next_cell[0], next_cell[1]] = 0
                    stack.append(next_cell)

        # After generating the map, choose valid start and goal cells.
        self._set_start_goal()

    def _set_start_goal(self):
        """Set start and goal positions in unblocked cells."""
        # Find every unblocked cell in the grid.
        unblocked = np.argwhere(self.grid == 0)

        # If there are not enough open cells, force two corner cells to be open.
        # This guarantees a valid start and goal exist.
        if len(unblocked) < 2:
            self.grid[0, 0] = 0
            self.grid[self.size - 1, self.size - 1] = 0
            self.start = (0, 0)
            self.goal = (self.size - 1, self.size - 1)
        else:
            # Randomly choose two different open cells for start and goal.
            idx1 = random.randint(0, len(unblocked) - 1)
            idx2 = random.randint(0, len(unblocked) - 1)

            while idx1 == idx2:
                idx2 = random.randint(0, len(unblocked) - 1)

            self.start = tuple(unblocked[idx1])
            self.goal = tuple(unblocked[idx2])

    def is_blocked(self, pos):
        """Check if a position is blocked."""
        x, y = pos

        # Any position outside the grid is treated as blocked.
        # This is convenient because pathfinding should never move off the map.
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True

        return self.grid[x, y] == 1

    def get_neighbors(self, pos):
        """Get valid neighbors (4-connected)."""
        x, y = pos
        neighbors = []

        # Only allow movement in the four main directions.
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # Add the neighbor only if it is still inside the grid.
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))

        return neighbors

    def visualize(self, path=None, expanded=None, title="GridWorld", save_path=None):
        """Visualize the gridworld."""
        # Create a plotting area large enough to clearly see the world.
        fig, ax = plt.subplots(figsize=(10, 10))

        # Work on a copy so visualization does not modify the real environment.
        vis_grid = np.copy(self.grid).astype(float)

        # If a set of expanded cells is provided, color them differently.
        # Only recolor cells that are actually unblocked in the real grid.
        if expanded is not None:
            for cell in expanded:
                if vis_grid[cell[0], cell[1]] == 0:
                    vis_grid[cell[0], cell[1]] = 0.5

        # White  -> open cells
        # Black  -> blocked cells
        # Light blue -> expanded cells
        cmap = ListedColormap(['white', 'black', 'lightblue'])
        ax.imshow(vis_grid, cmap=cmap, origin='upper')

        # If a path is provided, draw it as a red line over the grid.
        if path is not None and len(path) > 0:
            path_array = np.array(path)

            # Note the coordinate swap:
            # matplotlib uses (column, row) while our grid stores (row, column).
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='Path')

        # Draw start and goal markers if they exist.
        if self.start:
            ax.plot(self.start[1], self.start[0], 'go', markersize=15, label='Start')
        if self.goal:
            ax.plot(self.goal[1], self.goal[0], 'r*', markersize=20, label='Goal')

        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save the figure if an output file path was provided.
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Close the plot so repeated experiment runs do not leave many figures open.
        plt.close()

        return fig


class Agent:
    """Agent that navigates the gridworld with limited visibility."""
    # The agent does NOT begin with full knowledge of blocked cells.
    # It learns about blocked cells only by observing adjacent cells
    # from its current position.

    def __init__(self, gridworld):
        # Reference to the world the agent is moving through.
        self.gridworld = gridworld

        # The agent always starts at the world's start cell.
        self.position = gridworld.start

        # Stores blocked cells the agent has discovered so far.
        self.known_blocked = set()

        # Stores cells the agent has physically visited.
        self.visited_cells = set()

    def observe(self):
        """Observe adjacent cells and update known blocked cells."""
        # The agent can sense its immediate neighbors.
        # If any adjacent neighbor is truly blocked in the environment,
        # that cell gets added to the agent's knowledge base.
        neighbors = self.gridworld.get_neighbors(self.position)

        for neighbor in neighbors:
            if self.gridworld.is_blocked(neighbor):
                self.known_blocked.add(neighbor)

        # Mark the current cell as visited.
        self.visited_cells.add(self.position)

    def is_known_blocked(self, pos):
        """Check if a position is known to be blocked."""
        return pos in self.known_blocked

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        # Manhattan distance is used because movement is 4-directional.
        # It counts how many horizontal + vertical steps are needed
        # if there were no obstacles.
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class RepeatedForwardAStar:
    """Repeated Forward A* implementation."""

    def __init__(self, agent, tie_breaking='large_g'):
        # The planning agent that provides current position and known obstacles.
        self.agent = agent

        # Tie-breaking controls how A* chooses between states with the same f-value.
        # 'large_g' prefers states deeper from the start.
        # 'small_g' prefers states closer to the start.
        self.tie_breaking = tie_breaking

        # A simple counter for the number of searches started.
        self.counter = 0

        # Experimental statistics collected across the full run.
        self.stats = {
            'total_expansions': 0,
            'searches': 0,
            'path_length': 0
        }

    def heuristic(self, pos):
        """Manhattan distance heuristic."""
        # Forward A* estimates cost from the current cell to the goal.
        return self.agent.manhattan_distance(pos, self.agent.gridworld.goal)

    def compute_path(self, start, goal):
        """Run A* search from start to goal using agent's current knowledge"""
        ``
        # Initialize
        g_values = {start: 0}
        h = self.agent.manhattan_distance(start, goal)
        f_values = {start: h}
        
        # Calculate initial priority
        if self.tie_breaking == 'large_g':
            priority = f_values[start] * 100000 - g_values[start]
        else:  # small_g
            priority = f_values[start] * 100000 + g_values[start]
        
        open_list = [(priority, 0, start)]
        open_set = {start}
        open_dict = {start: 0}  # ⭐ NEW: Track counter for each cell in open list
        closed = set()
        tree = {}
        counter = 0
        
        # Main A* loop
        while open_list:
            _, pop_counter, current = heapq.heappop(open_list)
            
            # ⭐ NEW: Skip stale entries (old versions with worse priority)
            if current in open_dict and open_dict[current] != pop_counter:
                continue
            
            # Skip if already closed
            if current in closed:
                continue
            
            # Remove from open tracking
            open_set.discard(current)
            if current in open_dict:
                del open_dict[current]
            
            # Goal check
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node in tree:
                    path.append(node)
                    node = tree[node]
                path.append(start)
                path.reverse()
                return path, closed, g_values
            
            # Mark as expanded
            closed.add(current)
            self.stats['total_expansions'] += 1
            
            # Expand neighbors
            neighbors = self.agent.gridworld.get_neighbors(current)
            
            for neighbor in neighbors:
                # Skip if blocked or closed
                if self.agent.is_known_blocked(neighbor):
                    continue
                if neighbor in closed:
                    continue
                
                # Calculate tentative g
                tentative_g = g_values[current] + 1
                
                # Check if this is a better path
                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    # Update values
                    g_values[neighbor] = tentative_g
                    h = self.agent.manhattan_distance(neighbor, goal)
                    f_values[neighbor] = tentative_g + h
                    tree[neighbor] = current
                    
                    # Calculate priority
                    if self.tie_breaking == 'large_g':
                        priority = f_values[neighbor] * 100000 - g_values[neighbor]
                    else:  # small_g
                        priority = f_values[neighbor] * 100000 + g_values[neighbor]
                    
                    # Add to heap
                    counter += 1
                    heapq.heappush(open_list, (priority, counter, neighbor))
                    open_set.add(neighbor)
                    open_dict[neighbor] = counter  # ⭐ NEW: Track this counter
        
        # No path found
        return None, closed, g_values

    def find_path(self):
        """Main loop: repeatedly find paths until goal is reached or impossible."""
        # This method controls the full repeated-planning process:
        # 1. Observe nearby cells.
        # 2. Plan a path using current knowledge.
        # 3. Follow that path step by step.
        # 4. If new blocked information invalidates the remaining path, replan.

        # trajectory stores every real movement the agent makes in the world.
        trajectory = [self.agent.position]

        # all_expanded stores the expanded set from each separate search.
        all_expanded = []

        while self.agent.position != self.agent.gridworld.goal:
            # Before planning, update knowledge based on local observation.
            self.agent.observe()

            path, expanded, g_values = self.compute_path(
                self.agent.position,
                self.agent.gridworld.goal
            )

            all_expanded.append(expanded)

            # If planning fails, the goal is not reachable with the discovered map.
            if path is None:
                print("Cannot reach the target.")
                return trajectory, all_expanded, False

            # Try to execute the planned path in the real world.
            for i in range(1, len(path)):
                next_pos = path[i]

                # Move the agent one step forward.
                self.agent.position = next_pos
                trajectory.append(next_pos)
                self.stats['path_length'] += 1

                # Observe surroundings from the new position.
                self.agent.observe()

                # Stop immediately if the goal has been reached.
                if self.agent.position == self.agent.gridworld.goal:
                    print("Reached the target!")
                    return trajectory, all_expanded, True

                # If the next future step on the planned path is now known to be blocked,
                # abandon the rest of the plan and trigger replanning.
                if i + 1 < len(path) and self.agent.is_known_blocked(path[i + 1]):
                    break

        print("Reached the target!")
        return trajectory, all_expanded, True


class RepeatedBackwardAStar:
    """Repeated Backward A* implementation (searches from goal to start)."""

    def __init__(self, agent, tie_breaking='large_g'):
        # Same general setup as forward A*, but search is performed backward.
        self.agent = agent
        self.tie_breaking = tie_breaking
        self.counter = 0
        self.stats = {
            'total_expansions': 0,
            'searches': 0,
            'path_length': 0
        }

    def heuristic(self, pos, start):
        """Manhattan distance from pos to start."""
        # In backward A*, the search is directed toward the current start position,
        # so the heuristic is measured relative to start rather than goal.
        return self.agent.manhattan_distance(pos, start)

    def compute_path(self, start, goal):
        """Run A* search from goal to start (backward)."""
        self.counter += 1
        self.stats['searches'] += 1

        # If the current start is already known blocked, no path can start there.
        if self.agent.is_known_blocked(start):
            return None, set(), {}

        # Backward search begins at the goal and tries to reach the start.
        g_values = {goal: 0}
        f_values = {goal: self.heuristic(goal, start)}
        tree = {}
        closed = set()

        # Same priority rule as the forward version.
        if self.tie_breaking == 'large_g':
            priority = f_values[goal] * 100000 - g_values[goal]
        else:
            priority = f_values[goal] * 100000 + g_values[goal]

        open_list = [(priority, 0, goal)]
        open_set = {goal}
        item_counter = 1

        # Extra safety bound to avoid infinite loops if something goes wrong.
        max_iterations = self.agent.gridworld.size * self.agent.gridworld.size * 2
        iterations = 0

        while open_list and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_list)

            # Skip stale heap entries.
            if current not in open_set:
                continue
            open_set.remove(current)

            # If search reaches the start, reconstruct the path from start to goal.
            if current == start:
                path = [start]
                node = start

                # Because tree[child] = parent and we searched backward,
                # following parent pointers naturally produces start -> ... -> goal.
                while node in tree:
                    node = tree[node]
                    path.append(node)

                return path, closed, g_values

            # Ignore nodes already fully expanded.
            if current in closed:
                continue

            closed.add(current)
            self.stats['total_expansions'] += 1

            # Expand each valid neighbor.
            for neighbor in self.agent.gridworld.get_neighbors(current):
                # Do not plan through cells already known to be blocked.
                if self.agent.is_known_blocked(neighbor):
                    continue

                # Standard graph-search closed-list check.
                if neighbor in closed:
                    continue

                tentative_g = g_values[current] + 1

                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_values[neighbor] = tentative_g + self.heuristic(neighbor, start)
                    tree[neighbor] = current

                    if self.tie_breaking == 'large_g':
                        priority = f_values[neighbor] * 100000 - g_values[neighbor]
                    else:
                        priority = f_values[neighbor] * 100000 + g_values[neighbor]

                    heapq.heappush(open_list, (priority, item_counter, neighbor))
                    open_set.add(neighbor)
                    item_counter += 1

        return None, closed, g_values

    def find_path(self):
        """Main loop for backward A*."""
        trajectory = [self.agent.position]
        all_expanded = []

        while self.agent.position != self.agent.gridworld.goal:
            self.agent.observe()

            path, expanded, g_values = self.compute_path(
                self.agent.position,
                self.agent.gridworld.goal
            )

            all_expanded.append(expanded)

            if path is None:
                print("Cannot reach the target.")
                return trajectory, all_expanded, False

            # Track whether the agent successfully made progress this round.
            moved = False

            # Execute the planned route one step at a time.
            for i in range(1, len(path)):
                next_pos = path[i]

                # If the very next cell is now known blocked, the plan is unusable.
                if self.agent.is_known_blocked(next_pos):
                    break

                self.agent.position = next_pos
                trajectory.append(next_pos)
                self.stats['path_length'] += 1
                moved = True

                # After moving, gather new local information.
                self.agent.observe()

                if self.agent.position == self.agent.gridworld.goal:
                    print("Reached the target!")
                    return trajectory, all_expanded, True

                # If the upcoming next step becomes known blocked, stop and replan.
                if i + 1 < len(path) and self.agent.is_known_blocked(path[i + 1]):
                    break

            # If we could not move even one step, the algorithm treats it as failure.
            # In practice, this means the current plan immediately became unusable.
            if not moved:
                print("Cannot reach the target.")
                return trajectory, all_expanded, False

        print("Reached the target!")
        return trajectory, all_expanded, True


class AdaptiveAStar:
    """Adaptive A* with learning heuristics."""
    # Adaptive A* improves repeated searches by learning better heuristic values
    # from earlier searches. Over time, this can reduce the number of expansions.

    def __init__(self, agent, tie_breaking='large_g'):
        self.agent = agent
        self.tie_breaking = tie_breaking
        self.counter = 0

        # h_values stores learned heuristic values for cells expanded in prior searches.
        self.h_values = {}

        self.stats = {
            'total_expansions': 0,
            'searches': 0,
            'path_length': 0
        }

    def heuristic(self, pos):
        """Get heuristic value (learned or Manhattan)."""
        # If this state has a learned adaptive heuristic, use it.
        # Otherwise fall back to standard Manhattan distance.
        if pos in self.h_values:
            return self.h_values[pos]
        return self.agent.manhattan_distance(pos, self.agent.gridworld.goal)

    def compute_path(self, start, goal):
        """Run A* search and update heuristics."""
        self.counter += 1
        self.stats['searches'] += 1

        g_values = {start: 0}
        f_values = {start: self.heuristic(start)}
        tree = {}
        closed = set()

        if self.tie_breaking == 'large_g':
            priority = f_values[start] * 100000 - g_values[start]
        else:
            priority = f_values[start] * 100000 + g_values[start]

        open_list = [(priority, 0, start)]
        open_set = {start}
        item_counter = 1

        while open_list:
            _, _, current = heapq.heappop(open_list)

            # Ignore stale heap entries created by reinsertion updates.
            if current not in open_set:
                continue
            open_set.remove(current)

            if current == goal:
                # Reconstruct the successful path.
                path = []
                node = goal
                while node in tree:
                    path.append(node)
                    node = tree[node]
                path.append(start)
                path.reverse()

                # Adaptive A* update rule:
                # For every expanded state s,
                #     h_new(s) = g(goal) - g(s)
                #
                # Intuition:
                # This stores the exact distance from s to the goal
                # based on the latest search tree, making future searches smarter.
                if goal in g_values:
                    goal_g = g_values[goal]
                    for cell in closed:
                        if cell in g_values:
                            self.h_values[cell] = goal_g - g_values[cell]

                return path, closed, g_values

            # Same early stopping logic used in the forward version.
            if g_values.get(goal, float('inf')) <= f_values.get(current, float('inf')):
                break

            closed.add(current)
            self.stats['total_expansions'] += 1

            # Standard A* neighbor relaxation step.
            for neighbor in self.agent.gridworld.get_neighbors(current):
                if self.agent.is_known_blocked(neighbor):
                    continue

                tentative_g = g_values[current] + 1

                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_values[neighbor] = tentative_g + self.heuristic(neighbor)
                    tree[neighbor] = current

                    if self.tie_breaking == 'large_g':
                        priority = f_values[neighbor] * 100000 - g_values[neighbor]
                    else:
                        priority = f_values[neighbor] * 100000 + g_values[neighbor]

                    heapq.heappush(open_list, (priority, item_counter, neighbor))
                    open_set.add(neighbor)
                    item_counter += 1

        return None, closed, g_values

    def find_path(self):
        """Main loop for Adaptive A*."""
        trajectory = [self.agent.position]
        all_expanded = []

        while self.agent.position != self.agent.gridworld.goal:
            self.agent.observe()

            path, expanded, g_values = self.compute_path(
                self.agent.position,
                self.agent.gridworld.goal
            )

            all_expanded.append(expanded)

            if path is None:
                print("Cannot reach the target.")
                return trajectory, all_expanded, False

            # Execute the current best path until the goal is reached
            # or the remaining plan becomes invalid.
            for i in range(1, len(path)):
                next_pos = path[i]
                self.agent.position = next_pos
                trajectory.append(next_pos)
                self.stats['path_length'] += 1
                self.agent.observe()

                if self.agent.position == self.agent.gridworld.goal:
                    print("Reached the target!")
                    return trajectory, all_expanded, True

                if i + 1 < len(path) and self.agent.is_known_blocked(path[i + 1]):
                    break

        print("Reached the target!")
        return trajectory, all_expanded, True


def generate_and_save_environments(num_envs=30, size=51, output_dir='environments'):
    """Generate and save gridworld environments."""
    # Create the output folders if they do not already exist.
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/images', exist_ok=True)

    environments = []

    # Build the requested number of random environments.
    for i in range(num_envs):
        print(f"Generating environment {i+1}/{num_envs}")

        gw = GridWorld(size=size)
        gw.generate_maze_dfs(block_probability=0.3, seed=i)

        # Save the full GridWorld object so the same exact environment
        # can be reused later during experiments.
        with open(f'{output_dir}/env_{i}.pkl', 'wb') as f:
            pickle.dump(gw, f)

        # Save an image of the generated environment for quick inspection.
        gw.visualize(
            title=f'Environment {i}',
            save_path=f'{output_dir}/images/env_{i}.png'
        )

        environments.append(gw)

    print(f"Saved {num_envs} environments to {output_dir}/")
    return environments


def load_environment(env_id, env_dir='environments'):
    """Load a saved environment."""
    # Read the pickled GridWorld object from disk and return it.
    with open(f'{env_dir}/env_{env_id}.pkl', 'rb') as f:
        return pickle.load(f)


def run_experiments(num_envs=30, env_dir='environments'):
    """Run all experiments and collect statistics."""
    # results stores a list of statistics dictionaries for each algorithm/configuration.
    results = {
        'forward_large_g': [],
        'forward_small_g': [],
        'backward_large_g': [],
        'adaptive': []
    }

    for i in range(num_envs):
        print(f"\n{'='*60}")
        print(f"Testing Environment {i+1}/{num_envs}")
        print(f"{'='*60}")

        # Load a previously saved environment so all algorithms are tested fairly
        # on the exact same grid, start, and goal.
        gw = load_environment(i, env_dir)

        # ------------------------------------------------------------
        # Repeated Forward A* with large-g tie-breaking
        # ------------------------------------------------------------
        print("Forward A* (large-g)...", end=' ', flush=True)
        agent = Agent(gw)
        solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        print(f"✓ ({runtime:.2f}s)")

        results['forward_large_g'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })

        # ------------------------------------------------------------
        # Repeated Forward A* with small-g tie-breaking
        # ------------------------------------------------------------
        print("Forward A* (small-g)...", end=' ', flush=True)
        agent = Agent(gw)
        solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        print(f"✓ ({runtime:.2f}s)")

        results['forward_small_g'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })

        # ------------------------------------------------------------
        # Repeated Backward A*
        # ------------------------------------------------------------
        print("Backward A* (large-g)...", end=' ', flush=True)
        agent = Agent(gw)
        solver = RepeatedBackwardAStar(agent, tie_breaking='large_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        print(f"✓ ({runtime:.2f}s)")

        results['backward_large_g'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })

        # ------------------------------------------------------------
        # Adaptive A*
        # ------------------------------------------------------------
        print("Adaptive A* (large-g)...", end=' ', flush=True)
        agent = Agent(gw)
        solver = AdaptiveAStar(agent, tie_breaking='large_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        print(f"✓ ({runtime:.2f}s)")

        results['adaptive'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })

    return results


def print_statistics(results):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)

    for method, data in results.items():
        # Keep only successful runs when computing averages.
        successful = [d for d in data if d['success']]

        # Report the percentage of environments solved successfully.
        success_rate = len(successful) / len(data) * 100

        print(f"\n{method.upper()}:")
        print(f"  Success Rate: {success_rate:.1f}%")

        # If nothing succeeded, avoid dividing by zero and explain the result clearly.
        if len(successful) == 0:
            print(f"  No successful runs - algorithm failed on all {len(data)} environments")
        else:
            # Compute averages over only the successful runs.
            avg_expansions = np.mean([d['expansions'] for d in successful])
            avg_searches = np.mean([d['searches'] for d in successful])
            avg_path = np.mean([d['path_length'] for d in successful])
            avg_runtime = np.mean([d['runtime'] for d in successful])

            print(f"  Avg Expansions: {avg_expansions:.1f}")
            print(f"  Avg Searches: {avg_searches:.1f}")
            print(f"  Avg Path Length: {avg_path:.1f}")
            print(f"  Avg Runtime: {avg_runtime:.4f}s")


if __name__ == "__main__":
    # This script can be run directly to do the full workflow:
    # 1. Generate environments
    # 2. Run all algorithms on those environments
    # 3. Print summary statistics
    # 4. Save experiment results

    # Part 0: Generate environments
    print("Generating 30 gridworld environments...")
    generate_and_save_environments(num_envs=30, size=51)

    # Run experiments
    print("\nRunning experiments on all environments...")
    results = run_experiments(num_envs=30)

    # Print statistics
    print_statistics(results)

    # Save results to a pickle file for later analysis.
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nResults saved to results.pkl")