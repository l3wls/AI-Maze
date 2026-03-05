"""
AI Pathfinding Project: Repeated A* and Adaptive A* Implementation
Implements various A* algorithms for gridworld navigation with partial observability
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time
import random
from collections import defaultdict
import pickle
import os


class GridWorld:
    """Represents a gridworld environment with blocked and unblocked cells"""
    
    def __init__(self, size=51):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # 0 = unblocked, 1 = blocked
        self.start = None
        self.goal = None
        
    def generate_maze_dfs(self, block_probability=0.3, seed=None):
        """Generate maze using depth-first search with random tie-breaking"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initially all cells are unvisited (set to blocked)
        self.grid = np.ones((self.size, self.size), dtype=int)
        visited = np.zeros((self.size, self.size), dtype=bool)
        
        # Start from random cell
        start_x = random.randint(0, self.size - 1)
        start_y = random.randint(0, self.size - 1)
        
        stack = [(start_x, start_y)]
        visited[start_x, start_y] = True
        self.grid[start_x, start_y] = 0  # Mark as unblocked
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while np.sum(visited) < self.size * self.size:
            if len(stack) == 0:
                # Find unvisited cell to start new branch
                unvisited = np.argwhere(~visited)
                if len(unvisited) == 0:
                    break
                idx = random.randint(0, len(unvisited) - 1)
                current = tuple(unvisited[idx])
                stack.append(current)
                visited[current[0], current[1]] = True
                self.grid[current[0], current[1]] = 0
                continue
            
            current = stack[-1]
            
            # Get unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and not visited[nx, ny]:
                    neighbors.append((nx, ny))
            
            if len(neighbors) == 0:
                # Dead end, backtrack
                stack.pop()
            else:
                # Choose random neighbor
                next_cell = neighbors[random.randint(0, len(neighbors) - 1)]
                visited[next_cell[0], next_cell[1]] = True
                
                # With 30% probability mark as blocked, 70% as unblocked
                if random.random() < block_probability:
                    self.grid[next_cell[0], next_cell[1]] = 1
                else:
                    self.grid[next_cell[0], next_cell[1]] = 0
                    stack.append(next_cell)
        
        # Ensure start and goal are unblocked and reachable
        self._set_start_goal()
    
    def _set_start_goal(self):
        """Set start and goal positions in unblocked cells"""
        unblocked = np.argwhere(self.grid == 0)
        if len(unblocked) < 2:
            # If not enough unblocked cells, force some
            self.grid[0, 0] = 0
            self.grid[self.size-1, self.size-1] = 0
            self.start = (0, 0)
            self.goal = (self.size - 1, self.size - 1)
        else:
            idx1 = random.randint(0, len(unblocked) - 1)
            idx2 = random.randint(0, len(unblocked) - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, len(unblocked) - 1)
            self.start = tuple(unblocked[idx1])
            self.goal = tuple(unblocked[idx2])
    
    def is_blocked(self, pos):
        """Check if a position is blocked"""
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True
        return self.grid[x, y] == 1
    
    def get_neighbors(self, pos):
        """Get valid neighbors (4-connected)"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors
    
    def visualize(self, path=None, expanded=None, title="GridWorld", save_path=None):
        """Visualize the gridworld"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create visualization grid
        vis_grid = np.copy(self.grid).astype(float)
        
        # Color expanded cells
        if expanded is not None:
            for cell in expanded:
                if vis_grid[cell[0], cell[1]] == 0:
                    vis_grid[cell[0], cell[1]] = 0.5
        
        # Create custom colormap
        cmap = ListedColormap(['white', 'black', 'lightblue'])
        ax.imshow(vis_grid, cmap=cmap, origin='upper')
        
        # Draw path
        if path is not None and len(path) > 0:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='Path')
        
        # Mark start and goal
        if self.start:
            ax.plot(self.start[1], self.start[0], 'go', markersize=15, label='Start')
        if self.goal:
            ax.plot(self.goal[1], self.goal[0], 'r*', markersize=20, label='Goal')
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig


class Agent:
    """Agent that navigates the gridworld with limited visibility"""
    
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.position = gridworld.start
        self.known_blocked = set()  # Set of known blocked cells
        self.visited_cells = set()
        
    def observe(self):
        """Observe adjacent cells and update known blocked cells"""
        neighbors = self.gridworld.get_neighbors(self.position)
        for neighbor in neighbors:
            if self.gridworld.is_blocked(neighbor):
                self.known_blocked.add(neighbor)
        self.visited_cells.add(self.position)
    
    def is_known_blocked(self, pos):
        """Check if a position is known to be blocked"""
        return pos in self.known_blocked
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class RepeatedForwardAStar:
    """Repeated Forward A* implementation"""
    
    def __init__(self, agent, tie_breaking='large_g'):
        self.agent = agent
        self.tie_breaking = tie_breaking  # 'large_g' or 'small_g'
        self.counter = 0
        self.stats = {
            'total_expansions': 0,
            'searches': 0,
            'path_length': 0
        }
        
    def heuristic(self, pos):
        """Manhattan distance heuristic"""
        return self.agent.manhattan_distance(pos, self.agent.gridworld.goal)
    
    def compute_path(self, start, goal):
        """Run A* search from start to goal"""
        self.counter += 1
        self.stats['searches'] += 1
        
        # Initialize
        g_values = {start: 0}
        f_values = {start: self.heuristic(start)}
        tree = {}
        closed = set()
        
        # Priority queue: (priority, counter, g_value, position)
        # Priority depends on tie-breaking strategy
        if self.tie_breaking == 'large_g':
            # Break ties in favor of larger g-values
            priority = f_values[start] * 100000 - g_values[start]
        else:  # small_g
            # Break ties in favor of smaller g-values
            priority = f_values[start] * 100000 + g_values[start]
        
        open_list = [(priority, 0, start)]
        open_set = {start}
        item_counter = 1
        
        while open_list:
            _, _, current = heapq.heappop(open_list)
            
            if current not in open_set:
                continue
            open_set.remove(current)
            
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
            
            if g_values.get(goal, float('inf')) <= f_values.get(current, float('inf')):
                break
            
            closed.add(current)
            self.stats['total_expansions'] += 1
            
            # Expand neighbors
            for neighbor in self.agent.gridworld.get_neighbors(current):
                if self.agent.is_known_blocked(neighbor):
                    continue
                
                tentative_g = g_values[current] + 1
                
                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_values[neighbor] = tentative_g + self.heuristic(neighbor)
                    tree[neighbor] = current
                    
                    if neighbor in open_set:
                        # Update priority (by removing and reinserting)
                        pass  # Handled by checking open_set membership
                    
                    if self.tie_breaking == 'large_g':
                        priority = f_values[neighbor] * 100000 - g_values[neighbor]
                    else:
                        priority = f_values[neighbor] * 100000 + g_values[neighbor]
                    
                    heapq.heappush(open_list, (priority, item_counter, neighbor))
                    open_set.add(neighbor)
                    item_counter += 1
        
        return None, closed, g_values
    
    def find_path(self):
        """Main loop: repeatedly find paths until goal is reached or impossible"""
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
            
            # Move along path until obstacle found or goal reached
            for i in range(1, len(path)):
                next_pos = path[i]
                self.agent.position = next_pos
                trajectory.append(next_pos)
                self.stats['path_length'] += 1
                self.agent.observe()
                
                if self.agent.position == self.agent.gridworld.goal:
                    print("Reached the target!")
                    return trajectory, all_expanded, True
                
                # Check if path ahead is blocked
                if i + 1 < len(path) and self.agent.is_known_blocked(path[i + 1]):
                    break
        
        print("Reached the target!")
        return trajectory, all_expanded, True


class RepeatedBackwardAStar:
    """Repeated Backward A* implementation (searches from goal to start)"""
    
    def __init__(self, agent, tie_breaking='large_g'):
        self.agent = agent
        self.tie_breaking = tie_breaking
        self.counter = 0
        self.stats = {
            'total_expansions': 0,
            'searches': 0,
            'path_length': 0
        }
        
    def heuristic(self, pos, start):
        """Manhattan distance from pos to start"""
        return self.agent.manhattan_distance(pos, start)
    
    def compute_path(self, start, goal):
        """Run A* search from goal to start (backward)"""
        self.counter += 1
        self.stats['searches'] += 1
        
        # Search from goal to start
        g_values = {goal: 0}
        f_values = {goal: self.heuristic(goal, start)}
        tree = {}
        closed = set()
        
        if self.tie_breaking == 'large_g':
            priority = f_values[goal] * 100000 - g_values[goal]
        else:
            priority = f_values[goal] * 100000 + g_values[goal]
        
        open_list = [(priority, 0, goal)]
        open_set = {goal}
        item_counter = 1
        
        while open_list:
            _, _, current = heapq.heappop(open_list)
            
            if current not in open_set:
                continue
            open_set.remove(current)
            
            if current == start:
                # Reconstruct path
                path = []
                node = start
                while node in tree:
                    path.append(node)
                    node = tree[node]
                path.append(goal)
                path.reverse()
                return path, closed, g_values
            
            if g_values.get(start, float('inf')) <= f_values.get(current, float('inf')):
                break
            
            closed.add(current)
            self.stats['total_expansions'] += 1
            
            # Expand neighbors
            for neighbor in self.agent.gridworld.get_neighbors(current):
                if self.agent.is_known_blocked(neighbor):
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
        """Main loop for backward A*"""
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
            
            # Move along path
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


class AdaptiveAStar:
    """Adaptive A* with learning heuristics"""
    
    def __init__(self, agent, tie_breaking='large_g'):
        self.agent = agent
        self.tie_breaking = tie_breaking
        self.counter = 0
        self.h_values = {}  # Learned heuristics
        self.stats = {
            'total_expansions': 0,
            'searches': 0,
            'path_length': 0
        }
        
    def heuristic(self, pos):
        """Get heuristic value (learned or Manhattan)"""
        if pos in self.h_values:
            return self.h_values[pos]
        return self.agent.manhattan_distance(pos, self.agent.gridworld.goal)
    
    def compute_path(self, start, goal):
        """Run A* search and update heuristics"""
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
            
            if current not in open_set:
                continue
            open_set.remove(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node in tree:
                    path.append(node)
                    node = tree[node]
                path.append(start)
                path.reverse()
                
                # Update heuristics for all expanded cells
                if goal in g_values:
                    goal_g = g_values[goal]
                    for cell in closed:
                        if cell in g_values:
                            self.h_values[cell] = goal_g - g_values[cell]
                
                return path, closed, g_values
            
            if g_values.get(goal, float('inf')) <= f_values.get(current, float('inf')):
                break
            
            closed.add(current)
            self.stats['total_expansions'] += 1
            
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
        """Main loop for Adaptive A*"""
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
    """Generate and save gridworld environments"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    
    environments = []
    for i in range(num_envs):
        print(f"Generating environment {i+1}/{num_envs}")
        gw = GridWorld(size=size)
        gw.generate_maze_dfs(block_probability=0.3, seed=i)
        
        # Save environment
        with open(f'{output_dir}/env_{i}.pkl', 'wb') as f:
            pickle.dump(gw, f)
        
        # Save visualization
        gw.visualize(title=f'Environment {i}', 
                    save_path=f'{output_dir}/images/env_{i}.png')
        
        environments.append(gw)
    
    print(f"Saved {num_envs} environments to {output_dir}/")
    return environments


def load_environment(env_id, env_dir='environments'):
    """Load a saved environment"""
    with open(f'{env_dir}/env_{env_id}.pkl', 'rb') as f:
        return pickle.load(f)


def run_experiments(num_envs=30, env_dir='environments'):
    """Run all experiments and collect statistics"""
    
    results = {
        'forward_large_g': [],
        'forward_small_g': [],
        'backward_large_g': [],
        'adaptive': []
    }
    
    for i in range(num_envs):
        print(f"\n{'='*60}")
        print(f"Testing Environment {i}")
        print(f"{'='*60}")
        
        gw = load_environment(i, env_dir)
        
        # Test Forward A* with large-g
        print("\nForward A* (large-g)...")
        agent = Agent(gw)
        solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        results['forward_large_g'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })
        
        # Test Forward A* with small-g
        print("Forward A* (small-g)...")
        agent = Agent(gw)
        solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        results['forward_small_g'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })
        
        # Test Backward A*
        print("Backward A* (large-g)...")
        agent = Agent(gw)
        solver = RepeatedBackwardAStar(agent, tie_breaking='large_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        results['backward_large_g'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })
        
        # Test Adaptive A*
        print("Adaptive A* (large-g)...")
        agent = Agent(gw)
        solver = AdaptiveAStar(agent, tie_breaking='large_g')
        start_time = time.time()
        trajectory, expanded, success = solver.find_path()
        runtime = time.time() - start_time
        results['adaptive'].append({
            'expansions': solver.stats['total_expansions'],
            'searches': solver.stats['searches'],
            'path_length': solver.stats['path_length'],
            'runtime': runtime,
            'success': success
        })
    
    return results


def print_statistics(results):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    for method, data in results.items():
        successful = [d for d in data if d['success']]
        if len(successful) == 0:
            continue
        
        avg_expansions = np.mean([d['expansions'] for d in successful])
        avg_searches = np.mean([d['searches'] for d in successful])
        avg_path = np.mean([d['path_length'] for d in successful])
        avg_runtime = np.mean([d['runtime'] for d in successful])
        success_rate = len(successful) / len(data) * 100
        
        print(f"\n{method.upper()}:")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Expansions: {avg_expansions:.1f}")
        print(f"  Avg Searches: {avg_searches:.1f}")
        print(f"  Avg Path Length: {avg_path:.1f}")
        print(f"  Avg Runtime: {avg_runtime:.4f}s")


if __name__ == "__main__":
    # Part 0: Generate environments
    print("Generating 30 gridworld environments...")
    generate_and_save_environments(num_envs=30, size=51)
    
    # Run experiments
    print("\nRunning experiments on all environments...")
    results = run_experiments(num_envs=30)
    
    # Print statistics
    print_statistics(results)
    
    # Save results
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to results.pkl")
