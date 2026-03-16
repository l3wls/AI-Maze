#!/usr/bin/env python3
"""
Real-time maze generation demo.
Shows the DFS maze generation process step-by-step.
"""

import sys
sys.path.insert(0, '/mnt/user-data/outputs')

from pathfinding import GridWorld
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import random

# Hide matplotlib toolbar
plt.rcParams['toolbar'] = 'None'

class MazeGenerationVisualizer:
    """Visualize maze generation in real-time"""
    
    def __init__(self, size=51, block_probability=0.3, seed=None):
        self.size = size
        self.block_probability = block_probability
        self.seed = seed
        
        # Initialize grid (all blocked initially)
        self.grid = np.ones((size, size), dtype=int)
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Choose random start
        self.start = (random.randint(0, size-1), random.randint(0, size-1))
        self.goal = None
        
        # Track visited cells
        self.visited = set()
        
        # DFS stack
        self.stack = [self.start]
        self.grid[self.start[0], self.start[1]] = 0
        self.visited.add(self.start)
        
        # Animation state
        self.current_cell = self.start
        self.steps = []
        self.complete = False
    
    def get_unvisited_neighbors(self, pos):
        """Get unvisited neighbors of a position"""
        neighbors = []
        row, col = pos
        
        # All 4 directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.size and 
                0 <= new_col < self.size and 
                (new_row, new_col) not in self.visited):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def step(self):
        """Perform one step of DFS maze generation"""
        if self.complete:
            return False
        
        if not self.stack:
            # If stack is empty, check for unvisited cells
            unvisited = [(i, j) for i in range(self.size) 
                        for j in range(self.size) 
                        if (i, j) not in self.visited]
            
            if not unvisited:
                # All cells visited - choose goal
                available = [(i, j) for i in range(self.size) 
                           for j in range(self.size) 
                           if self.grid[i, j] == 0 and (i, j) != self.start]
                if available:
                    self.goal = random.choice(available)
                self.complete = True
                return False
            
            # Start from random unvisited cell
            new_start = random.choice(unvisited)
            self.stack = [new_start]
            self.grid[new_start[0], new_start[1]] = 0
            self.visited.add(new_start)
            self.current_cell = new_start
            return True
        
        # Get current cell
        current = self.stack[-1]
        self.current_cell = current
        
        # Get unvisited neighbors
        neighbors = self.get_unvisited_neighbors(current)
        
        if neighbors:
            # Choose random neighbor
            next_cell = random.choice(neighbors)
            
            # Mark as visited
            self.visited.add(next_cell)
            
            # Decide if blocked or unblocked
            if random.random() < self.block_probability:
                self.grid[next_cell[0], next_cell[1]] = 1  # Blocked
            else:
                self.grid[next_cell[0], next_cell[1]] = 0  # Unblocked
                self.stack.append(next_cell)  # Only add unblocked to stack
        else:
            # Backtrack
            self.stack.pop()
        
        return True
    
    def generate_all_steps(self):
        """Generate all steps for animation"""
        steps = []
        while self.step():
            steps.append({
                'grid': self.grid.copy(),
                'current': self.current_cell,
                'stack_size': len(self.stack),
                'visited_count': len(self.visited)
            })
        
        # Add final state
        steps.append({
            'grid': self.grid.copy(),
            'current': None,
            'stack_size': 0,
            'visited_count': len(self.visited)
        })
        
        return steps

def animate_maze_generation(size=25, block_probability=0.3, seed=None, interval=50):
    """Animate the maze generation process"""
    
    viz = MazeGenerationVisualizer(size, block_probability, seed)
    
    # Generate all steps
    print("Generating maze steps...")
    steps = viz.generate_all_steps()
    print(f"Generated {len(steps)} steps")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def draw_frame(frame_num):
        ax.clear()
        
        step = steps[frame_num]
        grid = step['grid']
        current = step['current']
        
        # Draw grid
        for i in range(size):
            for j in range(size):
                if grid[i, j] == 1:  # Blocked
                    color = 'black'
                elif (i, j) == viz.start:  # Start
                    color = 'lightgreen'
                elif viz.goal and (i, j) == viz.goal:  # Goal
                    color = 'lightcoral'
                elif current and (i, j) == current:  # Current cell
                    color = 'yellow'
                else:  # Unblocked
                    color = 'white'
                
                rect = patches.Rectangle((j, size-1-i), 1, 1, 
                                        linewidth=0.5, 
                                        edgecolor='gray',
                                        facecolor=color)
                ax.add_patch(rect)
        
        # Mark start
        if viz.start:
            start_y = size - 1 - viz.start[0]
            ax.plot(viz.start[1] + 0.5, start_y + 0.5, 'go', 
                   markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Mark goal (if determined)
        if viz.goal:
            goal_y = size - 1 - viz.goal[0]
            ax.plot(viz.goal[1] + 0.5, goal_y + 0.5, 'r*', 
                   markersize=15, markeredgecolor='darkred', markeredgewidth=2)
        
        # Set appearance
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.set_title(f'DFS Maze Generation (Step {frame_num}/{len(steps)-1})\n'
                    f'Stack: {step["stack_size"]}, Visited: {step["visited_count"]}/{size*size}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Create animation
    anim = animation.FuncAnimation(fig, draw_frame, frames=len(steps),
                                  interval=interval, repeat=False)
    
    plt.tight_layout()
    return fig, anim

def generate_maze_instantly(size=51, block_probability=0.3, seed=None, show=True):
    """Generate a maze instantly and optionally display it"""
    
    print(f"Generating {size}x{size} maze with {block_probability*100}% block probability...")
    
    gw = GridWorld(size)
    
    import time
    start_time = time.time()
    gw.generate_maze_dfs(block_probability=block_probability, seed=seed)
    elapsed = time.time() - start_time
    
    print(f"✓ Generated in {elapsed:.3f} seconds")
    print(f"  Start: {gw.start}")
    print(f"  Goal: {gw.goal}")
    
    blocked_count = np.sum(gw.grid == 1)
    total_cells = size * size
    blocked_pct = (blocked_count / total_cells) * 100
    print(f"  Blocked cells: {blocked_count}/{total_cells} ({blocked_pct:.1f}%)")
    
    if show:
        # Display the maze
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i in range(size):
            for j in range(size):
                if gw.grid[i, j] == 1:
                    color = 'black'
                elif (i, j) == gw.start:
                    color = 'lightgreen'
                elif (i, j) == gw.goal:
                    color = 'lightcoral'
                else:
                    color = 'white'
                
                rect = patches.Rectangle((j, size-1-i), 1, 1, 
                                        linewidth=0.1 if size > 30 else 0.5,
                                        edgecolor='gray' if size <= 30 else 'lightgray',
                                        facecolor=color)
                ax.add_patch(rect)
        
        # Mark start
        start_y = size - 1 - gw.start[0]
        ax.plot(gw.start[1] + 0.5, start_y + 0.5, 'go', 
               markersize=15 if size <= 30 else 8, 
               markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Mark goal
        goal_y = size - 1 - gw.goal[0]
        ax.plot(gw.goal[1] + 0.5, goal_y + 0.5, 'r*', 
               markersize=20 if size <= 30 else 10,
               markeredgecolor='darkred', markeredgewidth=2)
        
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.set_title(f'{size}x{size} Maze (Generated in {elapsed:.3f}s)\n'
                    f'{blocked_pct:.1f}% blocked', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    return gw

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time maze generation demo')
    parser.add_argument('--size', type=int, default=25, 
                       help='Grid size (default: 25)')
    parser.add_argument('--probability', type=float, default=0.3,
                       help='Block probability (default: 0.3)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--animate', action='store_true',
                       help='Show step-by-step animation')
    parser.add_argument('--interval', type=int, default=50,
                       help='Animation interval in ms (default: 50)')
    parser.add_argument('--instant', action='store_true',
                       help='Generate instantly without animation')
    
    args = parser.parse_args()
    
    if args.animate:
        print(f"Creating animation for {args.size}x{args.size} maze...")
        print("Close the window to exit.")
        fig, anim = animate_maze_generation(args.size, args.probability, 
                                            args.seed, args.interval)
        plt.show()
    else:
        # Instant generation
        gw = generate_maze_instantly(args.size, args.probability, 
                                     args.seed, show=not args.instant)