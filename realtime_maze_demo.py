#!/usr/bin/env python3
"""
Real-time maze generation demo.
Shows the DFS maze generation process step-by-step.
"""

import sys

# Add path so we can import the pathfinding module correctly
sys.path.insert(0, '/mnt/user-data/outputs')

from pathfinding import GridWorld

import matplotlib

# Use TkAgg so the animation window works properly
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import numpy as np
import random

# Hide toolbar for cleaner display during demo
plt.rcParams['toolbar'] = 'None'


class MazeGenerationVisualizer:
    """Visualize maze generation in real-time"""
    # This class handles building the maze step-by-step using DFS
    # and keeps track of everything needed for animation

    def __init__(self, size=51, block_probability=0.3, seed=None):
        # Basic setup for grid size \
        self.size = size
        self.block_probability = block_probability
        self.seed = seed
        
        # Start with all cells blocked (1 = blocked, 0 = open)
        self.grid = np.ones((size, size), dtype=int)
        
        # Set random seed so results can be reproduced if needed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Pick a random starting cell
        self.start = (random.randint(0, size-1), random.randint(0, size-1))
        self.goal = None
        
        # Track visited cells so DFS doesn't revisit them
        self.visited = set()
        
        # DFS stack used for backtracking
        self.stack = [self.start]

        # Mark starting cell as open
        self.grid[self.start[0], self.start[1]] = 0
        self.visited.add(self.start)
        
        # Track current animation state
        self.current_cell = self.start
        self.steps = []
        self.complete = False
    
    def get_unvisited_neighbors(self, pos):
        """Get unvisited neighbors of a position"""
        # Returns all neighbors that are inside the grid
        # and have not been visited yet

        neighbors = []
        row, col = pos
        
        # Check all 4 directions (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc

            # Make sure it's inside bounds and not visited yet
            if (0 <= new_row < self.size and 
                0 <= new_col < self.size and 
                (new_row, new_col) not in self.visited):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def step(self):
        """Perform one step of DFS maze generation"""
        # This function does ONe step of DFS:
        # either expand forward or backtrack

        if self.complete:
            return False
        
        # If stack is empty, DFS branch is done
        if not self.stack:

            # Find any remaining unvisited cells
            unvisited = [(i, j) for i in range(self.size) 
                        for j in range(self.size) 
                        if (i, j) not in self.visited]
            
            if not unvisited:
                # If everything is visited, choose a goal cell
                available = [(i, j) for i in range(self.size) 
                           for j in range(self.size) 
                           if self.grid[i, j] == 0 and (i, j) != self.start]
                if available:
                    self.goal = random.choice(available)

                self.complete = True
                return False
            
            # Restart DFS from a random unvisited cell
            new_start = random.choice(unvisited)
            self.stack = [new_start]

            self.grid[new_start[0], new_start[1]] = 0
            self.visited.add(new_start)
            self.current_cell = new_start
            return True
        
        # Current cell is top of the stack
        current = self.stack[-1]
        self.current_cell = current
        
        # Get all unvisited neighbors
        neighbors = self.get_unvisited_neighbors(current)
        
        if neighbors:
            # Pick one random neighbor to explore
            next_cell = random.choice(neighbors)
            
            # Mark it as visited
            self.visited.add(next_cell)
            
            # Randomly decide if it's blocked or open
            if random.random() < self.block_probability:
                self.grid[next_cell[0], next_cell[1]] = 1  # Blocked
            else:
                self.grid[next_cell[0], next_cell[1]] = 0  # Open

                # Only push open cells to continue DFS
                self.stack.append(next_cell)
        else:
            # No neighbors -> backtrack
            self.stack.pop()
        
        return True
    
    def generate_all_steps(self):
        """Generate all steps for animation"""
        # Runs DFS fully and stores each step
        

        steps = []
        while self.step():
            steps.append({
                'grid': self.grid.copy(),
                'current': self.current_cell,
                'stack_size': len(self.stack),
                'visited_count': len(self.visited)
            })
        
        # Add final state after generation is complete
        steps.append({
            'grid': self.grid.copy(),
            'current': None,
            'stack_size': 0,
            'visited_count': len(self.visited)
        })
        
        return steps


def animate_maze_generation(size=25, block_probability=0.3, seed=None, interval=50):
    """Animate the maze generation process"""
    # This function sets up the animation using matplotlib

    viz = MazeGenerationVisualizer(size, block_probability, seed)
    
    print("Generating maze steps...")
    steps = viz.generate_all_steps()
    print(f"Generated {len(steps)} steps")
    
    # Create plot figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def draw_frame(frame_num):
        # Draw one frame of the animation
        ax.clear()
        
        step = steps[frame_num]
        grid = step['grid']
        current = step['current']
        
        # Draw every cell
        for i in range(size):
            for j in range(size):
                if grid[i, j] == 1:
                    color = 'black'
                elif (i, j) == viz.start:
                    color = 'lightgreen'
                elif viz.goal and (i, j) == viz.goal:
                    color = 'lightcoral'
                elif current and (i, j) == current:
                    color = 'yellow'
                else:
                    color = 'white'
                
                rect = patches.Rectangle((j, size-1-i), 1, 1,
                                        linewidth=0.5,
                                        edgecolor='gray',
                                        facecolor=color)
                ax.add_patch(rect)
        
        # Draw start point
        if viz.start:
            start_y = size - 1 - viz.start[0]
            ax.plot(viz.start[1] + 0.5, start_y + 0.5, 'go',
                   markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Draw goal point (if chosen)
        if viz.goal:
            goal_y = size - 1 - viz.goal[0]
            ax.plot(viz.goal[1] + 0.5, goal_y + 0.5, 'r*',
                   markersize=15, markeredgecolor='darkred', markeredgewidth=2)
        
        # Configure axes
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')

        # Title shows progress info
        ax.set_title(f'DFS Maze Generation (Step {frame_num}/{len(steps)-1})\n'
                    f'Stack: {step["stack_size"]}, Visited: {step["visited_count"]}/{size*size}',
                    fontsize=14, fontweight='bold')
        
        ax.axis('off')
    
    # Create animation object
    anim = animation.FuncAnimation(fig, draw_frame, frames=len(steps),
                                  interval=interval, repeat=False)
    
    plt.tight_layout()
    return fig, anim


def generate_maze_instantly(size=51, block_probability=0.3, seed=None, show=True):
    """Generate a maze instantly and optionally display it"""
    # This skips animation and builds the maze immediately

    print(f"Generating {size}x{size} maze with {block_probability*100}% block probability...")
    
    gw = GridWorld(size)
    
    import time
    start_time = time.time()

    # Run DFS maze generation from GridWorld
    gw.generate_maze_dfs(block_probability=block_probability, seed=seed)

    elapsed = time.time() - start_time
    
    print(f"✓ Generated in {elapsed:.3f} seconds")
    print(f"  Start: {gw.start}")
    print(f"  Goal: {gw.goal}")
    
    # Count blocked cells for stats
    blocked_count = np.sum(gw.grid == 1)
    total_cells = size * size
    blocked_pct = (blocked_count / total_cells) * 100

    print(f"  Blocked cells: {blocked_count}/{total_cells} ({blocked_pct:.1f}%)")
    
    if show:
        # Draw the maze
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
        
        # Draw start
        start_y = size - 1 - gw.start[0]
        ax.plot(gw.start[1] + 0.5, start_y + 0.5, 'go',
               markersize=15 if size <= 30 else 8,
               markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Draw goal
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
    
    # CLI arguments so we can test different configs easily
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
        # Run instant generation mode
        gw = generate_maze_instantly(args.size, args.probability,
                                     args.seed, show=not args.instant)