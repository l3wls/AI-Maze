#!/usr/bin/env python3
"""
Simple Demo Runner - Easy way to test the pathfinding algorithms

This file acts as a driver script that allows you to:
1. Run a single demo of A* on a grid
2. Generate animation frames of the algorithm
3. Compare different A* variations across multiple tests

It is essentially a user-friendly interface for your pathfinding system.
"""

# Standard library imports
import sys
import os

# Add current file directory to Python path so local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pathfinding components
from pathfinding import GridWorld, Agent, RepeatedForwardAStar, AdaptiveAStar

# Configure matplotlib for non-interactive use (no GUI pop-ups)
import matplotlib
matplotlib.use('Agg')

# Visualization + numerical libraries (not heavily used here but useful)
import matplotlib.pyplot as plt
import numpy as np


def simple_demo(size=10, seed=42):
    """
    Runs a single demonstration of pathfinding on a generated grid.

    This function:
    - Creates a grid world
    - Displays it in ASCII format
    - Runs multiple A* variants
    - Prints their performance metrics

    Parameters:
        size (int): Grid size (size x size)
        seed (int): Random seed for reproducibility
    """
    
    # Print header
    print("="*70)
    print(" "*20 + "A* PATHFINDING DEMO")
    print("="*70)
    print()
    
    # ---------------------------
    # Create gridworld
    # ---------------------------
    print(f"Creating {size}x{size} gridworld (seed: {seed})...")
    
    gw = GridWorld(size=size)
    
    # Generate maze using DFS with random obstacles
    gw.generate_maze_dfs(block_probability=0.3, seed=seed)
    
    # Print basic info
    print(f"  Start: {gw.start}")
    print(f"  Goal:  {gw.goal}")
    print(f"  Blocked cells: {int(gw.grid.sum())}/{size*size}")
    print()
    
    # ---------------------------
    # Display ASCII grid
    # ---------------------------
    # This helps visualize the grid in terminal without graphics
    print("Grid Layout:")
    print("-" * (size * 3 + 2))
    
    for i in range(size):
        row = "|"
        for j in range(size):
            if (i, j) == gw.start:
                row += " S"  # Start position
            elif (i, j) == gw.goal:
                row += " G"  # Goal position
            elif gw.grid[i, j] == 1:
                row += " █"  # Blocked cell
            else:
                row += " ·"  # Free cell
        row += " |"
        print(row)
    
    print("-" * (size * 3 + 2))
    print("S=Start, G=Goal, █=Blocked, ·=Unblocked")
    print()
    
    # ---------------------------
    # Run different algorithms
    # ---------------------------
    results = {}  # Store results for comparison
    
    for algo_name, algo_type in [
        ('Forward A* (large-g)', 'forward_large'),
        ('Forward A* (small-g)', 'forward_small'),
        ('Adaptive A*', 'adaptive')
    ]:
        print(f"Running {algo_name}...")
        
        # Create a fresh agent for each run
        agent = Agent(gw)
        
        # Select algorithm implementation
        if algo_type == 'forward_large':
            solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
        elif algo_type == 'forward_small':
            solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
        else:
            solver = AdaptiveAStar(agent, tie_breaking='large_g')
        
        # Run the algorithm
        trajectory, expanded, success = solver.find_path()
        
        # Check result
        if success:
            print(f"  ✓ Success!")
            print(f"    Expansions: {solver.stats['total_expansions']}")
            print(f"    Searches:   {solver.stats['searches']}")
            print(f"    Path length: {solver.stats['path_length']}")
            
            # Save stats for comparison
            results[algo_name] = solver.stats
        else:
            print(f"  ✗ Failed - no path found")
            results[algo_name] = None
        
        print()
    
    # ---------------------------
    # Print comparison table
    # ---------------------------
    print("="*70)
    print(" "*25 + "COMPARISON")
    print("="*70)
    
    # Table header
    print(f"{'Algorithm':<25} {'Expansions':<12} {'Searches':<10} {'Path Length':<12}")
    print("-"*70)
    
    # Print results row by row
    for algo_name, stats in results.items():
        if stats:
            print(f"{algo_name:<25} {stats['total_expansions']:<12} {stats['searches']:<10} {stats['path_length']:<12}")
        else:
            print(f"{algo_name:<25} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
    
    print("="*70)
    print()


def run_animation(size=10, seed=42):
    """
    Generates a step-by-step animation of the pathfinding process.

    This function calls the animation generator from another file.
    It produces image frames showing the agent's movement and search progress.
    """
    
    # Import here to avoid unnecessary dependency if not used
    from animate_pathfinding import generate_animation
    
    print("\nGenerating animation frames...")
    
    # Run animation generator
    generate_animation(size, seed, 'forward_large', 'demo_animation')
    
    print("\n✓ Animation frames saved to demo_animation/")


def compare_algorithms(size=15, num_tests=5):
    """
    Runs multiple tests to compare algorithm performance.

    This function:
    - Generates multiple random grids
    - Runs each algorithm on each grid
    - Computes average expansions and success rates

    Parameters:
        size (int): Grid size
        num_tests (int): Number of test runs
    """
    
    print("="*70)
    print(" "*20 + "ALGORITHM COMPARISON")
    print("="*70)
    print(f"\nTesting on {num_tests} different {size}x{size} grids...\n")
    
    # Store results for each algorithm
    all_results = {
        'forward_large': [],
        'forward_small': [],
        'adaptive': []
    }
    
    # ---------------------------
    # Run multiple test cases
    # ---------------------------
    for test_num in range(num_tests):
        print(f"Test {test_num + 1}/{num_tests}...", end=' ')
        
        # Generate new grid each time
        gw = GridWorld(size=size)
        gw.generate_maze_dfs(block_probability=0.3, seed=test_num)
        
        for algo_key, algo_type in [
            ('forward_large', 'forward_large'),
            ('forward_small', 'forward_small'),
            ('adaptive', 'adaptive')
        ]:
            agent = Agent(gw)
            
            # Choose solver
            if algo_type == 'forward_large':
                solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
            elif algo_type == 'forward_small':
                solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
            else:
                solver = AdaptiveAStar(agent, tie_breaking='large_g')
            
            # Run algorithm
            trajectory, expanded, success = solver.find_path()
            
            # Record results
            if success:
                all_results[algo_key].append(solver.stats['total_expansions'])
            else:
                all_results[algo_key].append(None)
        
        print("✓")
    
    # ---------------------------
    # Compute averages
    # ---------------------------
    print("\n" + "="*70)
    print(" "*25 + "RESULTS")
    print("="*70)
    
    print(f"{'Algorithm':<25} {'Avg Expansions':<20} {'Success Rate':<15}")
    print("-"*70)
    
    for algo_name, algo_key in [
        ('Forward A* (large-g)', 'forward_large'),
        ('Forward A* (small-g)', 'forward_small'),
        ('Adaptive A*', 'adaptive')
    ]:
        results = all_results[algo_key]
        
        # Filter valid results (ignore failures)
        valid = [r for r in results if r is not None]
        
        if valid:
            avg = sum(valid) / len(valid)
            success_rate = len(valid) / len(results) * 100
            print(f"{algo_name:<25} {avg:<20.1f} {success_rate:<15.0f}%")
        else:
            print(f"{algo_name:<25} {'N/A':<20} {0:<15.0f}%")
    
    print("="*70)
    print()


def main():
    """
    Entry point of the script.

    This function reads command-line arguments and decides
    which functionality to run:
    - demo
    - animate
    - compare
    """
    
    # If arguments are provided
    if len(sys.argv) > 1:
        
        if sys.argv[1] == 'demo':
            size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
            simple_demo(size, seed)
        
        elif sys.argv[1] == 'animate':
            size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
            run_animation(size, seed)
        
        elif sys.argv[1] == 'compare':
            size = int(sys.argv[2]) if len(sys.argv) > 2 else 15
            num = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            compare_algorithms(size, num)
        
        else:
            print("Unknown command!")
            print_help()
    
    # If no arguments → show help
    else:
        print_help()


def print_help():
    """
    Prints usage instructions for the script.

    This is shown when:
    - No arguments are provided
    - Invalid command is entered
    """
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║              A* PATHFINDING DEMO - QUICK START                      ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  COMMANDS:                                                          ║
║                                                                     ║
║  python demo.py demo [size] [seed]                                  ║
║    Run simple demo with ASCII output                                ║
║                                                                     ║
║  python demo.py animate [size] [seed]                               ║
║    Generate step-by-step animation frames                           ║
║                                                                     ║
║  python demo.py compare [size] [num_tests]                          ║
║    Compare algorithms on multiple grids                             ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝
""")


# Run program if executed directly
if __name__ == "__main__":
    main()