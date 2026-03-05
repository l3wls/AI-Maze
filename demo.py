#!/usr/bin/env python3
"""
Simple Demo Runner - Easy way to test the pathfinding algorithms
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathfinding import GridWorld, Agent, RepeatedForwardAStar, AdaptiveAStar
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def simple_demo(size=10, seed=42):
    """Run a simple demo and show results"""
    
    print("="*70)
    print(" "*20 + "A* PATHFINDING DEMO")
    print("="*70)
    print()
    
    # Create gridworld
    print(f"Creating {size}x{size} gridworld (seed: {seed})...")
    gw = GridWorld(size=size)
    gw.generate_maze_dfs(block_probability=0.3, seed=seed)
    
    print(f"  Start: {gw.start}")
    print(f"  Goal:  {gw.goal}")
    print(f"  Blocked cells: {int(gw.grid.sum())}/{size*size}")
    print()
    
    # Show ASCII grid
    print("Grid Layout:")
    print("-" * (size * 3 + 2))
    for i in range(size):
        row = "|"
        for j in range(size):
            if (i, j) == gw.start:
                row += " S"
            elif (i, j) == gw.goal:
                row += " G"
            elif gw.grid[i, j] == 1:
                row += " █"
            else:
                row += " ·"
        row += " |"
        print(row)
    print("-" * (size * 3 + 2))
    print("S=Start, G=Goal, █=Blocked, ·=Unblocked")
    print()
    
    # Test algorithms
    results = {}
    
    for algo_name, algo_type in [
        ('Forward A* (large-g)', 'forward_large'),
        ('Forward A* (small-g)', 'forward_small'),
        ('Adaptive A*', 'adaptive')
    ]:
        print(f"Running {algo_name}...")
        
        agent = Agent(gw)
        if algo_type == 'forward_large':
            solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
        elif algo_type == 'forward_small':
            solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
        else:
            solver = AdaptiveAStar(agent, tie_breaking='large_g')
        
        trajectory, expanded, success = solver.find_path()
        
        if success:
            print(f"  ✓ Success!")
            print(f"    Expansions: {solver.stats['total_expansions']}")
            print(f"    Searches:   {solver.stats['searches']}")
            print(f"    Path length: {solver.stats['path_length']}")
            results[algo_name] = solver.stats
        else:
            print(f"  ✗ Failed - no path found")
            results[algo_name] = None
        print()
    
    # Summary
    print("="*70)
    print(" "*25 + "COMPARISON")
    print("="*70)
    print(f"{'Algorithm':<25} {'Expansions':<12} {'Searches':<10} {'Path Length':<12}")
    print("-"*70)
    
    for algo_name, stats in results.items():
        if stats:
            print(f"{algo_name:<25} {stats['total_expansions']:<12} {stats['searches']:<10} {stats['path_length']:<12}")
        else:
            print(f"{algo_name:<25} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
    
    print("="*70)
    print()


def run_animation(size=10, seed=42):
    """Generate step-by-step animation"""
    from animate_pathfinding import generate_animation
    
    print("\nGenerating animation frames...")
    generate_animation(size, seed, 'forward_large', 'demo_animation')
    print("\n✓ Animation frames saved to demo_animation/")


def compare_algorithms(size=15, num_tests=5):
    """Compare algorithms on multiple grids"""
    
    print("="*70)
    print(" "*20 + "ALGORITHM COMPARISON")
    print("="*70)
    print(f"\nTesting on {num_tests} different {size}x{size} grids...\n")
    
    all_results = {
        'forward_large': [],
        'forward_small': [],
        'adaptive': []
    }
    
    for test_num in range(num_tests):
        print(f"Test {test_num + 1}/{num_tests}...", end=' ')
        
        gw = GridWorld(size=size)
        gw.generate_maze_dfs(block_probability=0.3, seed=test_num)
        
        for algo_key, algo_type in [
            ('forward_large', 'forward_large'),
            ('forward_small', 'forward_small'),
            ('adaptive', 'adaptive')
        ]:
            agent = Agent(gw)
            if algo_type == 'forward_large':
                solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
            elif algo_type == 'forward_small':
                solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
            else:
                solver = AdaptiveAStar(agent, tie_breaking='large_g')
            
            trajectory, expanded, success = solver.find_path()
            
            if success:
                all_results[algo_key].append(solver.stats['total_expansions'])
            else:
                all_results[algo_key].append(None)
        
        print("✓")
    
    # Calculate averages
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
    """Main menu"""
    
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
    else:
        print_help()


def print_help():
    """Print usage help"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║              A* PATHFINDING DEMO - QUICK START                      ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  COMMANDS:                                                          ║
║                                                                     ║
║  python demo.py demo [size] [seed]                                  ║
║    Run simple demo with ASCII output                                ║
║    Example: python demo.py demo 10 42                               ║
║                                                                     ║
║  python demo.py animate [size] [seed]                               ║
║    Generate step-by-step animation frames                           ║
║    Example: python demo.py animate 15 100                           ║
║                                                                     ║
║  python demo.py compare [size] [num_tests]                          ║
║    Compare algorithms on multiple grids                             ║
║    Example: python demo.py compare 20 10                            ║
║                                                                     ║
╠════════════════════════════════════════════════════════════════════╣
║  DEFAULT USAGE (no args):                                           ║
║    python demo.py         → Shows this help                         ║
║                                                                     ║
║  TRY THESE:                                                         ║
║    python demo.py demo 5 42        → Small 5x5 grid                 ║
║    python demo.py demo 10 15       → Medium 10x10 grid              ║
║    python demo.py animate 10 15    → Animation of 10x10             ║
║    python demo.py compare 15 10    → Compare on 10 grids            ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
