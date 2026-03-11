"""
Visualization and Demonstration Script
Shows step-by-step execution of different A* variants
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathfinding import *
import pickle


def visualize_step_by_step(gridworld, agent_class, algorithm_name, save_dir='visualizations'):
    """Create step-by-step visualization of algorithm execution"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/{algorithm_name}', exist_ok=True)
    
    agent = Agent(gridworld)
    
    if agent_class == 'forward_large_g':
        solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
    elif agent_class == 'forward_small_g':
        solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
    elif agent_class == 'backward':
        solver = RepeatedBackwardAStar(agent, tie_breaking='large_g')
    elif agent_class == 'adaptive':
        solver = AdaptiveAStar(agent, tie_breaking='large_g')
    
    # Custom execution with visualization
    trajectory = [agent.position]
    search_num = 0
    
    while agent.position != gridworld.goal:
        agent.observe()
        
        path, expanded, g_values = solver.compute_path(agent.position, gridworld.goal)
        search_num += 1
        
        if path is None:
            print(f"Cannot reach target in {algorithm_name}")
            break
        
        # Visualize this search
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Create base grid
        vis_grid = np.copy(gridworld.grid).astype(float)
        
        # Mark expanded cells
        for cell in expanded:
            if vis_grid[cell[0], cell[1]] == 0:
                vis_grid[cell[0], cell[1]] = 0.3
        
        # Create colormap
        cmap = ListedColormap(['white', 'black', 'lightblue'])
        ax.imshow(vis_grid, cmap=cmap, origin='upper', alpha=0.8)
        
        # Draw path
        if len(path) > 0:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=3, label='Planned Path')
        
        # Draw trajectory so far
        if len(trajectory) > 0:
            traj_array = np.array(trajectory)
            ax.plot(traj_array[:, 1], traj_array[:, 0], 'orange', linewidth=2, 
                   linestyle='--', label='Trajectory')
        
        # Mark current position, start, goal
        ax.plot(agent.position[1], agent.position[0], 'bo', markersize=15, label='Current')
        ax.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
        ax.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
        
        # Add g-values and f-values for expanded cells
        for cell in list(expanded)[:20]:  # Limit to avoid clutter
            if cell in g_values:
                g = g_values[cell]
                h = solver.heuristic(cell) if hasattr(solver, 'heuristic') else \
                    agent.manhattan_distance(cell, gridworld.goal)
                f = g + h
                ax.text(cell[1]-0.3, cell[0]-0.2, f'g:{g}', fontsize=6, color='blue')
                ax.text(cell[1]-0.3, cell[0]+0.2, f'f:{f}', fontsize=6, color='red')
        
        ax.set_title(f'{algorithm_name} - Search #{search_num}\n'
                    f'Expansions: {len(expanded)}, Path Length: {len(path)}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f'{save_dir}/{algorithm_name}/search_{search_num}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Move along path
        path_blocked = False
        for i in range(1, len(path)):
            next_pos = path[i]
            agent.position = next_pos
            trajectory.append(next_pos)
            agent.observe()
            
            if agent.position == gridworld.goal:
                break
            
            if i + 1 < len(path) and agent.is_known_blocked(path[i + 1]):
                path_blocked = True
                break
        
        if agent.position == gridworld.goal:
            break
    
    # Final visualization
    fig, ax = plt.subplots(figsize=(12, 12))
    vis_grid = np.copy(gridworld.grid).astype(float)
    cmap = ListedColormap(['white', 'black'])
    ax.imshow(vis_grid, cmap=cmap, origin='upper')
    
    if len(trajectory) > 0:
        traj_array = np.array(trajectory)
        ax.plot(traj_array[:, 1], traj_array[:, 0], 'r-', linewidth=3, label='Final Path')
    
    ax.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
    ax.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
    ax.set_title(f'{algorithm_name} - Final Result\n'
                f'Total Searches: {search_num}, Total Moves: {len(trajectory)-1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f'{save_dir}/{algorithm_name}/final.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualizations to {save_dir}/{algorithm_name}/")


def create_comparison_plot(results, save_path='comparison.png'):
    """Create comparison plots for all algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(results.keys())
    
    # Extract data for successful runs
    data = {}
    for method in methods:
        successful = [d for d in results[method] if d['success']]
        data[method] = {
            'expansions': [d['expansions'] for d in successful],
            'searches': [d['searches'] for d in successful],
            'path_length': [d['path_length'] for d in successful],
            'runtime': [d['runtime'] for d in successful]
        }
    
    # Plot 1: Average Expansions
    ax = axes[0, 0]
    avg_expansions = [np.mean(data[m]['expansions']) for m in methods]
    ax.bar(range(len(methods)), avg_expansions, color=['blue', 'green', 'orange', 'red'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Average Expansions')
    ax.set_title('Cell Expansions Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average Searches
    ax = axes[0, 1]
    avg_searches = [np.mean(data[m]['searches']) for m in methods]
    ax.bar(range(len(methods)), avg_searches, color=['blue', 'green', 'orange', 'red'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Average Searches')
    ax.set_title('Number of Searches Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average Path Length
    ax = axes[1, 0]
    avg_path = [np.mean(data[m]['path_length']) for m in methods]
    ax.bar(range(len(methods)), avg_path, color=['blue', 'green', 'orange', 'red'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Average Path Length')
    ax.set_title('Path Length Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Average Runtime
    ax = axes[1, 1]
    avg_runtime = [np.mean(data[m]['runtime']) * 1000 for m in methods]  # Convert to ms
    ax.bar(range(len(methods)), avg_runtime, color=['blue', 'green', 'orange', 'red'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Average Runtime (ms)')
    ax.set_title('Runtime Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def demo_single_environment(env_id=0):
    """Run complete demo on a single environment"""
    print(f"\n{'='*80}")
    print(f"DEMONSTRATION ON ENVIRONMENT {env_id}")
    print(f"{'='*80}\n")
    
    gw = load_environment(env_id)
    
    print(f"Grid size: {gw.size}x{gw.size}")
    print(f"Start: {gw.start}")
    print(f"Goal: {gw.goal}")
    print(f"Blocked cells: {np.sum(gw.grid)}")
    
    # Visualize all algorithms
    algorithms = [
        ('forward_large_g', 'Forward A* (Large-g)'),
        ('forward_small_g', 'Forward A* (Small-g)'),
        ('backward', 'Backward A*'),
        ('adaptive', 'Adaptive A*')
    ]
    
    for algo_class, algo_name in algorithms:
        print(f"\nRunning {algo_name}...")
        visualize_step_by_step(gw, algo_class, algo_name)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        env_id = int(sys.argv[1])
    else:
        env_id = 0
    
    # Run demonstration
    demo_single_environment(env_id)
    
    # Load and visualize results
    try:
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
        create_comparison_plot(results)
    except FileNotFoundError:
        print("No results.pkl found. Run pathfinding.py first.")
