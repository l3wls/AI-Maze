"""
Step-by-Step Animation Generator
Creates a series of images showing pathfinding progress
"""

import sys
sys.path.insert(0, '/home/claude')
from pathfinding import GridWorld, Agent, RepeatedForwardAStar, AdaptiveAStar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import os


def create_frame(gridworld, agent, trajectory, current_path, expanded_cells, 
                search_num, step_num, message, save_path):
    """Create a single frame of the animation"""
    
    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'A* Pathfinding - Search #{search_num}, Step #{step_num}', 
                 fontsize=16, fontweight='bold')
    
    grid_size = gridworld.size
    
    # Main grid
    ax_main.set_xlim(-0.5, grid_size - 0.5)
    ax_main.set_ylim(-0.5, grid_size - 0.5)
    ax_main.set_aspect('equal')
    ax_main.invert_yaxis()
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title('Grid World with Fog of War')
    
    # Draw cells
    for i in range(grid_size):
        for j in range(grid_size):
            pos = (i, j)
            
            if pos == agent.position:
                # Current agent position
                ax_main.add_patch(plt.Circle((j, i), 0.35, color='blue', zorder=5))
                ax_main.text(j, i, 'A', ha='center', va='center', 
                           fontweight='bold', fontsize=12, color='white', zorder=6)
            elif pos == gridworld.goal:
                # Goal
                ax_main.plot(j, i, 'r*', markersize=35, zorder=5)
                ax_main.text(j, i-0.5, 'GOAL', ha='center', va='top', 
                           fontweight='bold', fontsize=10, color='red')
            elif pos == gridworld.start:
                # Start (faded)
                ax_main.plot(j, i, 'go', markersize=12, alpha=0.4, zorder=4)
            elif gridworld.is_blocked(pos):
                if pos in agent.known_blocked:
                    # Known blocked
                    ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                    color='black', zorder=1))
                else:
                    # Unknown (fog of war)
                    ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                    color='#404040', alpha=0.7, zorder=1))
                    ax_main.text(j, i, '?', ha='center', va='center',
                               fontsize=20, color='gray', alpha=0.5)
            elif pos in agent.visited_cells:
                # Visited
                ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                color='lightgreen', alpha=0.25, zorder=1))
    
    # Draw expanded cells
    if expanded_cells:
        for cell in expanded_cells:
            i, j = cell
            ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           color='cyan', alpha=0.35, zorder=2))
    
    # Draw trajectory
    if len(trajectory) > 1:
        traj_array = np.array(trajectory)
        ax_main.plot(traj_array[:, 1], traj_array[:, 0], 
                    'orange', linewidth=4, alpha=0.6, zorder=3, 
                    label='Path Taken', marker='o', markersize=6)
    
    # Draw planned path
    if current_path and len(current_path) > 1:
        path_array = np.array(current_path)
        ax_main.plot(path_array[:, 1], path_array[:, 0], 
                    'r--', linewidth=3, alpha=0.7, zorder=3, 
                    label='Planned Path', marker='s', markersize=4)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='blue', label='Agent (Current)'),
        mpatches.Patch(color='red', label='Goal'),
        mpatches.Patch(color='black', label='Known Blocked'),
        mpatches.Patch(color='#404040', label='Unknown (Fog)', alpha=0.7),
        mpatches.Patch(color='cyan', label='Expanded Cells', alpha=0.35),
        mpatches.Patch(color='lightgreen', label='Visited', alpha=0.25),
        mpatches.Patch(color='orange', label='Path Taken'),
        mpatches.Patch(color='red', label='Planned Path', alpha=0.7),
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Info panel
    ax_info.axis('off')
    
    info = f"""
═══════════════════════════════════════
        PATHFINDING PROGRESS
═══════════════════════════════════════

Grid Size: {grid_size}x{grid_size}
Current Position: {agent.position}
Goal Position: {gridworld.goal}
Manhattan Distance: {abs(agent.position[0]-gridworld.goal[0]) + abs(agent.position[1]-gridworld.goal[1])}

───────────────────────────────────────
Search #{search_num}
───────────────────────────────────────
Cells Expanded: {len(expanded_cells) if expanded_cells else 0}
Path Length: {len(current_path)-1 if current_path else 'N/A'}

───────────────────────────────────────
Overall Statistics
───────────────────────────────────────
Total Searches: {search_num}
Total Steps: {step_num}
Known Blocked Cells: {len(agent.known_blocked)}
Visited Cells: {len(agent.visited_cells)}

───────────────────────────────────────
Status
───────────────────────────────────────
{message}

═══════════════════════════════════════
"""
    
    ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                verticalalignment='top', fontfamily='monospace',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def generate_animation(grid_size=10, seed=42, algorithm='forward_large', output_dir='animation'):
    """Generate complete step-by-step animation"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("GENERATING STEP-BY-STEP ANIMATION")
    print("="*60)
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Algorithm: {algorithm}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}/")
    print()
    
    # Create gridworld
    gridworld = GridWorld(size=grid_size)
    gridworld.generate_maze_dfs(block_probability=0.3, seed=seed)
    
    print(f"Start: {gridworld.start}")
    print(f"Goal: {gridworld.goal}")
    print(f"Blocked cells: {int(gridworld.grid.sum())}")
    print()
    
    # Create agent
    agent = Agent(gridworld)
    
    # Choose algorithm
    if algorithm == 'forward_large':
        solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
        algo_name = "Forward A* (large-g)"
    elif algorithm == 'forward_small':
        solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
        algo_name = "Forward A* (small-g)"
    elif algorithm == 'adaptive':
        solver = AdaptiveAStar(agent, tie_breaking='large_g')
        algo_name = "Adaptive A*"
    
    print(f"Running {algo_name}...\n")
    
    trajectory = [agent.position]
    search_num = 0
    step_num = 0
    frame_num = 0
    
    # Initial frame
    create_frame(gridworld, agent, trajectory, None, None, 
                0, 0, "Initial state - Agent at start position",
                f"{output_dir}/frame_{frame_num:03d}.png")
    frame_num += 1
    print(f"Frame {frame_num}: Initial state")
    
    # Main loop
    while agent.position != gridworld.goal:
        agent.observe()
        search_num += 1
        
        # Search
        path, expanded, g_values = solver.compute_path(agent.position, gridworld.goal)
        
        if path is None:
            create_frame(gridworld, agent, trajectory, None, expanded,
                        search_num, step_num, "❌ No path found! Goal is unreachable.",
                        f"{output_dir}/frame_{frame_num:03d}.png")
            frame_num += 1
            print(f"Frame {frame_num}: No path found")
            break
        
        # Frame after search
        create_frame(gridworld, agent, trajectory, path, expanded,
                    search_num, step_num, 
                    f"Search #{search_num} complete - Found path of length {len(path)-1}",
                    f"{output_dir}/frame_{frame_num:03d}.png")
        frame_num += 1
        print(f"Frame {frame_num}: Search #{search_num} - found path (length {len(path)-1}, {len(expanded)} expansions)")
        
        # Move along path
        path_blocked = False
        for i in range(1, len(path)):
            next_pos = path[i]
            agent.position = next_pos
            trajectory.append(next_pos)
            step_num += 1
            agent.observe()
            
            # Frame after move
            msg = f"Moving to {next_pos}"
            if agent.position == gridworld.goal:
                msg = "✓ GOAL REACHED!"
            elif i + 1 < len(path) and agent.is_known_blocked(path[i + 1]):
                msg = f"⚠️ Path blocked at {path[i+1]}! Need to replan."
                path_blocked = True
            
            create_frame(gridworld, agent, trajectory, path, expanded,
                        search_num, step_num, msg,
                        f"{output_dir}/frame_{frame_num:03d}.png")
            frame_num += 1
            print(f"Frame {frame_num}: Step {step_num} - {msg}")
            
            if agent.position == gridworld.goal:
                break
            
            if path_blocked:
                break
        
        if agent.position == gridworld.goal:
            break
    
    # Final summary
    print()
    print("="*60)
    print("ANIMATION COMPLETE")
    print("="*60)
    print(f"Total Frames: {frame_num}")
    print(f"Total Searches: {search_num}")
    print(f"Total Steps: {step_num}")
    print(f"Final Path Length: {len(trajectory) - 1}")
    print(f"Total Expansions: {solver.stats['total_expansions']}")
    print()
    print(f"Frames saved to {output_dir}/")
    print(f"View them in sequence to see the pathfinding process!")
    
    return frame_num


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate step-by-step pathfinding animation')
    parser.add_argument('--size', type=int, default=10, help='Grid size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--algorithm', type=str, default='forward_large',
                       choices=['forward_large', 'forward_small', 'adaptive'])
    parser.add_argument('--output', type=str, default='animation', help='Output directory')
    
    args = parser.parse_args()
    
    generate_animation(args.size, args.seed, args.algorithm, args.output)
