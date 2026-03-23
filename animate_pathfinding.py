"""
Step-by-Step Animation Generator
Creates a series of images showing pathfinding progress
"""

# Standard library imports
import sys

# Add custom module path so Python can find your pathfinding implementation
sys.path.insert(0, '/home/claude')

# Import your pathfinding components
from pathfinding import GridWorld, Agent, RepeatedForwardAStar, AdaptiveAStar

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Numerical operations (used for handling paths as arrays)
import numpy as np

# File system operations (used for saving frames)
import os


def create_frame(gridworld, agent, trajectory, current_path, expanded_cells, 
                search_num, step_num, message, save_path):
    """
    Creates a single visualization frame of the pathfinding process.

    This function draws:
    - The grid world (including obstacles and fog of war)
    - The agent's current position
    - The goal and start positions
    - Cells that were expanded during search
    - The path taken so far (trajectory)
    - The current planned path (if available)
    - A side panel with statistics and status updates

    Parameters:
        gridworld: The environment containing grid layout and obstacles
        agent: The agent navigating the grid
        trajectory: List of positions the agent has visited so far
        current_path: The path returned by A* (planned path)
        expanded_cells: Cells explored during the current search
        search_num: Current search iteration number
        step_num: Total steps taken so far
        message: Status message to display
        save_path: File path to save the generated image
    """

    # Create figure with two panels:
    # Left = grid visualization, Right = info panel
    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(16, 8))

    # Overall title for the frame
    fig.suptitle(f'A* Pathfinding - Search #{search_num}, Step #{step_num}', 
                 fontsize=16, fontweight='bold')

    grid_size = gridworld.size
    
    # ---------------------------
    # Configure main grid display
    # ---------------------------
    ax_main.set_xlim(-0.5, grid_size - 0.5)
    ax_main.set_ylim(-0.5, grid_size - 0.5)
    ax_main.set_aspect('equal')  # Ensure square cells
    ax_main.invert_yaxis()       # Flip Y axis to match grid indexing
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title('Grid World with Fog of War')
    
    # ---------------------------
    # Draw each cell in the grid
    # ---------------------------
    for i in range(grid_size):
        for j in range(grid_size):
            pos = (i, j)
            
            if pos == agent.position:
                # Draw agent as a blue circle
                ax_main.add_patch(plt.Circle((j, i), 0.35, color='blue', zorder=5))
                ax_main.text(j, i, 'A', ha='center', va='center', 
                           fontweight='bold', fontsize=12, color='white', zorder=6)

            elif pos == gridworld.goal:
                # Draw goal as a red star
                ax_main.plot(j, i, 'r*', markersize=35, zorder=5)
                ax_main.text(j, i-0.5, 'GOAL', ha='center', va='top', 
                           fontweight='bold', fontsize=10, color='red')

            elif pos == gridworld.start:
                # Draw start (faded since agent may have moved)
                ax_main.plot(j, i, 'go', markersize=12, alpha=0.4, zorder=4)

            elif gridworld.is_blocked(pos):
                if pos in agent.known_blocked:
                    # Known obstacle → fully visible black
                    ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                    color='black', zorder=1))
                else:
                    # Unknown obstacle (fog of war)
                    ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                    color='#404040', alpha=0.7, zorder=1))
                    ax_main.text(j, i, '?', ha='center', va='center',
                               fontsize=20, color='gray', alpha=0.5)

            elif pos in agent.visited_cells:
                # Cells the agent has already visited
                ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                color='lightgreen', alpha=0.25, zorder=1))
    
    # ---------------------------
    # Highlight expanded cells
    # ---------------------------
    # These are the nodes A* explored during search
    if expanded_cells:
        for cell in expanded_cells:
            i, j = cell
            ax_main.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           color='cyan', alpha=0.35, zorder=2))
    
    # ---------------------------
    # Draw actual path taken
    # ---------------------------
    if len(trajectory) > 1:
        traj_array = np.array(trajectory)
        ax_main.plot(traj_array[:, 1], traj_array[:, 0], 
                    'orange', linewidth=4, alpha=0.6, zorder=3, 
                    label='Path Taken', marker='o', markersize=6)
    
    # ---------------------------
    # Draw current planned path
    # ---------------------------
    if current_path and len(current_path) > 1:
        path_array = np.array(current_path)
        ax_main.plot(path_array[:, 1], path_array[:, 0], 
                    'r--', linewidth=3, alpha=0.7, zorder=3, 
                    label='Planned Path', marker='s', markersize=4)
    
    # ---------------------------
    # Legend for visualization
    # ---------------------------
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
    
    # ---------------------------
    # Info panel (right side)
    # ---------------------------
    ax_info.axis('off')  # Hide axes
    
    # Text block showing stats and progress
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
    
    # Render info text box
    ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                verticalalignment='top', fontfamily='monospace',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save frame to file
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def generate_animation(grid_size=10, seed=42, algorithm='forward_large', output_dir='animation'):
    """
    Main driver function that runs the pathfinding simulation
    and generates a sequence of frames showing each step.

    This simulates:
    - Agent exploring unknown grid
    - Running A* repeatedly
    - Moving step-by-step
    - Replanning when obstacles are discovered
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print setup info
    print("="*60)
    print("GENERATING STEP-BY-STEP ANIMATION")
    print("="*60)

    # ---------------------------
    # Initialize environment
    # ---------------------------
    gridworld = GridWorld(size=grid_size)
    gridworld.generate_maze_dfs(block_probability=0.3, seed=seed)
    
    # Create agent
    agent = Agent(gridworld)
    
    # ---------------------------
    # Select algorithm
    # ---------------------------
    if algorithm == 'forward_large':
        solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
    elif algorithm == 'forward_small':
        solver = RepeatedForwardAStar(agent, tie_breaking='small_g')
    elif algorithm == 'adaptive':
        solver = AdaptiveAStar(agent, tie_breaking='large_g')
    
    # Track progress
    trajectory = [agent.position]  # where agent has been
    search_num = 0
    step_num = 0
    frame_num = 0
    
    # ---------------------------
    # Initial frame
    # ---------------------------
    create_frame(gridworld, agent, trajectory, None, None, 
                0, 0, "Initial state - Agent at start position",
                f"{output_dir}/frame_{frame_num:03d}.png")
    frame_num += 1
    
    # ---------------------------
    # Main loop (until goal reached)
    # ---------------------------
    while agent.position != gridworld.goal:
        agent.observe()  # update knowledge
        
        search_num += 1
        
        # Run A* search
        path, expanded, g_values = solver.compute_path(agent.position, gridworld.goal)
        
        # If no path exists
        if path is None:
            create_frame(gridworld, agent, trajectory, None, expanded,
                        search_num, step_num, "❌ No path found! Goal is unreachable.",
                        f"{output_dir}/frame_{frame_num:03d}.png")
            break
        
        # Frame after planning
        create_frame(gridworld, agent, trajectory, path, expanded,
                    search_num, step_num, 
                    f"Search #{search_num} complete - Found path",
                    f"{output_dir}/frame_{frame_num:03d}.png")
        
        # ---------------------------
        # Move along planned path
        # ---------------------------
        path_blocked = False
        
        for i in range(1, len(path)):
            next_pos = path[i]
            agent.position = next_pos
            trajectory.append(next_pos)
            step_num += 1
            agent.observe()
            
            # Determine message
            msg = f"Moving to {next_pos}"
            
            if agent.position == gridworld.goal:
                msg = "✓ GOAL REACHED!"
            elif i + 1 < len(path) and agent.is_known_blocked(path[i + 1]):
                # Path becomes invalid → must replan
                msg = f"⚠️ Path blocked! Replanning required."
                path_blocked = True
            
            # Create frame after movement
            create_frame(gridworld, agent, trajectory, path, expanded,
                        search_num, step_num, msg,
                        f"{output_dir}/frame_{frame_num:03d}.png")
            
            if agent.position == gridworld.goal or path_blocked:
                break
        
        if agent.position == gridworld.goal:
            break
    
    return frame_num


# ---------------------------
# Command line execution
# ---------------------------
if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate step-by-step pathfinding animation')
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', type=str, default='forward_large',
                       choices=['forward_large', 'forward_small', 'adaptive'])
    parser.add_argument('--output', type=str, default='animation')
    
    args = parser.parse_args()
    
    # Run animation generator
    generate_animation(args.size, args.seed, args.algorithm, args.output)