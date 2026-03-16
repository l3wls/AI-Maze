#!/usr/bin/env python3
"""
Visualize all saved environments as PNG images.
This creates viewable images from the .pkl files.
"""

import sys
import os
sys.path.insert(0, '/mnt/user-data/outputs')

from pathfinding import load_environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_environment(env_id, env_dir='environments', output_dir='environments/images'):
    """Create a PNG visualization of an environment"""
    
    # Load the environment
    gw = load_environment(env_id, env_dir)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    size = gw.size
    for i in range(size):
        for j in range(size):
            if gw.grid[i, j] == 1:  # Blocked
                rect = patches.Rectangle((j, size-1-i), 1, 1, 
                                        linewidth=0, 
                                        facecolor='black')
                ax.add_patch(rect)
            else:  # Unblocked
                rect = patches.Rectangle((j, size-1-i), 1, 1, 
                                        linewidth=0.5, 
                                        edgecolor='lightgray',
                                        facecolor='white')
                ax.add_patch(rect)
    
    # Mark start (green circle)
    start_y = size - 1 - gw.start[0]
    ax.plot(gw.start[1] + 0.5, start_y + 0.5, 'go', 
            markersize=15, label='Start', markeredgecolor='darkgreen', 
            markeredgewidth=2)
    
    # Mark goal (red star)
    goal_y = size - 1 - gw.goal[0]
    ax.plot(gw.goal[1] + 0.5, goal_y + 0.5, 'r*', 
            markersize=20, label='Goal', markeredgecolor='darkred', 
            markeredgewidth=2)
    
    # Set limits and appearance
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_title(f'Environment {env_id} ({size}x{size})', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/env_{env_id}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'✓ Created {output_path}')
    return output_path

def visualize_all_environments(num_envs=30, env_dir='environments'):
    """Visualize all environments"""
    
    print(f"Creating PNG visualizations for {num_envs} environments...")
    print("="*70)
    
    for i in range(num_envs):
        try:
            visualize_environment(i, env_dir)
        except FileNotFoundError:
            print(f'✗ Environment {i} not found - skipping')
        except Exception as e:
            print(f'✗ Error with environment {i}: {e}')
    
    print("="*70)
    print(f"Done! Check the {env_dir}/images/ folder for PNG files.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize saved environments')
    parser.add_argument('--num', type=int, default=30, 
                       help='Number of environments to visualize (default: 30)')
    parser.add_argument('--env-dir', type=str, default='environments',
                       help='Directory containing environment .pkl files')
    parser.add_argument('--single', type=int, default=None,
                       help='Visualize only a single environment by ID')
    
    args = parser.parse_args()
    
    if args.single is not None:
        visualize_environment(args.single, args.env_dir)
    else:
        visualize_all_environments(args.num, args.env_dir)
