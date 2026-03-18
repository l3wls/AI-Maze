"""
Interactive Pygame Visualizer for A* Pathfinding
Real-time grid display with step-by-step pathfinding visualization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
except ImportError:
    print("=" * 70)
    print("ERROR: Pygame not installed!")
    print("=" * 70)
    print()
    print("To install pygame, run:")
    print("  pip install pygame")
    print()
    print("Or:")
    print("  pip install pygame --user")
    print()
    sys.exit(1)

from pathfinding import GridWorld, Agent, RepeatedForwardAStar, RepeatedBackwardAStar, AdaptiveAStar
import time

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
LIGHT_GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
PURPLE = (128, 0, 128)
LIGHT_GREEN = (144, 238, 144)
LIGHT_BLUE = (173, 216, 230)


class PygamePathfindingVisualizer:
    """Interactive Pygame visualization of A* pathfinding"""
    
    def __init__(self, grid_size=15, cell_size=None, algorithm='forward_large', seed=None):
        pygame.init()
        
        self.grid_size = grid_size
        
        # Auto-adjust cell size for large grids to fit on screen
        if cell_size is None:
            # Get screen resolution
            display_info = pygame.display.Info()
            screen_width = display_info.current_w
            screen_height = display_info.current_h
            
            # Leave room for info panel (400px) and taskbar/borders (100px)
            available_width = screen_width - 500
            available_height = screen_height - 200
            
            # Calculate cell size to fit screen
            max_cell_from_width = available_width // grid_size
            max_cell_from_height = available_height // grid_size
            
            # Use the smaller of the two, but keep between 5 and 40 pixels
            self.cell_size = max(5, min(40, min(max_cell_from_width, max_cell_from_height)))
            
            print(f"Auto-adjusting for {grid_size}x{grid_size} grid:")
            print(f"  Screen size: {screen_width}x{screen_height}")
            print(f"  Cell size: {self.cell_size}px")
        else:
            self.cell_size = cell_size
        
        self.algorithm = algorithm
        
        # Calculate window size
        self.grid_width = grid_size * self.cell_size
        self.info_width = 400
        self.window_width = self.grid_width + self.info_width
        self.window_height = grid_size * self.cell_size + 100  # Extra space for controls
        
        # Make sure window fits on screen
        display_info = pygame.display.Info()
        if self.window_width > display_info.current_w - 50:
            self.window_width = display_info.current_w - 50
        if self.window_height > display_info.current_h - 100:
            self.window_height = display_info.current_h - 100
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f"A* Pathfinding - {algorithm}")
        
        # Create gridworld
        self.gridworld = GridWorld(size=grid_size)
        self.gridworld.generate_maze_dfs(block_probability=0.3, seed=seed)
        
        # Create agent
        self.agent = Agent(self.gridworld)
        
        # Choose algorithm
        if algorithm == 'forward_large':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='large_g')
        elif algorithm == 'forward_small':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='small_g')
        elif algorithm == 'backward_large':
            self.solver = RepeatedBackwardAStar(self.agent, tie_breaking='large_g')
        elif algorithm == 'adaptive':
            self.solver = AdaptiveAStar(self.agent, tie_breaking='large_g')
        
        # State
        self.trajectory = [self.agent.position]
        self.current_path = None
        self.current_expanded = None
        self.search_count = 0
        self.step_count = 0
        self.finished = False
        self.paused = True
        self.speed = 5  # Steps per second
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Clock for timing
        self.clock = pygame.time.Clock()
        
    def draw_cell(self, row, col, color, border=True):
        """Draw a single cell"""
        x = col * self.cell_size
        y = row * self.cell_size
        
        pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
        
        if border:
            pygame.draw.rect(self.screen, DARK_GRAY, 
                           (x, y, self.cell_size, self.cell_size), 1)
    
    def draw_grid(self):
        """Draw the entire grid"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = (i, j)
                
                # Determine base color
                if self.gridworld.is_blocked(pos):
                    if pos in self.agent.known_blocked:
                        # Known blocked
                        color = BLACK
                    else:
                        # Unknown (fog of war)
                        color = DARK_GRAY
                elif pos in self.agent.visited_cells:
                    # Visited
                    color = LIGHT_GREEN
                else:
                    # Unvisited
                    color = WHITE
                
                # Draw expanded cells overlay
                if self.current_expanded and pos in self.current_expanded:
                    color = CYAN
                
                self.draw_cell(i, j, color)
        
        # Draw trajectory
        for pos in self.trajectory:
            i, j = pos
            x = j * self.cell_size + self.cell_size // 2
            y = i * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, ORANGE, (x, y), self.cell_size // 6)
        
        # Draw planned path
        if self.current_path:
            for k in range(len(self.current_path) - 1):
                i1, j1 = self.current_path[k]
                i2, j2 = self.current_path[k + 1]
                x1 = j1 * self.cell_size + self.cell_size // 2
                y1 = i1 * self.cell_size + self.cell_size // 2
                x2 = j2 * self.cell_size + self.cell_size // 2
                y2 = i2 * self.cell_size + self.cell_size // 2
                pygame.draw.line(self.screen, RED, (x1, y1), (x2, y2), 3)
        
        # Draw start
        i, j = self.gridworld.start
        x = j * self.cell_size + self.cell_size // 2
        y = i * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, GREEN, (x, y), self.cell_size // 3, 3)
        
        # Draw goal
        i, j = self.gridworld.goal
        points = [
            (j * self.cell_size + self.cell_size // 2, i * self.cell_size + 5),
            (j * self.cell_size + 5, i * self.cell_size + self.cell_size - 5),
            (j * self.cell_size + self.cell_size - 5, i * self.cell_size + self.cell_size - 5)
        ]
        pygame.draw.polygon(self.screen, RED, points)
        
        # Draw agent (current position)
        i, j = self.agent.position
        x = j * self.cell_size + self.cell_size // 2
        y = i * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, BLUE, (x, y), self.cell_size // 2.5)
        
        # Draw agent "vision" (shows what it can see)
        neighbors = self.gridworld.get_neighbors(self.agent.position)
        for ni, nj in neighbors:
            x = nj * self.cell_size
            y = ni * self.cell_size
            pygame.draw.rect(self.screen, YELLOW, 
                           (x, y, self.cell_size, self.cell_size), 2)
    
    def draw_info_panel(self):
        """Draw the information panel"""
        panel_x = self.grid_width
        panel_y = 0
        
        # Background
        pygame.draw.rect(self.screen, LIGHT_GRAY, 
                        (panel_x, panel_y, self.info_width, self.window_height))
        
        y = 20
        line_height = 30
        
        # Title
        title = self.font.render(f"A* Pathfinding", True, BLACK)
        self.screen.blit(title, (panel_x + 20, y))
        y += line_height + 10
        
        # Algorithm
        algo_name = self.algorithm.replace('_', ' ').title()
        text = self.small_font.render(f"Algorithm: {algo_name}", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height
        
        # Grid info
        text = self.small_font.render(f"Grid: {self.grid_size}x{self.grid_size}", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height + 10
        
        # Separator
        pygame.draw.line(self.screen, DARK_GRAY, 
                        (panel_x + 10, y), (panel_x + self.info_width - 10, y), 2)
        y += 20
        
        # Statistics
        text = self.font.render("Statistics:", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height
        
        text = self.small_font.render(f"Searches: {self.search_count}", True, BLACK)
        self.screen.blit(text, (panel_x + 30, y))
        y += line_height
        
        text = self.small_font.render(f"Steps: {self.step_count}", True, BLACK)
        self.screen.blit(text, (panel_x + 30, y))
        y += line_height
        
        text = self.small_font.render(f"Known Blocked: {len(self.agent.known_blocked)}", True, BLACK)
        self.screen.blit(text, (panel_x + 30, y))
        y += line_height
        
        if self.current_expanded:
            text = self.small_font.render(f"Last Expansions: {len(self.current_expanded)}", True, BLACK)
            self.screen.blit(text, (panel_x + 30, y))
        y += line_height + 10
        
        # Separator
        pygame.draw.line(self.screen, DARK_GRAY, 
                        (panel_x + 10, y), (panel_x + self.info_width - 10, y), 2)
        y += 20
        
        # Legend
        text = self.font.render("Legend:", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height
        
        # Draw legend items
        legends = [
            (BLUE, "Agent (Current)"),
            (GREEN, "Start"),
            (RED, "Goal"),
            (BLACK, "Known Blocked"),
            (DARK_GRAY, "Unknown (Fog)"),
            (CYAN, "Expanded"),
            (LIGHT_GREEN, "Visited"),
            (ORANGE, "Trajectory"),
        ]
        
        for color, label in legends:
            # Draw color box
            pygame.draw.rect(self.screen, color, 
                           (panel_x + 30, y + 5, 15, 15))
            pygame.draw.rect(self.screen, BLACK, 
                           (panel_x + 30, y + 5, 15, 15), 1)
            # Draw label
            text = self.small_font.render(label, True, BLACK)
            self.screen.blit(text, (panel_x + 50, y))
            y += 25
        
        y += 10
        
        # Separator
        pygame.draw.line(self.screen, DARK_GRAY, 
                        (panel_x + 10, y), (panel_x + self.info_width - 10, y), 2)
        y += 20
        
        # Status
        text = self.font.render("Status:", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height
        
        if self.finished:
            if self.agent.position == self.gridworld.goal:
                status_text = "GOAL REACHED!"
                status_color = GREEN
            else:
                status_text = "No Path Found"
                status_color = RED
        elif self.paused:
            status_text = "Paused"
            status_color = ORANGE
        else:
            status_text = "Running..."
            status_color = BLUE
        
        text = self.small_font.render(status_text, True, status_color)
        self.screen.blit(text, (panel_x + 30, y))
    
    def draw_controls(self):
        """Draw control instructions at bottom"""
        # Controls removed - cleaner display for demos
        # Controls still work: SPACE, S, R, +/-, Q
        pass
    
    def search_step(self):
        """Perform one A* search"""
        if self.finished:
            return False
        
        self.agent.observe()
        self.search_count += 1
        
        path, expanded, g_values = self.solver.compute_path(
            self.agent.position,
            self.gridworld.goal
        )
        
        self.current_expanded = expanded
        
        if path is None:
            self.finished = True
            self.current_path = None
            return False
        
        self.current_path = path
        return True
    
    def move_step(self):
        """Move agent one step along path"""
        if not self.current_path or len(self.current_path) <= 1:
            return False
        
        # Move to next position in path
        next_pos = self.current_path[1]  # Index 1 is next step
        self.agent.position = next_pos
        self.trajectory.append(next_pos)
        self.step_count += 1
        self.agent.observe()
        
        # Update path (remove first element)
        self.current_path = self.current_path[1:]
        
        # Check if goal reached
        if self.agent.position == self.gridworld.goal:
            self.finished = True
            return False
        
        # Check if next step is blocked
        if len(self.current_path) > 1:
            if self.agent.is_known_blocked(self.current_path[1]):
                # Need to replan
                self.current_path = None
                return False
        
        return True
    
    def reset(self):
        """Reset the simulation"""
        self.agent = Agent(self.gridworld)
        
        if self.algorithm == 'forward_large':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='large_g')
        elif self.algorithm == 'forward_small':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='small_g')
        elif self.algorithm == 'backward_large':
            self.solver = RepeatedBackwardAStar(self.agent, tie_breaking='large_g')
        elif self.algorithm == 'adaptive':
            self.solver = AdaptiveAStar(self.agent, tie_breaking='large_g')
        
        self.trajectory = [self.agent.position]
        self.current_path = None
        self.current_expanded = None
        self.search_count = 0
        self.step_count = 0
        self.finished = False
        self.paused = True
    
    def run(self):
        """Main game loop"""
        running = True
        last_step_time = time.time()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Toggle pause
                        self.paused = not self.paused
                    
                    elif event.key == pygame.K_s:
                        # Step once
                        if not self.finished:
                            if not self.current_path:
                                self.search_step()
                            else:
                                if not self.move_step():
                                    self.search_step()
                    
                    elif event.key == pygame.K_r:
                        # Reset
                        self.reset()
                    
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Speed up
                        self.speed = min(30, self.speed + 1)
                    
                    elif event.key == pygame.K_MINUS:
                        # Slow down
                        self.speed = max(1, self.speed - 1)
                    
                    elif event.key == pygame.K_q:
                        # Quit
                        running = False
            
            # Auto-step if not paused
            if not self.paused and not self.finished:
                current_time = time.time()
                if current_time - last_step_time >= 1.0 / self.speed:
                    if not self.current_path:
                        self.search_step()
                    else:
                        if not self.move_step():
                            self.search_step()
                    last_step_time = current_time
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_info_panel()
            self.draw_controls()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Pygame A* Pathfinding')
    parser.add_argument('--size', type=int, default=15, help='Grid size (default: 15)')
    parser.add_argument('--cell', type=int, default=None, 
                       help='Cell size in pixels (default: auto-adjust for screen)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--algorithm', type=str, default='forward_large',
                       choices=['forward_large', 'forward_small', 'backward_large', 'adaptive'],
                       help='Algorithm to use')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PYGAME A* PATHFINDING VISUALIZER")
    print("=" * 70)
    print(f"\nGrid Size: {args.size}x{args.size}")
    if args.cell:
        print(f"Cell Size: {args.cell} pixels (manual)")
    else:
        print(f"Cell Size: Auto-adjusted to fit your screen")
    print(f"Algorithm: {args.algorithm}")
    print(f"Seed: {args.seed if args.seed else 'Random'}")
    print()
    print("Starting visualization...")
    print("=" * 70)
    
    viz = PygamePathfindingVisualizer(
        grid_size=args.size,
        cell_size=args.cell,  # None = auto-adjust
        algorithm=args.algorithm,
        seed=args.seed
    )
    viz.run()


if __name__ == "__main__":
    main()