"""
Interactive Pygame Visualizer for A* Pathfinding
Real-time grid display with step-by-step pathfinding visualization
"""

import sys
import os

# Add the current file's directory to Python's path.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
except ImportError:
    # If pygame is missing, print  message
    
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

# Color definitions used throughout the visualization.

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
    # This class handles the full visualization.
    # It is responsible for:
    # creating the window
    # drawing the grid and info panel
    # stepping through search and movement
    # handling keyboard controls

    def __init__(self, grid_size=15, cell_size=None, algorithm='forward_large', seed=None):
        # Start pygame 
        pygame.init()
        
        # Store the grid size s
        self.grid_size = grid_size
        
        # If no cell size is given, automatically choose one that fits the screen.
        if cell_size is None:
            # Get the screen resolution from pygame.
            display_info = pygame.display.Info()
            screen_width = display_info.current_w
            screen_height = display_info.current_h
            
            # Leave extra room for the side info panel
            # and also some space for borders/taskbar.
            available_width = screen_width - 500
            available_height = screen_height - 200
            
            # Figure out the biggest cell size that still lets the grid fit.
            max_cell_from_width = available_width // grid_size
            max_cell_from_height = available_height // grid_size
            
            # Use the smaller limit so it fits both directions.
            # Also clamp it so cells do not get too tiny or too huge.
            self.cell_size = max(5, min(40, min(max_cell_from_width, max_cell_from_height)))
            
            # Print useful info so we know what auto-sizing chose.
            print(f"Auto-adjusting for {grid_size}x{grid_size} grid:")
            print(f"  Screen size: {screen_width}x{screen_height}")
            print(f"  Cell size: {self.cell_size}px")
        else:
            # If a cell size was provided manually, just use that value.
            self.cell_size = cell_size
        
        # Save which algorithm version the visualizer should run.
        self.algorithm = algorithm
        
        # Compute the space used by the actual grid.
        self.grid_width = grid_size * self.cell_size

        # Reserve a side panel for stats, legend, and status text.
        self.info_width = 400

        # Total window width = grid area + info panel.
        self.window_width = self.grid_width + self.info_width

        # Add extra vertical room near the bottom for controls if needed.
        self.window_height = grid_size * self.cell_size + 100
        
        # Make sure the window does not end up larger than the screen.
        display_info = pygame.display.Info()
        if self.window_width > display_info.current_w - 50:
            self.window_width = display_info.current_w - 50
        if self.window_height > display_info.current_h - 100:
            self.window_height = display_info.current_h - 100
        
        # Create the main pygame window.
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f"A* Pathfinding - {algorithm}")
        
        # Create the gridworld and generate a random maze environment.
        self.gridworld = GridWorld(size=grid_size)
        self.gridworld.generate_maze_dfs(block_probability=0.3, seed=seed)
        
        # Create the agent that will move through the environment.
        self.agent = Agent(self.gridworld)
        
        # Choose which pathfinding algorithm object to use.
        if algorithm == 'forward_large':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='large_g')
        elif algorithm == 'forward_small':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='small_g')
        elif algorithm == 'backward_large':
            self.solver = RepeatedBackwardAStar(self.agent, tie_breaking='large_g')
        elif algorithm == 'adaptive':
            self.solver = AdaptiveAStar(self.agent, tie_breaking='large_g')
        
        # These variables keep track of the current simulation state.
        self.trajectory = [self.agent.position]
        self.current_path = None
        self.current_expanded = None
        self.search_count = 0
        self.step_count = 0
        self.finished = False
        self.paused = True
        self.speed = 5  # Steps per second
        
        # Fonts used for text on the side panel.
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Clock is used to control frame rate and animation timing.
        self.clock = pygame.time.Clock()
        
    def draw_cell(self, row, col, color, border=True):
        """Draw a single cell"""
        # Convert grid coordinates into screen pixel coordinates.
        x = col * self.cell_size
        y = row * self.cell_size
        
        # Draw the filled rectangle for this cell.
        pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
        
        # Optionally draw a border so the grid is easier to see.
        if border:
            pygame.draw.rect(self.screen, DARK_GRAY, 
                           (x, y, self.cell_size, self.cell_size), 1)
    
    def draw_grid(self):
        """Draw the entire grid"""
        # Go through every cell  and decide how it should be colored.
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = (i, j)
                
                # First decide the base color for the cell.
                if self.gridworld.is_blocked(pos):
                    if pos in self.agent.known_blocked:
                        # Blocked and already discovered by the agent.
                        color = BLACK
                    else:
                        # Blocked in the real map, but still unknown to the agent.
                        color = DARK_GRAY
                elif pos in self.agent.visited_cells:
                    # Open cell that the agent has already visited.
                    color = LIGHT_GREEN
                else:
                    # Open cell that has not been visited yet.
                    color = WHITE
                
                # If the cell was expanded in the latest search,
                # it will highlight it on top of the base color.
                if self.current_expanded and pos in self.current_expanded:
                    color = CYAN
                
                self.draw_cell(i, j, color)
        
        # Draw the trajectory as orange dots so we can see
        # all the places the agent has actually traveled through.
        for pos in self.trajectory:
            i, j = pos
            x = j * self.cell_size + self.cell_size // 2
            y = i * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, ORANGE, (x, y), self.cell_size // 6)
        
        # Draw the currently planned path as red line segments.
        # This shows the route the algorithm currently wants to follow.
        if self.current_path:
            for k in range(len(self.current_path) - 1):
                i1, j1 = self.current_path[k]
                i2, j2 = self.current_path[k + 1]
                x1 = j1 * self.cell_size + self.cell_size // 2
                y1 = i1 * self.cell_size + self.cell_size // 2
                x2 = j2 * self.cell_size + self.cell_size // 2
                y2 = i2 * self.cell_size + self.cell_size // 2
                pygame.draw.line(self.screen, RED, (x1, y1), (x2, y2), 3)
        
        # Draw the start cell as a green circle outline.
        i, j = self.gridworld.start
        x = j * self.cell_size + self.cell_size // 2
        y = i * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, GREEN, (x, y), self.cell_size // 3, 3)
        
        # Draw the goal cell as a red triangle so it looks different from start.
        i, j = self.gridworld.goal
        points = [
            (j * self.cell_size + self.cell_size // 2, i * self.cell_size + 5),
            (j * self.cell_size + 5, i * self.cell_size + self.cell_size - 5),
            (j * self.cell_size + self.cell_size - 5, i * self.cell_size + self.cell_size - 5)
        ]
        pygame.draw.polygon(self.screen, RED, points)
        
        # Draw the agent’s current position as a blue circle.
        i, j = self.agent.position
        x = j * self.cell_size + self.cell_size // 2
        y = i * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, BLUE, (x, y), self.cell_size // 2.5)
        
        # Highlight the adjacent cells the agent can currently observe.
        
        neighbors = self.gridworld.get_neighbors(self.agent.position)
        for ni, nj in neighbors:
            x = nj * self.cell_size
            y = ni * self.cell_size
            pygame.draw.rect(self.screen, YELLOW, 
                           (x, y, self.cell_size, self.cell_size), 2)
    
    def draw_info_panel(self):
        """Draw the information panel"""
        # The side panel starts right after the grid area.
        panel_x = self.grid_width
        panel_y = 0
        
        # Draw the background of the panel.
        pygame.draw.rect(self.screen, LIGHT_GRAY, 
                        (panel_x, panel_y, self.info_width, self.window_height))
        
        y = 20
        line_height = 30
        
        # Title at the top of the panel.
        title = self.font.render(f"A* Pathfinding", True, BLACK)
        self.screen.blit(title, (panel_x + 20, y))
        y += line_height + 10
        
        # Show the selected algorithm in a cleaner readable format.
        algo_name = self.algorithm.replace('_', ' ').title()
        text = self.small_font.render(f"Algorithm: {algo_name}", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height
        
        # Show the grid size.
        text = self.small_font.render(f"Grid: {self.grid_size}x{self.grid_size}", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height + 10
        
        # Separator line.
        pygame.draw.line(self.screen, DARK_GRAY, 
                        (panel_x + 10, y), (panel_x + self.info_width - 10, y), 2)
        y += 20
        
        # Statistics section.
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
        
        # Show how many states were expanded in the most recent search.
        if self.current_expanded:
            text = self.small_font.render(f"Last Expansions: {len(self.current_expanded)}", True, BLACK)
            self.screen.blit(text, (panel_x + 30, y))
        y += line_height + 10
        
        # Separator line.
        pygame.draw.line(self.screen, DARK_GRAY, 
                        (panel_x + 10, y), (panel_x + self.info_width - 10, y), 2)
        y += 20
        
        # Legend section.
        text = self.font.render("Legend:", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height
        
        # Each item explains one color or marker used in the grid.
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
            # Draw the small color box.
            pygame.draw.rect(self.screen, color, 
                           (panel_x + 30, y + 5, 15, 15))
            pygame.draw.rect(self.screen, BLACK, 
                           (panel_x + 30, y + 5, 15, 15), 1)

            # Draw the label next to it.
            text = self.small_font.render(label, True, BLACK)
            self.screen.blit(text, (panel_x + 50, y))
            y += 25
        
        y += 10
        
        # Separator before the status section.
        pygame.draw.line(self.screen, DARK_GRAY, 
                        (panel_x + 10, y), (panel_x + self.info_width - 10, y), 2)
        y += 20
        
        # Status section title.
        text = self.font.render("Status:", True, BLACK)
        self.screen.blit(text, (panel_x + 20, y))
        y += line_height
        
        # Decide what status message should be shown.
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
        # Controls were intentionally removed from the screen
        # to keep the display cleaner for demos.
        # The keyboard controls still work:
        # SPACE, S, R, +/-, Q
        pass
    
    def search_step(self):
        """Perform one A* search"""
        # If the run is already over, do nothing.
        if self.finished:
            return False
        
        # ffirst update what the agent knows about nearby cells.
        self.agent.observe()
        self.search_count += 1
        
        # Compute a path from the current agent position to the goal.
        path, expanded, g_values = self.solver.compute_path(
            self.agent.position,
            self.gridworld.goal
        )
        
        # Save expanded cells so they can be highlighted on the grid.
        self.current_expanded = expanded
        
        # If no path was found, mark the simulation as finished.
        if path is None:
            self.finished = True
            self.current_path = None
            return False
        
        # Otherwise save the planned path.
        self.current_path = path
        return True
    
    def move_step(self):
        """Move agent one step along path"""
        # If there is no usable path, movement cannot happen.
        if not self.current_path or len(self.current_path) <= 1:
            return False
        
        # Move to the next cell in the planned path.
        # Index 0 is current position, so index 1 is the next step.
        next_pos = self.current_path[1]
        self.agent.position = next_pos
        self.trajectory.append(next_pos)
        self.step_count += 1
        self.agent.observe()
        
        # Remove the old first position since the agent already moved.
        self.current_path = self.current_path[1:]
        
        # Stop if the goal has been reached.
        if self.agent.position == self.gridworld.goal:
            self.finished = True
            return False
        
        # If the next future step is now known blocked,
        # throw away the rest of the current plan and replan.
        if len(self.current_path) > 1:
            if self.agent.is_known_blocked(self.current_path[1]):
                self.current_path = None
                return False
        
        return True
    
    def reset(self):
        """Reset the simulation"""
        # Create a fresh agent on the same gridworld.
        self.agent = Agent(self.gridworld)
        
        # Recreate the solver too so its state is reset.
        if self.algorithm == 'forward_large':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='large_g')
        elif self.algorithm == 'forward_small':
            self.solver = RepeatedForwardAStar(self.agent, tie_breaking='small_g')
        elif self.algorithm == 'backward_large':
            self.solver = RepeatedBackwardAStar(self.agent, tie_breaking='large_g')
        elif self.algorithm == 'adaptive':
            self.solver = AdaptiveAStar(self.agent, tie_breaking='large_g')
        
        # Reset all tracking variables for a clean run.
        self.trajectory = [self.agent.position]
        self.current_path = None
        self.current_expanded = None
        self.search_count = 0
        self.step_count = 0
        self.finished = False
        self.paused = True
    
    def run(self):
        """Main game loop"""
        # This loop keeps running until the window is closed or quit is requested.
        running = True
        last_step_time = time.time()
        
        while running:
            # Handle window events and keyboard input.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Pause or unpause the simulation.
                        self.paused = not self.paused
                    
                    elif event.key == pygame.K_s:
                        # Step once manually.
                        # If there is no path, search first.
                        # Otherwise try to move one step.
                        if not self.finished:
                            if not self.current_path:
                                self.search_step()
                            else:
                                if not self.move_step():
                                    self.search_step()
                    
                    elif event.key == pygame.K_r:
                        # Reset the simulation.
                        self.reset()
                    
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Increase the auto-run speed.
                        self.speed = min(30, self.speed + 1)
                    
                    elif event.key == pygame.K_MINUS:
                        # Decrease the auto-run speed.
                        self.speed = max(1, self.speed - 1)
                    
                    elif event.key == pygame.K_q:
                        # Quit the program.
                        running = False
            
            # If not paused, automatically perform steps based on the speed setting.
            if not self.paused and not self.finished:
                current_time = time.time()
                if current_time - last_step_time >= 1.0 / self.speed:
                    if not self.current_path:
                        self.search_step()
                    else:
                        if not self.move_step():
                            self.search_step()
                    last_step_time = current_time
            
            # Clear the screen before drawing the next frame.
            self.screen.fill(WHITE)

            # Draw all visual pieces.
            self.draw_grid()
            self.draw_info_panel()
            self.draw_controls()
            
            # Update the display with the new frame.
            pygame.display.flip()

            # Limit rendering to 60 frames per second.
            self.clock.tick(60)
        
        # Cleanly shut down pygame when the loop ends.
        pygame.quit()


def main():
    """Main entry point"""
    import argparse
    
    # Command-line arguments make it easy to test different settings
    #  and without editing the file itself.
    parser = argparse.ArgumentParser(description='Interactive Pygame A* Pathfinding')
    parser.add_argument('--size', type=int, default=15, help='Grid size (default: 15)')
    parser.add_argument('--cell', type=int, default=None, 
                       help='Cell size in pixels (default: auto-adjust for screen)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--algorithm', type=str, default='forward_large',
                       choices=['forward_large', 'forward_small', 'backward_large', 'adaptive'],
                       help='Algorithm to use')
    
    # Parse command-line inputs.
    args = parser.parse_args()
    
    # Print a summary of the selected settings before starting.
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
    
    # Create the visualizer object using the chosen settings.
    viz = PygamePathfindingVisualizer(
        grid_size=args.size,
        cell_size=args.cell,
        algorithm=args.algorithm,
        seed=args.seed
    )

    # Start the main visualization loop.
    viz.run()


if __name__ == "__main__":
    
    main()