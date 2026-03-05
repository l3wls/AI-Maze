# 🎮 Pygame Interactive Visualizer - Setup Guide

## What is This?

**`pygame_visualizer.py`** creates a **real-time interactive window** showing:
- ✨ Live grid with A* pathfinding happening in real-time
- 🎨 Beautiful graphics with colors showing agent, path, obstacles
- 🕹️ Interactive controls (play/pause, step-by-step, speed control)
- 👁️ Fog of war visualization (agent discovers obstacles as it moves)
- 📊 Live statistics panel

## Installation

### Step 1: Install Pygame

```bash
pip install pygame
```

Or if that doesn't work:

```bash
pip install pygame --user
```

Or:

```bash
python -m pip install pygame
```

### Step 2: Verify Installation

```bash
python -c "import pygame; print('Pygame installed successfully!')"
```

## How to Run

### Basic Usage (15x15 grid):
```bash
python pygame_visualizer.py
```

### Custom Grid Size:
```bash
python pygame_visualizer.py --size 20
```

### Smaller Cells (bigger grid on screen):
```bash
python pygame_visualizer.py --size 30 --cell 25
```

### Try Different Algorithms:
```bash
# Forward A* with large-g (best performance)
python pygame_visualizer.py --algorithm forward_large

# Forward A* with small-g (usually fails)
python pygame_visualizer.py --algorithm forward_small

# Adaptive A* (learning algorithm)
python pygame_visualizer.py --algorithm adaptive
```

### Use Specific Seed (reproducible maze):
```bash
python pygame_visualizer.py --size 20 --seed 42
```

## 🎮 Controls

Once the window opens:

| Key | Action |
|-----|--------|
| **SPACE** | Play/Pause the simulation |
| **S** | Step once (move one step forward) |
| **R** | Reset to beginning |
| **F** | Toggle Fog of War (cheat mode to see all obstacles) |
| **+** or **=** | Speed up |
| **-** | Slow down |
| **Q** | Quit |

## 🎨 What the Colors Mean

| Color | Meaning |
|-------|---------|
| 🔵 **Blue Circle** | Agent (current position) |
| 🟢 **Green Circle** | Start position |
| 🔺 **Red Triangle** | Goal |
| ⬛ **Black** | Known blocked cells |
| ⬛ **Dark Gray** | Unknown cells (fog of war) |
| 🟦 **Cyan** | Cells expanded in last search |
| 🟩 **Light Green** | Visited cells |
| 🟠 **Orange Dots** | Agent's trajectory (path taken) |
| 🔴 **Red Line** | Planned path |
| 🟡 **Yellow Border** | Agent's vision (what it can see) |

## 📊 Info Panel (Right Side)

Shows real-time statistics:
- **Algorithm** being used
- **Grid size**
- **Number of searches** performed
- **Total steps** taken
- **Known blocked cells**
- **Expansions in last search**

## Example Commands

### Small & Fast (Good for Testing):
```bash
python pygame_visualizer.py --size 10 --seed 42
```

### Medium (Recommended):
```bash
python pygame_visualizer.py --size 15 --seed 15
```

### Large Grid:
```bash
python pygame_visualizer.py --size 25 --cell 30 --seed 100
```

### Compare Algorithms:
```bash
# Run this 3 times with different algorithms
python pygame_visualizer.py --size 15 --seed 42 --algorithm forward_large
python pygame_visualizer.py --size 15 --seed 42 --algorithm forward_small
python pygame_visualizer.py --size 15 --seed 42 --algorithm adaptive
```

## 💡 Tips

### Tip 1: Watch It Step-by-Step
1. Run the program
2. Press **S** repeatedly to step through one move at a time
3. Watch how the agent discovers obstacles and replans

### Tip 2: See All Obstacles (Cheat Mode)
1. Run the program
2. Press **F** to turn off fog of war
3. See the entire maze layout

### Tip 3: Perfect for Presentations
1. Use a medium size: `--size 15 --cell 40`
2. Press **SPACE** to start/stop at key moments
3. Press **S** to step through important decisions
4. Toggle fog (**F**) to show/hide the full maze

### Tip 4: Compare Performance
Use the same seed with different algorithms:
```bash
python pygame_visualizer.py --seed 42 --algorithm forward_large
# Watch and note the number of expansions

python pygame_visualizer.py --seed 42 --algorithm adaptive
# Compare - Adaptive should expand fewer cells!
```

## 🐛 Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame
```

### Window doesn't open
Make sure you're running from terminal/command prompt, not from a script editor.

### "pygame.error: No available video device"
You need a display. This won't work on headless servers. Use the non-pygame versions instead:
```bash
python demo.py demo 15 42
python demo.py animate 15 42
```

### Black screen
Just wait a second - the grid is generating. Press **SPACE** to start.

## 🎬 Typical Workflow

```bash
# 1. Start with small grid to understand controls
python pygame_visualizer.py --size 10

# 2. Press SPACE to start
# 3. Watch it run automatically
# 4. Press SPACE to pause
# 5. Press S to step through
# 6. Press F to toggle fog
# 7. Press R to reset
# 8. Press Q to quit

# 9. Try a bigger grid
python pygame_visualizer.py --size 20 --seed 100

# 10. Compare algorithms
python pygame_visualizer.py --size 15 --seed 42 --algorithm adaptive
```

## ✨ What Makes This Cool

1. **Interactive**: You control the pace
2. **Visual**: See exactly what the algorithm is thinking
3. **Educational**: Perfect for understanding A* step-by-step
4. **Real-time**: Watch the fog of war in action
5. **Comparative**: Easy to compare different algorithms

## 📝 For Your Project Demo

1. **Start with a clean grid:**
   ```bash
   python pygame_visualizer.py --size 15 --seed 42
   ```

2. **Explain the setup:**
   - Point out Start (green), Goal (red), Agent (blue)
   - Explain fog of war (gray areas are unknown)

3. **Step through first search:**
   - Press **S** to search
   - Show cyan expanded cells
   - Show red planned path

4. **Let it run:**
   - Press **SPACE** to start
   - Watch agent discover obstacles
   - Watch it replan when blocked

5. **Compare algorithms:**
   - Reset with **R**
   - Run with different `--algorithm` settings
   - Show performance differences

---

**Enjoy the interactive visualization! 🎮✨**
