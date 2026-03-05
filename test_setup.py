#!/usr/bin/env python3
"""
Test Script - Verify your setup is working correctly
Run this first to make sure everything is installed properly
"""

import sys
import os

print("="*70)
print(" "*20 + "SETUP VERIFICATION TEST")
print("="*70)
print()

# Test 1: Python version
print("Test 1: Python Version")
print(f"  Python {sys.version.split()[0]}", end=" ")
version = sys.version_info
if version.major >= 3 and version.minor >= 6:
    print("✓ OK")
else:
    print("✗ FAIL - Need Python 3.6+")
    sys.exit(1)
print()

# Test 2: NumPy
print("Test 2: NumPy")
try:
    import numpy as np
    print(f"  NumPy {np.__version__} ✓ OK")
except ImportError:
    print("  ✗ FAIL - NumPy not installed")
    print("  Run: pip install numpy")
    sys.exit(1)
print()

# Test 3: Matplotlib
print("Test 3: Matplotlib")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print(f"  Matplotlib {matplotlib.__version__} ✓ OK")
except ImportError:
    print("  ✗ FAIL - Matplotlib not installed")
    print("  Run: pip install matplotlib")
    sys.exit(1)
print()

# Test 4: Check for required files
print("Test 4: Required Files")
required_files = [
    'pathfinding.py',
    'demo.py',
    'animate_pathfinding.py'
]

all_present = True
for filename in required_files:
    if os.path.exists(filename):
        print(f"  {filename} ✓ Found")
    else:
        print(f"  {filename} ✗ Missing")
        all_present = False

if not all_present:
    print("\n  Some files are missing!")
    print("  Make sure all files are in the same folder.")
    sys.exit(1)
print()

# Test 5: Import pathfinding module
print("Test 5: Import Pathfinding Module")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathfinding import GridWorld, Agent, RepeatedForwardAStar
    print("  ✓ Module imports successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)
print()

# Test 6: Create a small grid
print("Test 6: Create Test Grid")
try:
    gw = GridWorld(size=5)
    gw.generate_maze_dfs(block_probability=0.3, seed=42)
    print(f"  ✓ Created 5x5 grid")
    print(f"    Start: {gw.start}")
    print(f"    Goal: {gw.goal}")
except Exception as e:
    print(f"  ✗ Failed to create grid: {e}")
    sys.exit(1)
print()

# Test 7: Run A* search
print("Test 7: Run A* Search")
try:
    agent = Agent(gw)
    solver = RepeatedForwardAStar(agent, tie_breaking='large_g')
    trajectory, expanded, success = solver.find_path()
    
    if success:
        print(f"  ✓ A* search successful!")
        print(f"    Expansions: {solver.stats['total_expansions']}")
        print(f"    Path length: {solver.stats['path_length']}")
    else:
        print(f"  ⚠ Search completed but no path found (this is OK)")
except Exception as e:
    print(f"  ✗ Search failed: {e}")
    sys.exit(1)
print()

# Test 8: Generate visualization
print("Test 8: Generate Visualization")
try:
    test_dir = 'test_output'
    os.makedirs(test_dir, exist_ok=True)
    
    gw.visualize(title="Test Grid", save_path=f"{test_dir}/test.png")
    
    if os.path.exists(f"{test_dir}/test.png"):
        print(f"  ✓ Visualization saved to {test_dir}/test.png")
        print(f"    You can open this file to see the grid!")
    else:
        print(f"  ✗ Visualization file not created")
except Exception as e:
    print(f"  ✗ Visualization failed: {e}")
    sys.exit(1)
print()

# All tests passed!
print("="*70)
print(" "*25 + "ALL TESTS PASSED! ✓")
print("="*70)
print()
print("Your setup is working correctly!")
print()
print("Try these commands:")
print("  python demo.py demo 5 42         # Simple 5x5 demo")
print("  python demo.py demo 10 15        # Bigger 10x10 demo")
print("  python demo.py animate 10 15     # Generate animation")
print("  python demo.py compare 15 5      # Compare algorithms")
print()
print("Check the 'test_output/' folder for your test visualization!")
print()
