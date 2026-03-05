# 🚀 QUICK START GUIDE - Getting Started in 5 Minutes

## Step 1: Get All Your Files (30 seconds)

Download and put these files in the **same folder**:

```
📁 my_astar_project/
   ├── pathfinding.py          ⭐ Core algorithms
   ├── demo.py                 ⭐ Main program to run
   ├── animate_pathfinding.py  ⭐ Animation maker
   ├── test_setup.py           ⭐ Test everything works
   ├── visualize.py            (optional - advanced)
   ├── quickstart.py           (optional - alternative demo)
   ├── README.md               (documentation)
   └── VSCODE_SETUP.md         (VSCode help)
```

**Minimum required:** Just the 4 files marked with ⭐

---

## Step 2: Install Python Packages (1 minute)

Open your terminal/command prompt in VSCode:
- **Windows:** Press `` Ctrl+` ``
- **Mac:** Press `` Cmd+` ``
- **Or:** View → Terminal

Then run:
```bash
pip install numpy matplotlib
```

**If that doesn't work, try:**
```bash
pip install numpy matplotlib --user
```

**Or:**
```bash
python -m pip install numpy matplotlib
```

---

## Step 3: Test Your Setup (30 seconds)

Run the test script:
```bash
python test_setup.py
```

**You should see:**
```
✓ Python version OK
✓ NumPy OK
✓ Matplotlib OK
✓ All files found
✓ Module imports
✓ Grid creation works
✓ A* search works
✓ Visualization works

ALL TESTS PASSED! ✓
```

**If you see errors:** Check VSCODE_SETUP.md for troubleshooting

---

## Step 4: Run Your First Demo (1 minute)

```bash
python demo.py demo 5 42
```

**You should see:**
```
Creating 5x5 gridworld...
Grid Layout:
-----------------
| · · · █ · |
| █ · █ · █ |
| · █ · G █ |
| · · · S · |
| █ · · █ █ |
-----------------

Running Forward A* (large-g)...
✓ Success!
```

---

## Step 5: Try Different Commands (2 minutes)

### Small and Fast (5x5 grid):
```bash
python demo.py demo 5 42
```

### Medium Size (10x10 grid):
```bash
python demo.py demo 10 15
```

### Generate Pretty Pictures:
```bash
python demo.py animate 10 15
```
**Look in `demo_animation/` folder for images!**

### Compare Algorithms:
```bash
python demo.py compare 12 5
```
**Shows which algorithm is best!**

---

## 🎯 What Each Command Does

### `python demo.py demo [size] [seed]`
- Quick test on one grid
- Shows ASCII visualization
- Compares all 3 algorithms
- **Example:** `python demo.py demo 10 15`

### `python demo.py animate [size] [seed]`
- Creates step-by-step images
- Shows fog-of-war in action
- Great for presentations!
- **Example:** `python demo.py animate 15 20`

### `python demo.py compare [size] [num_tests]`
- Tests on multiple grids
- Calculates averages
- Shows which algorithm wins
- **Example:** `python demo.py compare 20 10`

---

## 📊 Understanding the Output

### Grid Symbols:
- `S` = Start position (agent begins here)
- `G` = Goal position (trying to reach this)
- `█` = Blocked cell (walls/obstacles)
- `·` = Unblocked cell (can walk through)
- `?` = Unknown (fog of war - not explored yet)

### Algorithm Names:
- **Forward A* (large-g)** - Standard algorithm, good performance
- **Forward A* (small-g)** - Standard algorithm, poor performance
- **Adaptive A*** - Learning algorithm, best performance

### Statistics:
- **Expansions** - How many cells it checked (lower is better)
- **Searches** - How many times it had to replan
- **Path Length** - Total moves to reach goal

---

## 🎬 Example Session (Copy & Paste!)

```bash
# 1. Test everything works
python test_setup.py

# 2. Try a small grid
python demo.py demo 5 42

# 3. Try a bigger grid
python demo.py demo 10 15

# 4. Make an animation (creates images)
python demo.py animate 10 15

# 5. Open the demo_animation folder and look at the images!

# 6. Compare algorithms
python demo.py compare 15 5
```

---

## 🎓 For Your Project

### Generate 30 Test Environments:
```bash
python pathfinding.py
```
This creates:
- `environments/` folder with 30 grids
- `environments/images/` folder with visualizations
- `results.pkl` with all statistics
- Takes 2-5 minutes depending on your computer

### Create Cool Animations for Presentation:
```bash
python demo.py animate 15 10
python demo.py animate 15 20
python demo.py animate 15 30
```

### Get Statistics for Your Report:
```bash
python demo.py compare 51 30
```

---

## ❓ Common Issues

### "No module named 'numpy'"
```bash
pip install numpy matplotlib
```

### "python is not recognized"
Try `python3` or `py` instead:
```bash
python3 demo.py demo 5 42
```

### Can't find the images
After running `animate`, look for:
- `demo_animation/` folder in your project folder
- Open the `.png` files with any image viewer
- They're numbered in order (frame_000.png, frame_001.png, etc.)

### Nothing happens when I run
Make sure you:
1. Opened the **terminal** in VSCode (`` Ctrl+` ``)
2. You're in the correct folder (with all the files)
3. Try `python3` instead of `python`

---

## ✅ Success Checklist

- [ ] All 4 main files in same folder
- [ ] `pip install numpy matplotlib` completed
- [ ] `python test_setup.py` shows "ALL TESTS PASSED"
- [ ] `python demo.py demo 5 42` shows grid and results
- [ ] `python demo.py animate 10 15` creates images
- [ ] Can view images in `demo_animation/` folder

---

## 🎉 You're Ready!

Once all checkboxes are ✓, you're good to go!

**Next steps:**
1. Try different grid sizes
2. Generate animations for your presentation
3. Run comparisons for your report
4. Generate the full 30 environments: `python pathfinding.py`

**Need more help?** 
- Check `VSCODE_SETUP.md` for detailed VSCode instructions
- Check `README.md` for complete documentation
- Every Python file has comments explaining what it does

---

**Have fun exploring pathfinding! 🗺️✨**
