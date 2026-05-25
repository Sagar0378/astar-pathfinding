import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from heapq import heappush, heappop

# ─────────────────────────────────────────
#  GRID SETUP
# ─────────────────────────────────────────
GRID_ROWS = 20
GRID_COLS = 20

# 0 = free, 1 = wall
grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)

# Add walls/obstacles
walls = [
    (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
    (6, 8), (7, 8), (8, 8), (9, 8), (10, 8),
    (10, 7), (10, 6), (10, 5), (10, 4),
    (3, 12), (4, 12), (5, 12), (6, 12),
    (7, 12), (7, 13), (7, 14), (7, 15),
]
for (r, c) in walls:
    grid[r][c] = 1

START = (1, 1)
GOAL  = (18, 18)

# ─────────────────────────────────────────
#  HEURISTIC — Euclidean Distance
# ─────────────────────────────────────────
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ─────────────────────────────────────────
#  A* ALGORITHM
# ─────────────────────────────────────────
def astar(grid, start, goal):
    rows, cols = grid.shape
    
    # Each entry: (f, g, position, parent)
    open_list = []
    heappush(open_list, (0 + heuristic(start, goal), 0, start, None))
    
    came_from = {}   # tracks the path
    g_score   = {start: 0}
    visited   = set()

    while open_list:
        f, g, current, parent = heappop(open_list)

        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent

        # ── GOAL REACHED ──
        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, visited

        # ── EXPLORE NEIGHBORS (8 directions) ──
        r, c = current
        neighbors = [
            (r-1, c),   (r+1, c),   # up, down
            (r, c-1),   (r, c+1),   # left, right
            (r-1, c-1), (r-1, c+1), # diagonals
            (r+1, c-1), (r+1, c+1),
        ]

        for nr, nc in neighbors:
            neighbor = (nr, nc)
            # Skip out-of-bounds or walls
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue
            if neighbor in visited:
                continue

            # Diagonal movement costs more (√2 ≈ 1.414)
            move_cost = 1.414 if (nr != r and nc != c) else 1.0
            tentative_g = g + move_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heappush(open_list, (f_score, tentative_g, neighbor, current))

    return None, visited  # No path found

# ─────────────────────────────────────────
#  VISUALIZE
# ─────────────────────────────────────────
def visualize(grid, path, visited, start, goal):
    visual = np.zeros((*grid.shape, 3))  # RGB grid

    # Colors
    WHITE  = [1.00, 1.00, 1.00]  # free cell
    BLACK  = [0.15, 0.15, 0.15]  # wall
    BLUE   = [0.53, 0.81, 0.98]  # explored
    YELLOW = [1.00, 0.85, 0.00]  # path
    GREEN  = [0.18, 0.80, 0.44]  # start
    RED    = [0.91, 0.30, 0.24]  # goal

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r][c] == 1:
                visual[r][c] = BLACK
            else:
                visual[r][c] = WHITE

    for (r, c) in visited:
        if grid[r][c] == 0:
            visual[r][c] = BLUE

    if path:
        for (r, c) in path:
            visual[r][c] = YELLOW

    # Start and goal override everything
    visual[start[0]][start[1]] = GREEN
    visual[goal[0]][goal[1]]   = RED

    # ── PLOT ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(visual, origin='upper')
    ax.set_title("A* Pathfinding — Autonomous Robot Navigation", fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(-0.5, GRID_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_ROWS, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.3)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # Legend
    legend = [
        mpatches.Patch(color=GREEN,  label='Start'),
        mpatches.Patch(color=RED,    label='Goal'),
        mpatches.Patch(color=BLACK,  label='Wall'),
        mpatches.Patch(color=BLUE,   label='Explored Cells'),
        mpatches.Patch(color=YELLOW, label=f'Optimal Path ({len(path)} steps)'),
    ]
    ax.legend(handles=legend, loc='upper left',
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("astar_result.png", dpi=150)
    plt.show()
    print(f"\n✅ Path found! Length: {len(path)} steps")
    print(f"🔍 Cells explored: {len(visited)}")
    print(f"📁 Image saved as astar_result.png")

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Running A* Pathfinding...")
    path, visited = astar(grid, START, GOAL)

    if path:
        visualize(grid, path, visited, START, GOAL)
    else:
        print("❌ No path found!")