def do_a_star(grid, start, end, display_message):
    """Main function implementing the A* path planning algorithm, combining heuristic search and dynamic programming principles"""
    
    # --- Heuristic function: Euclidean distance calculation ---
    def heuristic(a, b):
        """Heuristic function: Calculates Euclidean distance (L2 norm) between node a and target node b"""
        (x1, y1) = a  # Current node coordinates (col, row)
        (x2, y2) = b  # Target node coordinates (col, row)
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5  # Formula: √[(Δx)² + (Δy)²]

    # --- Input validation and initialization ---
    COL = len(grid)    # Grid column count (horizontal cells, 0-based indexing)
    ROW = len(grid[0]) if COL > 0 else 0  # Grid row count (vertical cells)
    start_col, start_row = start  # Start coordinates (col, row)
    end_col, end_row = end        # Goal coordinates (col, row)

    # --- Data structure initialization (A* core components) ---
    open_set = []        # Priority queue (nodes to expand), format: (f-value, col, row)
    closed_set = set()   # Visited nodes (hash set), prevents redundant expansion
    g_scores = {}        # Actual cost dictionary: stores minimum known g(n) from start
    parents = {}         # Parent pointer dictionary: for path backtracking

    # Initialize start node
    start_node = (start_col, start_row)
    end_node = (end_col, end_row)
    g_scores[start_node] = 0                    # Start node's self-cost is 0
    h_start = heuristic(start_node, end_node)   # Heuristic estimate (h-value)
    f_start = g_scores[start_node] + h_start    # Total cost f(n) = g(n) + h(n)
    open_set.append((f_start, start_col, start_row))  # Add start node to open set

    found = False  # Flag indicating path discovery

    # --- A* main loop: Node expansion and cost updates ---
    while open_set:
        # Select node with minimal f-value from open set (priority queue operation)
        current = None
        current_idx = -1
        min_f = float('inf')
        # Linear search implementation (constrained by no external libraries)
        for idx, (f, x, y) in enumerate(open_set):
            if f < min_f:
                min_f = f
                current = (x, y)  # Current optimal node
                current_idx = idx

        if current is None:
            break  # Open set empty, no solution

        del open_set[current_idx]  # Remove current node from open set

        if current == end_node:    # Termination condition: reached goal node
            found = True
            break

        closed_set.add(current)    # Mark current node as expanded

        # --- Generate 4-connected neighbors (Manhattan movement model) ---
        x, y = current
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Left, right, down, up
            nx = x + dx  # Neighbor column index
            ny = y + dy  # Neighbor row index
            # Validate if neighbor is within grid and traversable (grid value 1)
            if 0 <= nx < COL and 0 <= ny < ROW and grid[nx][ny] == 1:
                neighbors.append((nx, ny))

        # --- Update neighbor costs and parent pointers ---
        for neighbor in neighbors:
            if neighbor in closed_set:  # Skip closed nodes
                continue

            # Calculate tentative actual cost: g(current) + movement cost (1 here)
            tentative_g = g_scores.get(current, float('inf')) + 1

            # If better path found (lower g-value), update neighbor
            if tentative_g < g_scores.get(neighbor, float('inf')):
                parents[neighbor] = current      # Update parent pointer
                g_scores[neighbor] = tentative_g  # Update actual cost
                h = heuristic(neighbor, end_node)  # Recalculate heuristic
                f = tentative_g + h               # Total cost f(n) = g(n) + h(n)

                # Check if neighbor exists in open set and update f-value if improved
                in_open = False
                for i, (old_f, ox, oy) in enumerate(open_set):
                    if (ox, oy) == neighbor:
                        in_open = True
                        if f < old_f:  # Replace with better f-value
                            open_set[i] = (f, neighbor[0], neighbor[1])
                        break
                if not in_open:  # Add to open set if new
                    open_set.append((f, neighbor[0], neighbor[1]))

    # --- Path not found handling ---
    if not found:
        display_message("No valid path found")  # Output warning
        return []

    # --- Path reconstruction: Backtrack parent pointers ---
    path = []
    current = end_node
    while current != start_node:
        path.append(current)          # Add current node to path
        current = parents.get(current)  # Backtrack parent
        if current is None:  # Path discontinuity check
            display_message("Path reconstruction failed")
            return []
    path.append(start_node)  # Append start node
    path.reverse()           # Reverse path to start-to-goal order

    # --- Debug output (following message protocol) ---
    display_message("Path found between the start and end points")  # Success flag
    display_message("Path found with length: {}".format(len(path)))  # Path length

    return path  