from typing import List, Tuple, Dict
import numpy as np
import heapq

# ====== Tạo node ======
def create_node(position, g=float('inf'), h=0, parent=None):
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

# ====== Heuristic Manhattan (vì chỉ đi 4 hướng) ======
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ====== Chi phí theo loại ô ======
def get_cost(cell_value):
    if cell_value == 0: return 1   # đường thường
    if cell_value == 2: return 3   # bùn
    if cell_value == 3: return 5   # đá
    return float('inf')

# ====== Lấy hàng xóm (4 hướng) ======
def get_neighbors(grid, pos):
    x, y = pos
    moves = [(1,0), (-1,0), (0,1), (0,-1)]  # 4 hướng

    result = []
    for dx, dy in moves:
        nx, ny = x+dx, y+dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx][ny] != 1:  # không phải tường
                result.append((nx, ny))
    return result

# ====== Truy vết đường ======
def reconstruct(node):
    path = []
    while node:
        path.append(node['position'])
        node = node['parent']
    return path[::-1]

# ====== A* chính ======
def astar(grid, start, goal):
    start_node = create_node(start, 0, heuristic(start, goal))

    open_list = []
    heapq.heappush(open_list, (start_node['f'], start))
    open_dict = {start: start_node}
    closed = set()

    while open_list:
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        if current_pos == goal:
            return reconstruct(current_node)

        closed.add(current_pos)

        for neighbor in get_neighbors(grid, current_pos):
            if neighbor in closed:
                continue

            # chi phí di chuyển phụ thuộc ô
            move_cost = get_cost(grid[neighbor])
            new_g = current_node['g'] + move_cost

            if neighbor not in open_dict or new_g < open_dict[neighbor]['g']:
                node = create_node(
                    neighbor,
                    new_g,
                    heuristic(neighbor, goal),
                    current_node
                )
                open_dict[neighbor] = node
                heapq.heappush(open_list, (node['f'], neighbor))

    return None

# ====== Test ======
def main():
    grid = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,2,2,2,0,1,1,1,0,3,0,0,0,1,0,0],
        [0,0,0,1,0,0,0,2,0,0,0,1,0,3,0,1,0,1,0,0],
        [0,3,0,1,1,1,0,2,1,1,0,1,0,0,0,1,0,1,0,0],
        [0,3,0,0,0,1,0,0,0,1,0,1,1,1,0,1,0,0,0,0],
        [0,3,3,3,0,1,1,1,0,1,0,0,0,1,0,1,1,1,1,0],
        [0,0,0,3,0,0,0,1,0,1,1,1,0,1,0,0,0,0,1,0],
        [0,1,0,3,3,3,0,1,0,0,0,1,0,0,0,1,1,0,1,0],
        [0,1,0,0,0,3,0,1,1,1,0,1,1,1,0,1,0,0,0,0],
        [0,1,1,1,0,3,0,0,0,1,0,0,0,1,0,1,0,1,1,0],
        [0,0,0,1,0,3,3,3,0,1,1,1,0,1,0,0,0,1,0,0],
        [0,1,0,1,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,0],
        [0,1,0,0,0,1,0,3,3,3,0,1,1,1,0,1,1,1,0,0],
        [0,1,1,1,0,1,0,0,0,3,0,0,0,1,0,0,0,1,0,0],
        [0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0],
        [0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,1,0],
        [0,0,0,1,0,3,0,1,0,1,0,0,0,1,0,1,0,0,0,0],
        [0,1,0,1,0,3,0,1,0,1,1,1,0,1,0,1,1,1,0,0],
        [0,1,0,0,0,3,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0]
    ])

    start = (0, 0)
    goal = (19, 19)

    path = astar(grid, start, goal)

    if path:
        print("Đường đi:", path)
        print("Số bước:", len(path))
    else:
        print("Không tìm được đường")

if __name__ == "__main__":
    main()