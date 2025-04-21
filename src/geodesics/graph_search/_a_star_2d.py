## ###############################################################
## DEPENDENCIES
## ###############################################################
import heapq
from typing import List, Tuple, Optional


## ###############################################################
## NODE CLASS
## ###############################################################
class Node:
  def __init__(
      self,
      position: Tuple[int, int],
      parent: Optional["Node"] = None
    ):
    self.position = position
    self.parent = parent
    self.distance_traveled = 0 # from start (g)
    self.estimated_distance_to_goal = 0 # heuristic (h)
    self.total_estimated_cost = 0 # g + h

  def __eq__(self, other):
    return isinstance(other, Node) and self.position == other.position

  def __lt__(self, other):
    return self.total_estimated_cost < other.total_estimated_cost

  def __repr__(self):
    return f"Node({self.position})"


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def _heuristic_estimate(
    current: Tuple[int, int],
    goal: Tuple[int, int]
  ) -> float:
  dx = goal[0] - current[0]
  dy = goal[1] - current[1]
  # return (dx**2 + dy**2)**0.5 # euclidean distance
  return abs(dx) + abs(dy) # manhattan distance

def _get_walkable_neighbors(
    position: Tuple[int, int],
    maze: List[List[int]]
  ) -> List[Tuple[int, int]]:
  directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
  neighbors = []
  for dx, dy in directions:
    x, y = position[0] + dx, position[1] + dy
    if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] == 0:
      neighbors.append((x, y))
  return neighbors

def _reconstruct_path(end_node: Node) -> List[Tuple[int, int]]:
  path = []
  current = end_node
  while current:
    path.append(current.position)
    current = current.parent
  path.reverse()
  return path


## ###############################################################
## A-STAR ALGORITHM
## ###############################################################
def a_star_2d(
    maze  : List[List[int]],
    start : Tuple[int, int],
    goal  : Tuple[int, int],
    use_priority_queue = True
  ) -> Tuple[List[Tuple[int, int]], float]:
  ## make sure the input maze is a list of lists
  if hasattr(maze, "tolist"): maze = maze.tolist()
  start_node = Node(start)
  start_node.distance_traveled = 0
  start_node.estimated_distance_to_goal = _heuristic_estimate(start, goal)
  start_node.total_estimated_cost = start_node.distance_traveled + start_node.estimated_distance_to_goal
  if use_priority_queue:
    open_nodes = []
    heapq.heappush(open_nodes, start_node)
  else: open_nodes = [start_node]
  closed_positions = set()
  while len(open_nodes) > 0:
    if use_priority_queue:
      current_node = heapq.heappop(open_nodes)
    else:
      current_node = open_nodes[0]
      for node in open_nodes:
        if node.total_estimated_cost < current_node.total_estimated_cost:
          current_node = node
      open_nodes.remove(current_node)
    if current_node.position == goal:
      return _reconstruct_path(current_node), current_node.total_estimated_cost
    closed_positions.add(current_node.position)
    for neighbor_position in _get_walkable_neighbors(current_node.position, maze):
      if neighbor_position in closed_positions:
        continue
      neighbor_node = Node(neighbor_position, parent=current_node)
      neighbor_node.distance_traveled = current_node.distance_traveled + 1
      neighbor_node.estimated_distance_to_goal = _heuristic_estimate(neighbor_position, goal)
      neighbor_node.total_estimated_cost = neighbor_node.distance_traveled + neighbor_node.estimated_distance_to_goal
      ## skip if a better path to this node is already in the open list
      better_path_exists = False
      for node in open_nodes:
        if (neighbor_node == node) and (neighbor_node.distance_traveled > node.distance_traveled):
          better_path_exists = True
          break
      if not better_path_exists:
        if use_priority_queue:
          heapq.heappush(open_nodes, neighbor_node)
        else: open_nodes.append(neighbor_node)
  return [], float("inf")


## END OF MODULE