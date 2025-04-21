import heapq
from typing import List, Dict, Tuple, Optional


def _euclidean_distance(
  p1: Tuple[float, float, float],
  p2: Tuple[float, float, float]
) -> float:
  dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
  return (dx**2 + dy**2 + dz**2) ** 0.5

class Node:
  def __init__(self, vertex_index: int, parent: Optional['Node'] = None):
    self.vertex_index = vertex_index
    self.parent = parent
    self.distance_traveled = 0.0 # g
    self.estimated_distance_to_goal = 0.0 # h
    self.total_estimated_cost = 0.0 # f = g + h

  def __eq__(self, other):
    return isinstance(other, Node) and self.vertex_index == other.vertex_index

  def __lt__(self, other):
    return self.total_estimated_cost < other.total_estimated_cost

  def __repr__(self):
    return f"Node(vi={self.vertex_index})"

def _reconstruct_path(
  end_node: Node,
  vertices: List[Tuple[float, float, float]]
) -> List[Tuple[float, float, float]]:
  path = []
  current = end_node
  while current:
    path.append(vertices[current.vertex_index])
    current = current.parent
  path.reverse()
  return path

def a_star_3d(
  vertices: List[Tuple[float, float, float]],
  adjacency: Dict[int, List[int]],
  start_vi: int,
  goal_vi: int
) -> Tuple[List[List[Tuple[float, float, float]]], List[float]]:
  start_node = Node(vertex_index=start_vi)
  goal_node = Node(vertex_index=goal_vi)
  open_nodes = []
  heapq.heappush(open_nodes, start_node)
  closed_nodes = set()
  solution_paths = []
  solution_costs = []
  while open_nodes:
    current_node = heapq.heappop(open_nodes)
    if current_node.vertex_index == goal_vi:
      path = _reconstruct_path(current_node, vertices)
      solution_paths.append(path)
      solution_costs.append(current_node.total_estimated_cost)
      break # stop after finding the first solution
    closed_nodes.add(current_node.vertex_index)
    for neighbor_vi in adjacency[current_node.vertex_index]:
      if neighbor_vi in closed_nodes:
        continue
      neighbor_node = Node(vertex_index=neighbor_vi, parent=current_node)
      step_cost = _euclidean_distance(
        vertices[current_node.vertex_index],
        vertices[neighbor_node.vertex_index]
      )
      heuristic_cost = _euclidean_distance(
        vertices[neighbor_node.vertex_index],
        vertices[goal_node.vertex_index]
      )
      neighbor_node.distance_traveled = current_node.distance_traveled + step_cost
      neighbor_node.estimated_distance_to_goal = heuristic_cost
      neighbor_node.total_estimated_cost = neighbor_node.distance_traveled + heuristic_cost
      ## check if a better path exists in open_nodes
      better_path_exists = False
      for existing_node in open_nodes:
        if neighbor_node == existing_node and neighbor_node.distance_traveled >= existing_node.distance_traveled:
          better_path_exists = True
          break
      if not better_path_exists:
        heapq.heappush(open_nodes, neighbor_node)
  return solution_paths, solution_costs
