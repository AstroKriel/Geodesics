## based on: Nicholas-Swift/astar.py
## https://gist.github.com/Nicholas-Swift/003e1932ef2804bebef2710527008f44

import os, time, copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

os.system("clear")

class Node():
  def __init__(self, parent=None, pos=None):
    self.parent = parent
    self.pos = pos
    self.g = 0 # distance from start
    self.h = 0 # distance to goal
    self.f = 0 # combined distances

  def __eq__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.pos == other.pos
  
  def __le__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.f < other.f
  
  def __gt__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.f > other.f

  def __repr__(self) -> str:
    return str(self.pos)

  def __str__(self) -> str:
    return str(self.pos)


def aStar(maze, first_cell, last_cell) -> list:
  ## check dimensions of the 2D maze
  nrows = len(maze) - 1
  ncols = len(maze[0]) - 1
  ## create start and finish nodes
  first_node = Node(parent=None, pos=first_cell)
  last_node  = Node(parent=None, pos=last_cell)
  ## initialize empty lists
  list_opened_nodes  = []
  list_closed_nodes  = []
  soln_paths_grouped = []
  soln_costs_grouped = []
  ## initialise the open list with the start cell
  list_opened_nodes.append(first_node)
  ## search until the shortest solution has been found
  while len(list_opened_nodes) > 0:
    ## GRAB THE NEXT MOST FAVOURABLE CELL
    ## ==================================
    current_node = list_opened_nodes[0]
    current_index = 0
    for node_index, node in enumerate(list_opened_nodes):
      if (node < current_node):
        current_node  = node
        current_index = node_index
    ## remove the current node from the open list
    list_opened_nodes.pop(current_index)
    ## CHECK IF THE GOAL HAS BEEN FOUND
    ## ================================
    if current_node == last_node:
      soln_path = []
      prev_node = current_node
      while prev_node is not None:
        soln_path.append(prev_node.pos)
        prev_node = prev_node.parent
      soln_paths_grouped.append(soln_path[::-1])
      soln_costs_grouped.append(current_node.f)
    ## GENERATE LIST OF VALID NEIGHBOURING CELLS
    ## =========================================
    for rel_index in [
        ( 0, 1),
        ( 1, 0),
        ( 0,-1),
        (-1, 0)
      ]:
      neighbour_pos = (
        current_node.pos[0] + rel_index[0],
        current_node.pos[1] + rel_index[1]
      )
      ## check that the neighbour is within the domain range
      if  ((neighbour_pos[0] < 0) or (nrows < neighbour_pos[0])) or\
          ((neighbour_pos[1] < 0) or (ncols < neighbour_pos[1])):
        continue
      ## check that the neighbour is walkable
      if not(maze[neighbour_pos[0]][neighbour_pos[1]] == 0):
        continue
      neighbour = Node(parent=current_node, pos=neighbour_pos)
      ## define score values
      neighbour.g = current_node.g + 1
      neighbour.h = (neighbour.pos[0] - last_node.pos[0])**2 +\
                    (neighbour.pos[1] - last_node.pos[1])**2
      neighbour.f = neighbour.g + neighbour.h
      ## check that the neighbour is not already in the open list
      if any([
        (neighbour == node) and (node < neighbour)
        for node in list_opened_nodes
      ]): continue
      ## check that the neighbour is not already in the closed list
      if any([
        (neighbour == node) and (node < neighbour)
        for node in list_closed_nodes
      ]): continue
      ## if the neighbour is in the open or closed list, then only consider
      ## it again if the score has improved
      list_opened_nodes.append(neighbour)
    ## add the node to the closed list
    list_closed_nodes.append(current_node)
  ## return solutions
  return soln_paths_grouped, soln_costs_grouped


def main():
  start_time = time.time()
  ## initialise figure + colormap
  fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, tight_layout=True, figsize=(10,5))
  cmap = mpl.colors.ListedColormap([
    "white",
    "black",
    "cornflowerblue",
    "lightcoral"
  ])
  nticks = cmap.N + 1
  norm = mpl.colors.BoundaryNorm(range(nticks), nticks)
  ## define maze (modified from: https://www.dcode.fr/maze-generator)
  maze = [
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1],
    [1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1],
    [1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1],
    [1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1],
    [1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1],
    [1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1],
    [1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1],
    [1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1],
    [1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
    [1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1],
    [1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
  ]
  soln = copy.deepcopy(maze)
  maze_backup = copy.deepcopy(maze)
  ## define start + finish points
  nrows, ncols = np.array(maze).shape
  first_cell = (0, 0)
  last_cell = (nrows-1, ncols-1)
  maze_backup[first_cell[0]][first_cell[1]] = 2
  maze_backup[last_cell[0]][last_cell[1]] = 2
  ## compute possible solutions
  print("Solving maze...")
  soln_path_grouped, soln_cost_grouped = aStar(maze, first_cell, last_cell)
  if len(soln_path_grouped) > 0:
    print(f"Found {len(soln_path_grouped)} possible solution(s)!")
    ## plot all solutions
    for soln_path in soln_path_grouped:
      list_pos_x = [ pos[1] for pos in soln_path ]
      list_pos_y = [ pos[0] for pos in soln_path ]
      axs[1].plot(list_pos_x, list_pos_y, ls="-", lw=1, marker="o", ms=2)
    ## find index of the shortest solution
    soln_index = np.argmin(soln_cost_grouped)
    ## draw shorest solution
    for pos in soln_path_grouped[soln_index]:
      row, col = pos
      soln[row][col] = 3
  else: print("Could not find a solution!")
  ## plot maze + best solution
  axs[0].imshow(maze_backup, cmap=cmap, norm=norm)
  mappable = axs[1].imshow(soln, cmap=cmap, norm=norm)
  # fig.colorbar(mappable, ticks=range(nticks))
  fig.savefig("astar.png")
  plt.close(fig)
  print("Saved figure.")
  print(" ")
  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time:.3f} seconds")


if __name__ == '__main__':
  main()

## END OF PROGRAM