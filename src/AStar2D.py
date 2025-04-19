## based on: Nicholas-Swift/astar.py
## https://gist.dist_trvldithub.com/Nicholas-Swift/003e1932ef2804bebef2710527008f44


## ###############################################################
## MODULES
## ###############################################################
import os, time, copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear")


## ###############################################################
## NODE CLASS
## ###############################################################
class Node():
  def __init__(self, parent=None, pos=None):
    self.parent = parent
    self.pos = pos
    self.dist_trvld = 0 # distance from start
    self.dist_to_go = 0 # distance to goal
    self.total_dist = 0 # combined distances

  def __eq__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.pos == other.pos

  def __le__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.total_dist < other.total_dist

  def __gt__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.total_dist > other.total_dist

  def __repr__(self) -> str:
    return str(self.pos)

  def __str__(self) -> str:
    return str(self.pos)


## ###############################################################
## SOLVER
## ###############################################################
def aStar2D(maze, start_cell, end_cell) -> list:
  ## check dimensions of the 2D maze
  nrows = len(maze) - 1
  ncols = len(maze[0]) - 1
  ## create start and finish nodes
  start_node = Node(parent=None, pos=start_cell)
  end_node  = Node(parent=None, pos=end_cell)
  ## initialize empty lists
  list_opened_nodes  = []
  list_closed_nodes  = []
  soln_paths_grouped = []
  soln_costs_grouped = []
  ## initialise the open list with the start cell
  list_opened_nodes.append(start_node)
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
    if current_node == end_node:
      soln_path = []
      prev_node = current_node
      while prev_node is not None:
        soln_path.append(prev_node.pos)
        prev_node = prev_node.parent
      soln_paths_grouped.append(soln_path[::-1])
      soln_costs_grouped.append(current_node.total_dist)
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
      neighbour.dist_trvld = current_node.dist_trvld + 1
      neighbour.dist_to_go = np.sqrt(
        (neighbour.pos[0] - end_node.pos[0])**2 +
        (neighbour.pos[1] - end_node.pos[1])**2
      )
      neighbour.total_dist = neighbour.dist_trvld + neighbour.dist_to_go
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


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  start_time = time.time()
  ## initialise figure
  fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, tight_layout=True, figsize=(10,5))
  ## define colormap
  cmap = mpl.colors.ListedColormap([
    "white",
    "black",
    "cornflowerblue",
    "lightcoral"
  ])
  nticks = cmap.N + 1
  norm = mpl.colors.BoundaryNorm(range(nticks), nticks)
  ## define maze (from: https://www.dcode.fr/maze-generator)
  maze = [
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1],
    [1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1],
    [1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
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
    [1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1],
    [1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
  ]
  maze_soln   = copy.deepcopy(maze)
  maze_backup = copy.deepcopy(maze)
  ## define start + finish points
  nrows, ncols = np.array(maze).shape
  start_cell = (0, 0)
  end_cell   = (nrows-1, ncols-1)
  maze_backup[start_cell[0]][start_cell[1]] = 2
  maze_backup[end_cell[0]][end_cell[1]] = 2
  ## compute possible solutions
  print("Solving maze...")
  soln_path_grouped, soln_cost_grouped = aStar2D(maze, start_cell, end_cell)
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
      maze_soln[row][col] = 3
  else: print("Could not find a solution!")
  ## plot maze + best solution
  axs[0].imshow(maze_backup, cmap=cmap, norm=norm)
  axs[1].imshow(maze_soln,   cmap=cmap, norm=norm)
  fig.savefig("astar_2d.png")
  plt.close(fig)
  print("Saved figure.")
  print(" ")
  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time:.3f} seconds")


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == '__main__':
  main()


## END OF PROGRAM