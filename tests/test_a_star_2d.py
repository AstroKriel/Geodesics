## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import random
import matplotlib.pyplot as mpl_plot
from geodesics.graph_search import a_star_2d


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def generate_maze(num_rows, num_cols, seed=None):
  if seed is not None: random.seed(seed)
  ## ensure odd number of cells for proper walls
  num_rows = num_rows if num_rows % 2 == 1 else num_rows+1
  num_cols = num_cols if num_cols % 2 == 1 else num_cols+1
  ## start with all walls
  maze = numpy.ones((num_rows, num_cols), dtype=int)
  ## helper function
  def _carve(row, col):
    maze[row, col] = 0
    directions = [ (-2, 0), (2, 0), (0, 2), (0, -2) ] # N, S, E, W
    random.shuffle(directions)
    for (delta_row, delta_col) in directions:
      new_row = row + delta_row
      new_col = col + delta_col
      if not(0 < new_col < num_cols): continue # should be in bounds
      if not(0 < new_row < num_rows): continue # should be in bounds
      if not(maze[new_row, new_col] == 1): continue # should be a wall
      ## remove the wall between this and the next position
      maze[row + delta_row // 2, col + delta_col // 2] = 0
      _carve(new_row, new_col)
  ## start carving from the start position
  _carve(1, 1)
  return maze


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  maze = generate_maze(11, 15)
  num_rows, num_cols = maze.shape
  start = (1, 1)
  goal = (num_rows-2, num_cols-2)
  fig, ax = mpl_plot.subplots(figsize=(8, 6), tight_layout=True)
  ax.imshow(maze, cmap="Greys", origin="upper")
  ax.plot(start[1], start[0], color="green", marker="o", ms=15, zorder=3)
  ax.plot(goal[1], goal[0], color="red", marker="o", ms=15, zorder=3)
  print("Solving maze...")
  path, cost = a_star_2d(maze.tolist(), start, goal, use_priority_queue=True)
  if len(path) > 0:
    print(f"Found a solution of cost {cost:.2f} with {len(path)} steps")
    y_path = [ pos[0] for pos in path ]
    x_path = [ pos[1] for pos in path ]
    ax.plot(x_path, y_path, color="blue", marker="o", ms=5, ls="-", lw=2, zorder=5)
  else: print("Could not find a solution!")
  ax.set_xticks([])
  ax.set_yticks([])
  fig.savefig("test_a_star_2d.png")
  mpl_plot.close(fig)
  print("Saved figure: test_a_star_2d.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()

## END OF SCRIPT