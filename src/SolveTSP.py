## ###############################################################
## MODULES
## ###############################################################
import copy
import numpy as np
import matplotlib.pyplot as plt


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def euclidean(p1, p2):
  return np.sqrt(
    (p2[0] - p1[0])**2 +
    (p2[1] - p1[1])**2 +
    (p2[2] - p1[2])**2
  )

def plotPoints(list_points):
  for point in list_points:
    mlab.points3d(point[0], point[1], point[2])

def plotPath(list_points):
  x = [ point[0] for point in list_points ]
  y = [ point[1] for point in list_points ]
  z = [ point[2] for point in list_points ]
  mlab.plot3d(x, y, z, range(len(list_points)), tube_radius=0.25)

def plotPDF(ax, list_data, color="blue"):
  list_dens, list_bin_edges = np.histogram(list_data, bins=20, density=True)
  list_dens_norm = np.append(0, list_dens / list_dens.sum())
  ax.fill_between(list_bin_edges, list_dens_norm, step="pre", alpha=0.2, color=color)
  ax.plot(list_bin_edges, list_dens_norm, drawstyle="steps", color=color)


## ###############################################################
## POINTS ALONG SURFACE PERIMETER
## ###############################################################
def getListPoints():
  return [
    [0.497, 0.000, 15.000],
    [0.977, 0.000, 28.000],
    [0.000, 0.000, 29.000],
    [0.000, 1.000, 28.023],
    [0.000, 1.025, 28.000],
    [0.000, 2.231, 27.000],
    [0.000, 4.000, 11.494],
    [0.000, 7.000, 10.878],
    [0.000, 8.684, 11.000],
    [0.000, 10.482, 26.000],
    [0.000, 11.500, 12.000],
    [0.000, 12.000, 12.304],
    [0.506, 14.000, 0.000],
    [0.000, 13.997, 14.000],
    [0.506, 14.000, 29.000],
    [0.000, 14.500, 29.000],
    [0.000, 16.090, 16.000],
    [0.000, 17.000, 16.696],
    [0.000, 18.518, 3.000],
    [0.000, 19.000, 17.691],
    [0.000, 20.316, 18.000],
    [0.000, 24.000, 17.839],
    [0.000, 25.013, 3.000],
    [0.000, 28.000, 0.977],
    [0.000, 29.000, 0.000],
    [1.000, 28.023, 0.000],
    [0.497, 29.000, 15.000],
    [0.000, 29.000, 14.500],
    [0.000, 29.000, 29.000],
    [1.427, 0.000, 16.000],
    [1.595, 13.000, 0.000],
    [1.595, 13.000, 29.000],
    [2.000, 27.166, 0.000],
    [1.025, 28.000, 29.000],
    [1.427, 29.000, 16.000],
    [2.198, 0.000, 17.000],
    [3.000, 0.000, 18.518],
    [2.981, 12.000, 0.000],
    [2.981, 12.000, 29.000],
    [3.000, 26.488, 0.000],
    [2.231, 27.000, 29.000],
    [3.000, 26.488, 29.000],
    [3.000, 29.000, 25.013],
    [3.340, 0.000, 24.000],
    [3.455, 29.000, 20.000],
    [5.000, 25.660, 29.000],
    [5.796, 11.000, 29.000],
    [8.684, 11.000, 0.000],
    [9.000, 11.045, 29.000],
    [9.000, 25.545, 0.000],
    [11.000, 0.000, 8.684],
    [10.482, 26.000, 0.000],
    [11.000, 26.220, 29.000],
    [11.000, 29.000, 5.796],
    [12.000, 0.000, 2.981],
    [12.000, 0.000, 11.500],
    [11.500, 12.000, 0.000],
    [11.500, 12.000, 29.000],
    [12.000, 29.000, 2.981],
    [11.045, 29.000, 9.000],
    [13.000, 0.000, 1.595],
    [12.304, 0.000, 12.000],
    [12.910, 13.000, 0.000],
    [13.000, 29.000, 1.595],
    [12.304, 29.000, 12.000],
    [14.000, 0.000, 0.506],
    [14.000, 0.000, 13.997],
    [13.997, 14.000, 0.000],
    [14.000, 29.000, 0.506],
    [14.500, 0.000, 0.000],
    [14.500, 0.000, 29.000],
    [15.000, 0.000, 28.494],
    [15.000, 14.997, 29.000],
    [14.500, 29.000, 0.000],
    [14.997, 29.000, 15.000],
    [14.500, 29.000, 29.000],
    [15.000, 29.000, 28.494],
    [16.000, 0.000, 16.090],
    [16.000, 0.000, 27.405],
    [16.000, 15.926, 29.000],
    [15.926, 29.000, 16.000],
    [16.000, 29.000, 27.405],
    [16.696, 0.000, 17.000],
    [16.090, 16.000, 0.000],
    [16.090, 16.000, 29.000],
    [16.696, 29.000, 17.000],
    [17.000, 29.000, 26.019],
    [18.000, 0.000, 20.316],
    [17.839, 0.000, 24.000],
    [17.500, 17.000, 0.000],
    [17.955, 29.000, 20.000],
    [18.000, 29.000, 23.204],
    [18.518, 3.000, 0.000],
    [20.000, 3.455, 29.000],
    [20.000, 17.955, 29.000],
    [20.316, 18.000, 0.000],
    [24.000, 3.340, 0.000],
    [24.000, 17.839, 0.000],
    [23.204, 18.000, 29.000],
    [25.545, 0.000, 9.000],
    [26.000, 0.000, 10.482],
    [26.000, 2.512, 0.000],
    [25.013, 3.000, 29.000],
    [26.000, 2.512, 29.000],
    [25.660, 29.000, 5.000],
    [27.000, 1.834, 29.000],
    [26.019, 17.000, 29.000],
    [27.000, 29.000, 2.231],
    [26.220, 29.000, 11.000],
    [28.000, 0.977, 0.000],
    [27.573, 0.000, 13.000],
    [27.405, 16.000, 0.000],
    [27.405, 16.000, 29.000],
    [28.000, 29.000, 1.025],
    [27.573, 29.000, 13.000],
    [29.000, 0.000, 0.000],
    [28.023, 0.000, 1.000],
    [28.503, 0.000, 14.000],
    [29.000, 0.000, 14.500],
    [29.000, 0.000, 29.000],
    [29.000, 1.025, 28.000],
    [29.000, 2.981, 12.000],
    [29.000, 2.231, 27.000],
    [29.000, 3.987, 26.000],
    [29.000, 5.796, 11.000],
    [29.000, 8.000, 10.905],
    [29.000, 10.000, 11.309],
    [29.000, 11.000, 26.220],
    [29.000, 12.000, 12.304],
    [29.000, 14.500, 0.000],
    [28.494, 15.000, 0.000],
    [29.000, 15.000, 14.997],
    [28.494, 15.000, 29.000],
    [29.000, 16.000, 15.926],
    [29.000, 17.000, 16.696],
    [29.000, 20.000, 17.955],
    [29.000, 22.000, 18.122],
    [29.000, 23.204, 18.000],
    [29.000, 25.000, 17.506],
    [29.000, 25.013, 3.000],
    [29.000, 26.769, 2.000],
    [29.000, 27.975, 1.000],
    [29.000, 29.000, 0.000],
    [28.503, 29.000, 14.000],
    [29.000, 28.494, 15.000]
  ]


## ###############################################################
## NEAREST NEIGHBOR SOLUTION
## ###############################################################
def solveTSP_nn(list_points):
  ''' Solve Travelling Salesman Problem using Nearest Neighbor method
  '''
  list_points_unsorted = copy.deepcopy(list_points)
  current_point = list_points_unsorted.pop(0)
  list_points_sorted = [ current_point ]
  list_dist_travelled = []
  while len(list_points_unsorted) > 0:
    next_point_index = np.argmin([
      euclidean(current_point, next_point)
      for next_point in list_points_unsorted
      if not(next_point == current_point)
    ])
    next_point = list_points_unsorted[next_point_index]
    list_dist_travelled.append(euclidean(current_point, next_point))
    list_points_unsorted.remove(next_point)
    list_points_sorted.append(next_point)
    current_point = next_point
  return list_points_sorted, list_dist_travelled


## ###############################################################
## MOVING AVERAGE, NEAREST NEIGHBOR SOLUTION
## ###############################################################
def solveTSP_nn_stable(list_points):
  list_points_unsorted = copy.deepcopy(list_points)
  current_point = list_points_unsorted.pop(0)
  prev_point = [0, 0, 0]
  list_points_sorted = [ current_point ]
  list_dist_travelled = []
  while len(list_points_unsorted) > 0:
    next_point_index = np.argmin([
      euclidean(current_point, next_point) + euclidean(prev_point, next_point)
      for next_point in list_points_unsorted
      if not(next_point == current_point)
    ])
    next_point = list_points_unsorted[next_point_index]
    list_dist_travelled.append(euclidean(current_point, next_point))
    list_points_unsorted.remove(next_point)
    list_points_sorted.append(next_point)
    prev_point = current_point
    current_point = next_point
  return list_points_sorted, list_dist_travelled


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  list_points = getListPoints()
  ## find shortest path connecting points
  print("Solving travelling salesman problem...")
  list_points_sorted, list_dist_travelled = solveTSP_nn(list_points[::-1])
  # ## plot solution path
  # from mayavi import mlab
  # print("Plotting solution path...")
  # plotPoints(list_points)
  # plotPath(list_points_sorted_nn)
  # mlab.show()
  ## plot pdf of distances travelled
  print("Plotting PDF of distances travelled...")
  fig, ax = plt.subplots()
  plotPDF(ax, list_dist_travelled)
  fig.savefig("dists.png")


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM