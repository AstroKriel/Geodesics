## ###############################################################
## MODULES
## ###############################################################
import time
import numpy as np
import matplotlib.pyplot as plt

from mayavi import mlab

from MyAlgorithms.PlotIsosurface import genPointCloud
from MyAlgorithms.AStar3D import genAdjDict_parallel, aStar


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def gyroid(x, y, z):
  return np.cos(x) * np.sin(y) + np.cos(y) * np.sin(z) + np.cos(z) * np.sin(x)

def printPoint(point, pre=""):
  print(f"{pre}[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}],")


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  start_time = time.time()
  ## generate issurface data
  print("Generating point cloud...")
  verts, faces, normals = genPointCloud(implicit_func=gyroid, res=30)
  print("Computing adjacency dictionary...")
  # dict_adj = genAdjDict(verts, faces)
  dict_adj = genAdjDict_parallel(verts.shape[0], faces.shape[0], faces)
  ## plot isosurface
  mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces)
  ## compute edge points
  list_vi = []
  for vi in range(len(verts)):
    if len(dict_adj[vi]) < 4:
      list_vi.append(vi)
      mlab.points3d(verts[vi,0], verts[vi,1], verts[vi,2])
      printPoint(verts[vi])
  ## show canvas
  mlab.show()
  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time:.3f} seconds")


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM