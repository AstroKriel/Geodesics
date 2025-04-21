## ###############################################################
## MODULES
## ###############################################################
import time
from mayavi import mlab


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################


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