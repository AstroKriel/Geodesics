## ###############################################################
## MODULES
## ###############################################################
import time
import numpy

from mayavi import mlab
from moviepy.editor import VideoClip
from scipy.spatial import KDTree

from PlotIsosurface import genPointCloud


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  start_time = time.time()
  ## generate data
  print("Generating point cloud...")
  res = 20
  verts, faces, normals = genPointCloud(implicit_func=gyroid, res=res)
  print("Computing adjacency dictionary...")
  dict_adj = genAdjDict_parallel(verts, faces)
  ## define start and end points
  tree = KDTree(verts)
  _, start_vi = tree.query([10, 20, 0])
  _, end_vi   = tree.query([20, 0, 20])
  start_point = verts[start_vi,:]
  end_point   = verts[end_vi,:]
  ## compute possible solutions
  print("Searching for geodesic path...")
  soln_path_grouped, soln_cost_grouped = aStar(verts, dict_adj, start_vi, end_vi)
  if len(soln_path_grouped) > 0:
    str_end = "s!" if len(soln_path_grouped) > 1 else "!"
    print(f"Found {len(soln_path_grouped)} possible geodesic path" + str_end)
    ## find index of the best (shortest) solution
    soln_index = numpy.argmin(soln_cost_grouped)
    soln_path = soln_path_grouped[soln_index]
    ## plot best solution
    x_geo = [ pos[0] for pos in soln_path ]
    y_geo = [ pos[1] for pos in soln_path ]
    z_geo = [ pos[2] for pos in soln_path ]
  else: print("Could not find a solution!")
  end_time = time.time()
  ## initialise figure
  fig = mlab.figure(size=(500,500), bgcolor=(1,1,1))
  ## plot isosurface
  # mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces, representation="surface")
  mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces, representation="mesh", tube_radius=0.05)
  ## plot geodesic
  if len(soln_path_grouped) > 0:
    mlab.points3d(start_point[0], start_point[1], start_point[2], color=(0,0,0))
    mlab.points3d(end_point[0], end_point[1], end_point[2], color=(0,0,0))
    # mlab.plot3d(x_geo, y_geo, z_geo, range(len(x_geo)), tube_radius=0.25)
    mlab.plot3d(x_geo, y_geo, z_geo, color=(0,0,0), tube_radius=0.2)
  # ## plot pseudo-axis
  # mlab.points3d(0, 0, 0)
  # mlab.points3d(2, 0, 0)
  # mlab.points3d(0, 4, 0)
  # mlab.points3d(0, 0, 6)
  # mlab.plot3d([0, 2.0], [0, 0], [0, 0], tube_radius=0.1)
  # mlab.plot3d([0, 0], [0, 4.0], [0, 0], tube_radius=0.1)
  # mlab.plot3d([0, 0], [0, 0], [0, 6.0], tube_radius=0.1)
  # mlab.points3d(res,res,res)
  ## animate
  duration = 7
  def make_frame(t):
    mlab.view(
      azimuth    = 360 * t/duration,
      elevation  = 70,
      distance   = 65,
      focalpoint = [res//2, res//2, res//2],
      figure     = fig
    )
    return mlab.screenshot(antialiased=True)
  animation = VideoClip(make_frame, duration=duration)
  animation.write_gif("geodesic.gif", fps=30)
  # ## show canvas
  # mlab.show()
  print(f"Elapsed time: {end_time - start_time:.3f} seconds")


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM