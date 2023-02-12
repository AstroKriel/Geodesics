## ###############################################################
## MODULES
## ###############################################################
import skimage
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def gyroid(x, y, z):
    return np.cos(x) * np.sin(y) + np.cos(y) * np.sin(z) + np.cos(z) * np.sin(x)


## ###############################################################
## PLOT ISOSURFACE DATA WITH MATPLOTLIB
## ###############################################################
def drawSurfaceMesh(ax, verts, faces):
  num_faces = faces.shape[0]
  ## draw points
  ax.plot(verts[:,0], verts[:,1], verts[:,2], color="black", marker=".", ms=1, ls="")
  ## draw conective mesh
  for face_index in range(num_faces-1):
    for comb_pair in list(combinations(faces[face_index], 2)):
      ax.plot(
        [ verts[comb_pair[0],0], verts[comb_pair[1],0] ],
        [ verts[comb_pair[0],1], verts[comb_pair[1],1] ],
        [ verts[comb_pair[0],2], verts[comb_pair[1],2] ],
        color="black", ls="-", lw=0.1, zorder=1
      )

class PlotWMatplotlib():
  def __init__(
      self,
      verts, faces, normals,
      soln_path      = [],
      start_point    = [],
      end_point      = [],
      bool_plot_mesh = False
    ) -> None:
    self.verts          = verts
    self.faces          = faces
    self.normals        = normals
    self.soln_path      = soln_path
    self.start_point    = start_point
    self.end_point      = end_point
    self.bool_plot_mesh = bool_plot_mesh

  def plot(self):
    ## create figure canvas and a 3D axis
    self.fig = plt.figure()
    ax = self.fig.add_subplot(111, projection="3d")
    ## draw isosurface mesh
    if self.bool_plot_mesh:
      drawSurfaceMesh(ax, self.verts, self.faces)
    ## draw data points
    ax.plot(self.verts[:,0], self.verts[:,1], self.verts[:,2], "k.", ms=0.1, alpha=0.5)
    ## draw solution(s)
    if len(self.soln_path) > 0:
      ax.plot(self.soln_path[0], self.soln_path[1], self.soln_path[2], color="black", ls="-", lw=2, zorder=3)
    ## draw start and end point
    if len(self.start_point) > 0: ax.plot(*self.start_point, "b.", zorder=3)
    if len(self.end_point) > 0:   ax.plot(*self.end_point,   "r.", zorder=3)
    ## set the plot labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ## save the plot
    ax.view_init(30, 26)
    self.fig.tight_layout()

  def showFig(self):
    plt.show()

  def saveFig(self):
    self.fig.savefig("astar_3d.png")
    plt.close(self.fig)
    print("Saved figure.")


## ###############################################################
## COMPUTE ISOSURFACE
## ###############################################################
def genPointCloud(implicit_func=gyroid, res=100):
  ## define domain range
  xmin, xmax = -np.pi, np.pi
  ymin, ymax = -np.pi, np.pi
  zmin, zmax = -np.pi, np.pi
  ## generate a grid of points
  x_3d, y_3d, z_3d = np.meshgrid(
    np.linspace(xmin, xmax, res),
    np.linspace(ymin, ymax, res),
    np.linspace(zmin, zmax, res)
  )
  ## evaluate gyroid function at each point in domain
  F = implicit_func(x_3d, y_3d, z_3d)
  ## extract the gyroid isosurface
  verts, faces, normals, values = skimage.measure.marching_cubes(F, 0)
  ## return useful mesh information
  return verts, faces, normals


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  ## generate isosurface data
  verts, faces, normals = genPointCloud(implicit_func=gyroid, res=200)
  ## plot isosurface
  plot_obj = PlotWMatplotlib(verts, faces, normals)
  plot_obj.plot()
  plot_obj.showFig()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM