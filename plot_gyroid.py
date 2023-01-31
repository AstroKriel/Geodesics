## ###############################################################
## MODULES
## ###############################################################
import time, skimage
import numpy as np
import multiprocessing as mproc
import matplotlib.pyplot as plt

from mayavi import mlab
from itertools import combinations


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def gyroid(x, y, z):
  return np.cos(x) * np.sin(y) + np.cos(y) * np.sin(z) + np.cos(z) * np.sin(x)

def euclidean(pos1, pos2):
  return np.sqrt(
    (pos2[0] - pos1[0])**2 +
    (pos2[1] - pos1[1])**2 +
    (pos2[2] - pos1[2])**2
  )

def getFlatNUniqueList(lol):
  return list(set([
    elem
    for inner_list in lol
    for elem in inner_list
  ]))

def printPoint(point, pre=""):
  print(f"{pre}[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}],")


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
  ## compute gyroid function at each point in domain
  F = implicit_func(x_3d, y_3d, z_3d)
  ## extract the gyroid isosurface
  verts, faces, normals, values = skimage.measure.marching_cubes(F, 0)
  ## return useful mesh information
  return verts, faces, normals


## ###############################################################
## GENERATE ADJACENCY DICTIONARY
## ###############################################################
def genAdjDict(verts, faces):
  num_points = verts.shape[0]
  num_faces  = faces.shape[0]
  ## initialise adjacency dictionary
  dict_adj = {}
  ## for each point store the list of point indices that are connected to it
  for point_index in range(num_points):
    dict_adj[point_index] = getFlatNUniqueList([
      [
        elem
        for elem in faces[face_index]
        if not (elem == point_index)
      ]
      for face_index in range(num_faces-1)
      if point_index in faces[face_index]
    ])
  ## return adjacency dictionary
  return dict_adj

def genAdjDict_helper(point_index, num_faces, faces):
    return point_index, getFlatNUniqueList([
      [
        elem
        for elem in faces[face_index]
        if not (elem == point_index)
      ]
      for face_index in range(num_faces-1)
      if point_index in faces[face_index]
    ])

def genAdjDict_parallel(num_points, num_faces, faces):
  with mproc.Pool() as pool_obj:
    return dict(pool_obj.starmap(
      genAdjDict_helper,
      [
        (point_index, num_faces, faces)
        for point_index in range(num_points)
      ]
    ))


## ###############################################################
## PLOT MESH WITH MATPLOTLIB
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
  def __init__(self, verts, faces, soln_path, start_point, end_point) -> None:
    self.verts       = verts
    self.faces       = faces
    self.soln_path   = soln_path
    self.start_point = start_point
    self.end_point   = end_point
  
  def plot(self):
    ## create figure canvas and a 3D axis
    self.fig = plt.figure()
    ax = self.fig.add_subplot(111, projection="3d")
    ## draw data
    drawSurfaceMesh(ax, self.verts, self.faces)
    ## draw solution(s)
    if len(self.soln_path) > 0:
      ax.plot(self.soln_path[0], self.soln_path[1], self.soln_path[2], color="black", ls="-", lw=2, zorder=3)
    ## draw start and end point
    ax.plot(*self.start_point, "b.", zorder=3)
    ax.plot(*self.end_point,   "r.", zorder=3)
    ## label axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ## show figure
    ax.view_init(15, 160)
    self.fig.tight_layout()

  def showFig(self):
    plt.show()

  def saveFig(self):
    self.fig.savefig("astar_3d.png")
    plt.close(self.fig)
    print("Saved figure.")


## ###############################################################
## NODE CLASS
## ###############################################################
class Node():
  def __init__(self, parent_node=None, vi=None):
    self.parent_node = parent_node
    self.vi = vi # list of vertices index (vi)
    self.dist_trvld = 0 # distance from start
    self.dist_to_go = 0 # distance to goal
    self.total_dist = 0 # combined distances

  def __eq__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.vi == other.vi
  
  def __le__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.total_dist < other.total_dist

  def __gt__(self, other) -> bool:
    if not isinstance(other, Node): return False
    return self.total_dist > other.total_dist

  def __repr__(self) -> str:
    return f"vertex index = {self.vi}"

  def __str__(self) -> str:
    return f"vertex index = {self.vi}"


## ###############################################################
## FIND SHORTEST PATH BETWEEN TWO POINTS
## ###############################################################
def aStar(verts, dict_adj, start_vi, end_vi) -> list:
  ## create start and finish nodes
  start_node = Node(vi = start_vi)
  end_node   = Node(vi = end_vi)
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
    current_ni = 0 # list of nodes index (ni)
    current_node : Node = list_opened_nodes[0]
    for node_ni, node in enumerate(list_opened_nodes):
      if (node < current_node):
        current_ni   = node_ni
        current_node = node
    ## remove the current node from the open list
    list_opened_nodes.pop(current_ni)
    ## CHECK IF THE GOAL HAS BEEN REACHED
    ## ==================================
    if current_node == end_node:
      soln_path = []
      ## trace solution path backwards from last to start node
      prev_node : Node = current_node
      while prev_node is not None:
        soln_path.append(verts[prev_node.vi,:])
        prev_node = prev_node.parent_node
      ## store solution details
      soln_paths_grouped.append(soln_path[::-1])
      soln_costs_grouped.append(current_node.total_dist)
    ## GENERATE LIST OF NEIGHBOURING POINTS
    ## ====================================
    for neighbour_vi in dict_adj[current_node.vi]:
      neighbour_node = Node(
        parent_node = current_node,
        vi          = neighbour_vi
      )
      ## define score values
      step_size = euclidean(
        verts[current_node.vi,:],
        verts[neighbour_node.vi,:]
      )
      dist_to_go = euclidean(
        verts[neighbour_node.vi,:],
        verts[end_node.vi,:]
      )
      neighbour_node.dist_trvld = current_node.dist_trvld + step_size
      neighbour_node.dist_to_go = dist_to_go
      neighbour_node.total_dist = neighbour_node.dist_trvld + neighbour_node.dist_to_go
      ## check that the neighbour is not already in the open list
      if any([
        (node == neighbour_node) and (node < neighbour_node)
        for node in list_opened_nodes
      ]): continue
      ## check that the neighbour is not already in the closed list
      if any([
        (node == neighbour_node) and (node < neighbour_node)
        for node in list_closed_nodes
      ]): continue
      ## if the neighbour is in either the open or closed lists,
      ## then only consider it again if the score has improved
      list_opened_nodes.append(neighbour_node)
    ## add the node to the closed list
    list_closed_nodes.append(current_node)
  ## return solutions
  return soln_paths_grouped, soln_costs_grouped


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  start_time = time.time()
  ## generate data
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