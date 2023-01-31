## ###############################################################
## MODULES
## ###############################################################
import time, skimage
import numpy as np
import multiprocessing as mproc

from  mayavi import mlab
from scipy.spatial import KDTree


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
  print(f"{pre}{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}")


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
  res = 20
  verts, faces, normals = genPointCloud(implicit_func=gyroid, res=res)
  print("Computing adjacency dictionary...")
  dict_adj = genAdjDict_parallel(verts.shape[0], faces.shape[0], faces)
  ## define start and end points
  tree = KDTree(verts)
  _, start_vi = tree.query([20, 20,  0])
  _, end_vi   = tree.query([ 0, 10, 20])
  start_point = verts[start_vi,:]
  end_point   = verts[end_vi,:]
  printPoint(start_point, "start point: ")
  printPoint(end_point,   "end point:   ")
  ## compute possible solutions
  print("Searching for geodesic path...")
  soln_path_grouped, soln_cost_grouped = aStar(verts, dict_adj, start_vi, end_vi)
  if len(soln_path_grouped) > 0:
    str_end = "s!" if len(soln_path_grouped) > 1 else "!"
    print(f"Found {len(soln_path_grouped)} possible geodesic path" + str_end)
    ## find index of the best (shortest) solution
    soln_index = np.argmin(soln_cost_grouped)
    soln_path = soln_path_grouped[soln_index]
    ## plot best solution
    x_geo = [ pos[0] for pos in soln_path ]
    y_geo = [ pos[1] for pos in soln_path ]
    z_geo = [ pos[2] for pos in soln_path ]
  else: print("Could not find a solution!")
  ## plot isosurface
  mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces, representation="surface")
  mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces, representation="mesh")
  ## plot geodesic
  if len(soln_path_grouped) > 0:
    mlab.points3d(start_point[0], start_point[1], start_point[2])
    mlab.points3d(end_point[0], end_point[1], end_point[2])
    mlab.plot3d(x_geo, y_geo, z_geo, range(len(x_geo)), tube_radius=0.25)
  ## plot pseudo-axis
  mlab.points3d(0, 0, 0)
  mlab.points3d(2, 0, 0)
  mlab.points3d(0, 4, 0)
  mlab.points3d(0, 0, 6)
  mlab.plot3d([0, 2.0], [0, 0], [0, 0], tube_radius=0.1)
  mlab.plot3d([0, 0], [0, 4.0], [0, 0], tube_radius=0.1)
  mlab.plot3d([0, 0], [0, 0], [0, 6.0], tube_radius=0.1)
  mlab.points3d(res,res,res)
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