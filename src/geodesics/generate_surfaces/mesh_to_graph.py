## ###############################################################
## DEPENDANCIES
## ###############################################################
import multiprocessing


## ###############################################################
## FUNCTIONS
## ###############################################################
def generate_adjacency_map(vertices, faces):
  num_vertices = vertices.shape[0]
  num_faces = faces.shape[0]
  dict_adj = {}
  ## find all vertices that are part of the same face: containing the current vertex
  for point_index in range(num_vertices):
    dict_adj[point_index] = [
      elem
      for face_index in range(num_faces)
      for elem in faces[face_index]
      if point_index in faces[face_index] and (elem != point_index)
    ]
  return dict_adj

def _find_adjacent_vertices(point_index, num_faces, faces):
    return point_index, [
      elem
      for face_index in range(num_faces)
      for elem in faces[face_index]
      if point_index in faces[face_index] and (elem != point_index)
    ]

def generate_adjacency_map_parallel(vertices, faces):
  num_vertices = vertices.shape[0]
  num_faces = faces.shape[0]
  with multiprocessing.Pool() as pool_obj:
    return dict(pool_obj.starmap(
      _find_adjacent_vertices,
      [
        (point_index, num_faces, faces)
        for point_index in range(num_vertices)
      ]
    ))


## END OF MODULE