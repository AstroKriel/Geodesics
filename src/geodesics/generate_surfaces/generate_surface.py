## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy
import skimage
import multiprocessing


## ###############################################################
## FUNCTIONS
## ###############################################################
def generate_surface_mesh(
    implicit_func,
    domain_bounds : tuple[float, float] = (-numpy.pi, numpy.pi),
    num_points    : int = 100
  ):
  values = numpy.linspace(domain_bounds[0], domain_bounds[1], num_points)
  x_3d, y_3d, z_3d = numpy.meshgrid(values, values, values)
  ## evaluate the function across the volume
  sfield_values = implicit_func(x_3d, y_3d, z_3d)
  ## the isosurface is defined as where the function is zero
  vertices, faces, normals, _ = skimage.measure.marching_cubes(sfield_values, 0)
  return vertices, faces, normals

def generate_adjacency_map(vertices, faces):
  num_vertices = vertices.shape[0]
  adjacency_map = {}
  ## find all vertices that are part of the same face: containing the current vertex
  for vertex_index in range(num_vertices):
    neighbors = set()
    for face in faces:
      if vertex_index in face:
        neighbors.update(
          neighbor_index
          for neighbor_index in face
          if neighbor_index != vertex_index
        )
    adjacency_map[vertex_index] = list(neighbors)
  return adjacency_map

def _create_vertex_face_dict(faces):
  vertex_faces = {}
  for face_index, face in enumerate(faces):
    for vertex in face:
      if vertex not in vertex_faces: vertex_faces[vertex] = []
      vertex_faces[vertex].append(face_index)
  return vertex_faces

def _find_neighbors_for_vertex(vertex_index, faces, vertex_faces):
  neighbors = set()
  for face_index in vertex_faces[vertex_index]:
    face = faces[face_index]
    neighbors.update(
      neighbor
      for neighbor in face
      if neighbor != vertex_index
    )
  return vertex_index, list(neighbors)

def generate_adjacency_map_parallel(vertices, faces):
  num_vertices = vertices.shape[0]
  vertex_faces = _create_vertex_face_dict(faces)
  with multiprocessing.Pool() as pool:
    adjacency_items = pool.starmap(
      _find_neighbors_for_vertex,
      [
        (vertex_index, faces, vertex_faces)
        for vertex_index in range(num_vertices)
      ]
    )
  return dict(adjacency_items)



## END OF MODULE