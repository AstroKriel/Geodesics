## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy
import skimage


## ###############################################################
## FUNCTIONS
## ###############################################################
def generate_mesh_from_point_cloud(
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


## END OF MODULE