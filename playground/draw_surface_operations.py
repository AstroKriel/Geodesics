## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import pyvista
from geodesics.generate_surfaces import implicit_surfaces, surface_operations, generate_surface


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  surfaces_to_plot = []
  ## basic shapes
  box    = implicit_surfaces.box(size=1.0)
  sphere = implicit_surfaces.sphere(radius=1.0)
  ## shape 1: Box
  shape_1 = box
  surfaces_to_plot.append(shape_1)
  ## shape 2: sphere
  shape_2_func = surface_operations.translate(sphere, 3, 0, 0)
  surfaces_to_plot.append(shape_2_func)
  ## shape 3: box with a sphere cut out
  shape_3_sphere = surface_operations.translate(sphere, 6, 0, 0)
  shape_3_box    = surface_operations.translate(box, 6, 0, 0)
  shape_3        = surface_operations.subtract(shape_3_box, shape_3_sphere)
  surfaces_to_plot.append(shape_3)
  ## shape 4: rotated box on a sphere
  shape_4_sphere = surface_operations.translate(sphere, 9, 0, 0)
  shape_4_box    = surface_operations.rotate_xy(box, 45)
  shape_4_box    = surface_operations.rotate_yz(shape_4_box, 45)
  shape_4_box    = surface_operations.translate(shape_4_box, 9, 0, 2)
  shape_4        = surface_operations.union(shape_4_sphere, shape_4_box)
  surfaces_to_plot.append(shape_4)
  ## shape 5: intersection of a sphere and box
  shape_5_sphere1 = surface_operations.translate(sphere, 11, 0, 0)
  shape_5_sphere2 = surface_operations.translate(sphere, 13, 0, 0)
  shape_5_sphere_union = surface_operations.union(shape_5_sphere1, shape_5_sphere2)
  shape_5_box     = surface_operations.translate(box, 12, 0, 0)
  shape_5         = surface_operations.intersect(shape_5_box, shape_5_sphere_union)
  surfaces_to_plot.append(shape_5)
  ## plot all shapes
  plotter = pyvista.Plotter(window_size=(3*600, 3*600))
  for func in surfaces_to_plot:
    vertices, faces, _ = generate_surface.generate_surface_mesh(
      implicit_func = func,
      domain_bounds = [-2, 15],
      num_points    = 200
    )
    num_faces = faces.shape[0]
    faces_pv  = numpy.hstack([numpy.full((num_faces, 1), 3), faces]).flatten()
    mesh_pv   = pyvista.PolyData(vertices, faces_pv)
    plotter.add_mesh(mesh_pv, show_edges=False, color="white", opacity=0.7)
  plotter.show_bounds(
    grid      = "all",
    location  = "outer",
    ticks     = "outside",
    xtitle    = "X",
    ytitle    = "Y",
    ztitle    = "Z",
    font_size = 20,
    color     = "black",
    use_2d    = False
  )
  plotter.show_axes()
  plotter.show()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
