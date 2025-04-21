## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import pyvista
from geodesics.generate_surfaces import implicit_surfaces, generate_surface


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  ## plot all shapes
  plotter = pyvista.Plotter(window_size=(3*600, 3*600))
  ## available interesting shapes: goursat_tangle, gyroid, tunnels
  vertices, faces, _ = generate_surface.generate_surface_mesh(
    implicit_func = implicit_surfaces.goursat_tangle(),
    domain_bounds = [-10, 10],
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
