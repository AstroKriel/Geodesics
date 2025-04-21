## ###############################################################
## DEPENDANCIES
## ###############################################################
import time
import numpy
import pyvista
from scipy.spatial import KDTree
from geodesics.generate_surfaces import implicit_surfaces, generate_surface
from geodesics.graph_search import a_star_3d


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def time_function(func):
  def wrapper(*args, **kwargs):
    time_start = time.time()
    try:
      result = func(*args, **kwargs)
    except Exception as error:
      raise RuntimeError(f"Error occurred in {func.__name__}() while measuring the elapsed time.") from error
    elapsed_time = time.time() - time_start
    print(f"{func.__name__}() took {elapsed_time:.3f} seconds to execute.")
    return result
  return wrapper


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  ## generate surface mesh from implicit function
  vertices, faces, _ = generate_surface.generate_surface_mesh(
    implicit_func = implicit_surfaces.goursat_tangle(),
    domain_bounds = [-5, 5],
    num_points    = 50,
  )
  ## find closest vertices to start and end point
  start_point = [0, 0, 0]
  end_point   = [72, 72, 72]
  tree = KDTree(vertices)
  _, start_index = tree.query(start_point)
  _, end_index   = tree.query(end_point)
  ## convert faces to PyVista format
  num_faces = faces.shape[0]
  faces_pv  = numpy.hstack([numpy.full((num_faces, 1), 3), faces]).flatten()
  mesh_pv   = pyvista.PolyData(vertices, faces_pv)
  ## define functions wrapped with timing-decorator
  @time_function
  def _generate_adjacency_map():
    return generate_surface.generate_adjacency_map_parallel(vertices, faces)
  @time_function
  def _run_a_star():
    return a_star_3d(vertices, adjacency_map, start_index, end_index)
  ## find geodesic
  adjacency_map = _generate_adjacency_map()
  solution_paths, solution_costs = _run_a_star()
  if len(solution_paths) > 0:
    print(f"Solution found: Path length = {len(solution_paths[0])}, Total cost = {solution_costs[0]:.4f}")
  else: print("No solution found.")
  print(" ")
  ## plot mesh with start and end points
  plotter = pyvista.Plotter(window_size=(3*600, 3*600))
  plotter.add_mesh(mesh_pv, show_edges=True, color="white", opacity=0.5)
  plotter.add_mesh(pyvista.Sphere(radius=1, center=vertices[start_index]), color="blue")
  plotter.add_mesh(pyvista.Sphere(radius=1, center=vertices[end_index]), color="red")
  ## draw the first solution found
  if len(solution_paths) > 0:
    solution_points = numpy.array(solution_paths[0])
    for point in solution_points:
      ## add small spheres at each point along the solution-path
      plotter.add_mesh(pyvista.Sphere(radius=0.25, center=point), color="black")
    num_points = solution_points.shape[0]
    ## connect spheres with lines
    lines = numpy.hstack(
      ([num_points], numpy.arange(num_points))
    ).astype(numpy.int32)
    solution_points_pv = pyvista.PolyData()
    solution_points_pv.points = solution_points
    solution_points_pv.lines = lines
    solution_tube = solution_points_pv.tube(radius=0.1)
    plotter.add_mesh(solution_tube, color="black")
  ## add axes and bounds for orientation
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