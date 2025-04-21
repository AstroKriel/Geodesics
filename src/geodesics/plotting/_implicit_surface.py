## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
## based on: https://stackoverflow.com/a/4687582
def plot_surface_contours(
    ax,
    implicit_func,
    domain_bounds      = (-1, 1),
    contour_resolution = 70,
    slice_resolution   = 30,
    color              = "black",
    alpha              = 0.75,
    zorder             = 3,
  ):
  plot_args      = dict(color=color, alpha=alpha, zorder=zorder)
  contour_coords = numpy.linspace(domain_bounds[0], domain_bounds[1], contour_resolution)
  slice_coords   = numpy.linspace(domain_bounds[0], domain_bounds[1], slice_resolution)
  A1, A2 = numpy.meshgrid(contour_coords, contour_coords)
  for slice_coord in slice_coords:
    X = implicit_func(slice_coord, A1, A2)
    Y = implicit_func(A1, slice_coord, A2)
    Z = implicit_func(A1, A2, slice_coord)
    ## plot contours in the YZ plane
    try: ax.contour(X+slice_coord, A1, A2, [slice_coord], zdir="x", **plot_args)
    except UserWarning: continue
    ## plot contours in the XZ plane
    try: ax.contour(A1, Y+slice_coord, A2, [slice_coord], zdir="y", **plot_args)
    except UserWarning: continue
    ## plot contours in the XY plane
    try: ax.contour(A1, A2, Z+slice_coord, [slice_coord], zdir="z", **plot_args)
    except UserWarning: continue


## END OF MODULE