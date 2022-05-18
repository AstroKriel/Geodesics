#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import warnings
import functools
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

from tqdm.auto import tqdm

## ###############################################################
## PREPARE TERMINAL/WORKSPACE/CODE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend
warnings.simplefilter("ignore", UserWarning) # hide warnings


## ###############################################################
## USER DEFINED FUNCTIONS
## ###############################################################
class ImplicitShapes():
  def sphere(x, y, z, r=1):
    return x*x + y*y + z*z - r*r
  def sphere_z(x, y, r=1):
    val_to_sqrt = r*r - x*x - y*y
    if val_to_sqrt < 0: return np.inf
    else: z = np.sqrt( val_to_sqrt )
    return z
  def torus(x, y, z, r1=2, r2=1):
    return z*z - r2*r2 + ( r1 - np.sqrt(x*x + y*y) )**2
  def torus_z(x, y, r1=2, r2=1):
    val_to_sqrt = r2*r2 - ( r1 - np.sqrt(x*x + y*y) )**2
    if val_to_sqrt < 0: return np.inf
    else: z = np.sqrt( val_to_sqrt )
    return z

def funcPlotImplicit(
    ax, shape_func,
    bbox = (-1, 1),
    res_contour = 70,
    res_slices  = 30,
    color = "black",
    alpha = 0.75
  ):
  ## get box domain
  ax_min, ax_max = bbox
  ## initialise the coordinates where contours are evaluated
  coords_contour = np.linspace(ax_min, ax_max, res_contour) 
  ## initialise the slice coordinates (along each direction)
  coords_slice = np.linspace(ax_min, ax_max, res_slices)
  ## initialise the grid for plotting contours
  A1, A2 = np.meshgrid(coords_contour, coords_contour)
  ## plot each contour
  for coord in coords_slice:
    ## evaluate the function
    X = shape_func(coord, A1, A2)
    Y = shape_func(A1, coord, A2)
    Z = shape_func(A1, A2, coord)
    ## plot contours in the YZ plane
    try: ax.contour(X+coord, A1, A2, [coord], zdir="x", colors=color, alpha=alpha, linewidth=2, linestyle="-")
    except UserWarning: continue
    ## plot contours in the XZ plane
    try: ax.contour(A1, Y+coord, A2, [coord], zdir="y", colors=color, alpha=alpha, linewidth=2, linestyle="-")
    except UserWarning: continue
    ## plot contours in the XY plane
    try: ax.contour(A1, A2, Z+coord, [coord], zdir="z", colors=color, alpha=alpha, linewidth=2, linestyle="-")
    except UserWarning: continue

def funcCalcTangent(
    x, y, z,
    func,
    x0, y0, z0
  ):
  epsilon = 1e-5
  grad_x = ( func(x0+epsilon, y0, z0) - func(x0-epsilon, y0, z0) ) / (2*epsilon)
  grad_y = ( func(x0, y0+epsilon, z0) - func(x0, y0-epsilon, z0) ) / (2*epsilon)
  grad_z = ( func(x0, y0, z0+epsilon) - func(x0, y0, z0-epsilon) ) / (2*epsilon)
  return grad_x * (x - x0) + grad_y * (y - y0) + grad_z * (z - z0)

def funcCalcIntercepts(
    func_z,
    x0, y0, z0,
    x_dir, y_dir, z_dir
  ):
  ## define parameters
  max_step_count = 1000
  delta_step = 0.01 # step size
  delta_tol  = 0.1 # intercept tolerance
  x_next, y_next, z_next = x0, y0, z0 # initialise current point
  list_intercepts = [] # initialise list of intercepts found
  ## integrate along line
  for _ in range(max_step_count):
    ## calculate next point
    x_next += delta_step * x_dir
    y_next += delta_step * y_dir
    z_next += delta_step * z_dir
    ## calculate value of the curve at this point (x, y)
    z_curve = func_z(x_next, y_next)
    ## check if the next point lies suficiently close to the curve
    z_dist = abs(z_curve - z_next)
    ## if the point is sufficiently close, then store the intercept point
    if z_dist < delta_tol:
      list_intercepts.append([ x_next, y_next, z_next ])
  ## return information
  if len(list_intercepts) > 0:
    return len(list_intercepts), list_intercepts
  else: return 0, None

def funcPlotHelicalCurve(ax, R1, R2, k=0.01, T=1):
  ## https://math.stackexchange.com/questions/324527/do-these-equations-create-a-helix-wrapped-into-a-torus
  theta = np.linspace(0, 2*np.pi * T/k, 10000)
  phi = k * theta
  x = (R1 + R2 * np.cos(theta)) * np.cos(phi)
  y = (R1 + R2 * np.cos(theta)) * np.sin(phi)
  z = R2 * np.sin(theta)
  ax.plot3D(
    x, y, z,
    color = "red",
    linewidth = 1.5
  )
  return

# def funcMeanCurvature(Z):
#     Zy, Zx   = np.gradient(Z)
#     Zxy, Zxx = np.gradient(Zx)
#     Zyy, _   = np.gradient(Zy)
#     H = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx
#     H = -H / (2*(Zx**2 + Zy**2 + 1)**(1.5))
#     return H

# def funcGaussianCurvature(Z):
#     Zy, Zx   = np.gradient(Z)
#     Zxy, Zxx = np.gradient(Zx)
#     Zyy, _   = np.gradient(Zy)
#     K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2
#     return K


BOOL_PLOT_SHAPE          = 1
BOOL_PLOT_ORIGIN         = 0
BOOL_PLOT_ALL_INTERCEPTS = 0
BOOL_PLOT_LAST_INTERCEPT = 0
BOOL_PLOT_TANGENT        = 0
BOOL_PLOT_HELIX_PATH     = 1
## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## initialise figure
  fig = plt.figure(figsize=(7.5, 7.5), constrained_layout=True)
  ax = plt.axes(projection="3d")
  ## draw sphere
  R1 = 1.5
  R2 = 0.5
  max_dist = 1.2 * (R1 + R2)
  ax_min, ax_max = -max_dist, max_dist
  func = functools.partial(
    ImplicitShapes.torus,
    r1 = R1,
    r2 = R2
  )
  func_z = functools.partial(
    ImplicitShapes.torus_z,
    r1 = R1,
    r2 = R2
  )
  if BOOL_PLOT_SHAPE:
    funcPlotImplicit(
      ax, func,
      bbox = (ax_min, ax_max),
      res_contour = 100,
      res_slices  = 30
    )
  print("\t> Plotted implicit shape.")
  ## starting (origin) point
  x0, y0, z0 = 0, 0, 0
  if BOOL_PLOT_ORIGIN:
    ax.scatter(x0, y0, z0, color="red", edgecolors="black", s=100)
  print("\t> Plotted origin point.")
  ## find points of intersection between line and curve
  x_dir, y_dir, z_dir = 1.0, 0.0, 0.0 # vector
  mag_dir  = x_dir*x_dir + y_dir*y_dir + z_dir*z_dir # calculate magnitude of vector
  x_dir_norm = x_dir / mag_dir
  y_dir_norm = y_dir / mag_dir
  z_dir_norm = z_dir / mag_dir
  num_intercepts, list_intercepts = funcCalcIntercepts(
    func_z,
    x0, y0, z0,
    x_dir_norm, y_dir_norm, z_dir_norm
  )
  ## for debugging. TODO: remove
  if BOOL_PLOT_ALL_INTERCEPTS:
    if num_intercepts > 0:
      ## plot all intercepts found
      for intercept_iter in range(num_intercepts):
        x_int, y_int, z_int = list_intercepts[intercept_iter]
        ax.scatter(x_int, y_int, z_int, color="blue", edgecolors="black", s=100)
  ## draw the last intercept
  if num_intercepts > 0:
    print("\t> Found {} point(s) of intersection.".format(num_intercepts))
    print("\t> Plotted final intercept.")
    if BOOL_PLOT_LAST_INTERCEPT:
      if num_intercepts == 1:
        x_int, y_int, z_int = list_intercepts
      else: x_int, y_int, z_int = list_intercepts[-1]
      ax.scatter(x_int, y_int, z_int, color="blue", edgecolors="black", s=100)
    ## draw approximated tangent plane
    if BOOL_PLOT_TANGENT:
      funcPlotImplicit(
        ax,
        functools.partial(
          funcCalcTangent,
          func = func,
          x0=x_int, y0=y_int, z0=z_int
        ),
        bbox = (ax_min, ax_max),
        res_contour = 50,
        res_slices  = 20,
        color = "green"
      )
    ## draw helical path
    if BOOL_PLOT_HELIX_PATH:
      funcPlotHelicalCurve(
        ax,
        R1, R2,
        k = 1 / 8,
        T = 1
      )
  ## set figure window bounds
  ax.set_xlim3d(ax_min, ax_max)
  ax.set_ylim3d(ax_min, ax_max)
  ax.set_zlim3d(ax_min, ax_max)
  ## add axis labels
  ax.set_xlabel(r"x", fontsize=20)
  ax.set_ylabel(r"y", fontsize=20)
  ax.set_zlabel(r"z", fontsize=20)
  ## set axis equal
  ax.set_box_aspect([1, 1, 1])
  ## change view axis
  ax.view_init(
    elev = 30,
    azim = 65
  )
  ## save image
  plt.savefig("fig.png")
  print("\t> Saved figure.")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  print("Beginning of program.")
  print("=====================")
  main()
  print("=====================")
  print("End of program.")
  sys.exit()


## ###############################################################
## END OF PROGRAM
## ###############################################################