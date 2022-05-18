import os, sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from functools import partial
from scipy.integrate import odeint

os.system("clear") # clear terminal window
warnings.simplefilter("ignore", UserWarning) # hide warnings

class Torus():
  def implicit(x, y, z, r2=2, r1=1):
    return z*z - r2*r2 + ( r1 - np.sqrt(x*x + y*y) )**2
  def parameterized(theta, phi, r1=2, r2=1):
    x = (r1 + r2*np.cos(theta)) * np.cos(phi)
    y = (r1 + r2*np.cos(theta)) * np.sin(phi)
    z = r2 * np.sin(theta)
    return [x, y, z]

def solnToGeodesicEqn(x, t, r1=2, r2=1):
  ## integration constants
  k = r1 + r2/2 # k = h
  l = 1 / (r2 * r2)
  ## extract variables
  theta, phi = x[0], x[1]
  ## coupled ODEs
  dtheta_dt = np.sqrt(np.max([
    0,
    l - k*k / (r1*r2 + r2*r2*np.cos(theta))**2
  ]))
  dphi_dt = k / (r1 + r2*np.cos(theta))**2
  return [dtheta_dt, dphi_dt]

def plotImplicit(
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
  ## define plot style
  args = {"colors":color, "alpha":alpha, "zorder":3}
  ## plot each contour
  for coord in coords_slice:
    ## evaluate the function
    X = shape_func(coord, A1, A2)
    Y = shape_func(A1, coord, A2)
    Z = shape_func(A1, A2, coord)
    ## plot contours in the YZ plane
    try: ax.contour(X+coord, A1, A2, [coord], zdir="x", **args)
    except UserWarning: continue
    ## plot contours in the XZ plane
    try: ax.contour(A1, Y+coord, A2, [coord], zdir="y", **args)
    except UserWarning: continue
    ## plot contours in the XY plane
    try: ax.contour(A1, A2, Z+coord, [coord], zdir="z", **args)
    except UserWarning: continue

def main():
  ## torus parameters
  r1, r2 = 2, 1
  ax_max = r1 + r2
  ## generate torus data points
  res_surface  = 100
  angle_domain = np.linspace(0, 2.*np.pi, res_surface)
  theta_input, phi_input = np.meshgrid(angle_domain, angle_domain)
  [x_torus, y_torus, z_torus] = Torus.parameterized(theta_input, phi_input, r1, r2*0.9)
  shape_func = partial(
    Torus.implicit,
    r1 = r1,
    r2 = r2
  )
  ## initialise figure
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ## plot surface of the torus
  # ax.plot_surface(x_torus, y_torus, z_torus, rstride=5, cstride=5)
  plotImplicit(
    ax, shape_func,
    bbox = (-ax_max, ax_max),
    res_contour = 100,
    res_slices  = 30,
    alpha       = 0.25
  )
  ax.set_xlim(-ax_max, ax_max)
  ax.set_ylim(-ax_max, ax_max)
  ax.set_zlim(-ax_max, ax_max)
  ## solve the geodesic equation
  num_geo_orbits = 100
  num_geo_points = 10**3
  t  = np.linspace(0, num_geo_orbits*2*np.pi, num_geo_points)
  x0 = [-np.pi/4, 0] # angles: theta, phi
  geodesic_func = partial(
    solnToGeodesicEqn,
    r1 = r1,
    r2 = r2
  )
  soln = odeint(geodesic_func, x0, t)
  theta_output = soln[:, 0]
  phi_output   = soln[:, 1]
  [x_geo, y_geo, z_geo] = Torus.parameterized(theta_output, phi_output, r1, r2)
  ## plot geodesic
  ax.plot(x_geo, y_geo, z_geo, color="r", ls="-", lw=2, zorder=5)
  ## show plot
  ax.view_init(36, 26)
  plt.show()

if __name__ == "__main__":
  main()
  sys.exit()
