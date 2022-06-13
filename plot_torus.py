#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.integrate import odeint

## import user libraries
from MyLibrary import MyTools, PlotFuncs


## ###############################################################
## PREPARE TERMINAL / WORKSPACE / CODE
## ###############################################################
os.system("clear") # clear terminal window
warnings.simplefilter('ignore', UserWarning) # hide warnings


## ###############################################################
## FUNCTIONS
## ###############################################################
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


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## torus parameters
  r1, r2 = 2, 1
  ax_max = r1 + r2
  ## generate torus data points
  res_surface  = 100
  angle_domain = np.linspace(0, 2.*np.pi, res_surface)
  theta_input, phi_input = np.meshgrid(angle_domain, angle_domain)
  [ x_torus, y_torus, z_torus ] = MyTools.Torus.parameterized(
    theta = theta_input,
    phi   = phi_input,
    r1    = r1,
    r2    = r2 * 0.9
  )
  func_shape = partial(
    MyTools.Torus.implicit,
    r1 = r1,
    r2 = r2
  )
  ## initialise figure
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ## plot surface of the torus
  # ax.plot_surface(x_torus, y_torus, z_torus, rstride=5, cstride=5)
  PlotFuncs.plotImplicit(
    ax, func_shape,
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
  func_geodesic = partial(
    solnToGeodesicEqn,
    r1 = r1,
    r2 = r2
  )
  soln = odeint(func_geodesic, x0, t)
  theta_output = soln[:, 0]
  phi_output   = soln[:, 1]
  [ x_geo, y_geo, z_geo ] = MyTools.Torus.parameterized(
    theta = theta_output,
    phi   = phi_output,
    r1    = r1,
    r2    = r2
  )
  ## plot geodesic
  ax.plot(x_geo, y_geo, z_geo, color="r", ls="-", lw=2, zorder=5)
  ## show plot
  ax.view_init(36, 26)
  plt.show()


## ###############################################################
## RUN MAIN
## ###############################################################
if __name__ == "__main__":
  main()


## ###############################################################
## END OF PROGRAM
## ###############################################################