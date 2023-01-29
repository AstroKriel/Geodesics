## ###############################################################
## MODULES
## ###############################################################
import os, warnings, functools
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

## import user libraries
from MyLibrary import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
warnings.simplefilter('ignore', UserWarning) # hide warnings


## ###############################################################
## SOLUTION TO GEODESIC EQUATION FOR TORUS
## ###############################################################
def geodesicEqnSoln(x, t, r1=2, r2=1):
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
  return [ dtheta_dt, dphi_dt ]


## ###############################################################
## TORUS EQUATIONS
## ###############################################################
class Torus():
  def implicit(x, y, z, r2=2, r1=1):
    return z*z - r2*r2 + ( r1 - np.sqrt(x*x + y*y) )**2

  def parameterized(theta, phi, r1=2, r2=1):
    x = (r1 + r2*np.cos(theta)) * np.cos(phi)
    y = (r1 + r2*np.cos(theta)) * np.sin(phi)
    z = r2 * np.sin(theta)
    return x, y, z

## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## torus parameters
  r1 = 2.0
  r2 = 1.0
  ax_max = r1 + r2
  ## initialise figure
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ## plot torus isosurface
  PlotFuncs.plotImplicit(
    ax,
    func        = functools.partial(Torus.implicit, r1=r1, r2=r2),
    bbox        = (-ax_max, ax_max),
    res_contour = 100,
    res_slices  = 30,
    alpha       = 0.25
  )
  ## plot the geodesic equation soln
  theta = -np.pi/4
  phi   = 0.0
  num_geo_orbits = 100
  num_geo_points = 10**3
  soln = odeint(
    func = functools.partial(geodesicEqnSoln, r1=r1, r2=r2),
    y0   = [ theta, phi ],
    t    = np.linspace(0, 2*np.pi*num_geo_orbits, num_geo_points)
  )
  [ x_geo, y_geo, z_geo ] = Torus.parameterized(
    theta = soln[:, 0],
    phi   = soln[:, 1],
    r1    = r1,
    r2    = r2
  )
  ## plot geodesic
  ax.plot(x_geo, y_geo, z_geo, color="r", ls="-", lw=2, zorder=5)
  ## #################
  ## SHOW FIGURE
  ## #################
  ax.set_xlim(-ax_max, ax_max)
  ax.set_ylim(-ax_max, ax_max)
  ax.set_zlim(-ax_max, ax_max)
  ax.view_init(36, 26)
  plt.show()


## ###############################################################
## RUN MAIN
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM