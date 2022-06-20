#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

## import user libraries
from MyLibrary import MyTools, PlotFuncs

## ###############################################################
## PREPARE TERMINAL / WORKSPACE / CODE
## ###############################################################
os.system("clear") # clear terminal window
warnings.simplefilter('ignore', UserWarning) # hide warnings


## ###############################################################
## HELPER FUNCTION: GENERATE POINT CLOUD FROM PARAMETERISED SHAPE
## ###############################################################
def pointCloudFromParamEqn():
  ## input arguments
  res = 100
  ## TODO: user should specify how many variables the func is parameterised with respect to
  ## (will it always be 2?)
  theta_bounds = [ 0, 2*np.pi ]
  phi_bounds   = [ 0, 2*np.pi ]
  ## calculate domain-mesh
  theta_input, phi_input = np.meshgrid(
    np.linspace(theta_bounds[0], theta_bounds[1], res),
    np.linspace(phi_bounds[0],   phi_bounds[1],   res)
  )
  ## TODO: this needs to be a lambda-ised input function
  return MyTools.Torus.parameterized(
    ## user needs to specify that these are the domain-variables
    theta = theta_input,
    phi   = phi_input,
    ## user can pre-define these with: functools.partial
    r1    = 2,
    r2    = 1
  )

## ###############################################################
## MAIN PROGRAM
## ###############################################################
BOOL_DEBUG = 0
# ## template for debugging
# import pdb; pdb.set_trace()

def main():
  ## initialise figure
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ## generate point cloud
  x_surf, y_surf, z_surf = pointCloudFromParamEqn()
  ## TODO: calculate furthest point away in a particular direction from some origin
  ## TODO: calculate closest point in point-cloud to a reference point
  ## plot surface
  ax.plot_surface(x_surf, y_surf, z_surf, rstride=5, cstride=5)
  ## show plot
  ax.view_init(36, 26)
  plt.show()


## ###############################################################
## RUN MAIN
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM