## ###############################################################
## MODULES
## ###############################################################
import os, re
import numpy as np

from MyLibrary import TheUsefuls


## ###############################################################
## FUNCTIONS
## ###############################################################
def animateFrames(filepath_frames, shape_name):
  print("Animating plots...")
  ## create filepath to where plots are saved
  filepath_input = TheUsefuls.createFilePath([filepath_frames, shape_name + "_%*.png"])
  ## create filepath to where animation should be saved
  filepath_output = TheUsefuls.createFilePath([filepath_frames, "..", shape_name + ".mp4"])
  ## animate plot frames
  os.system("ffmpeg -y -start_number 0 -i {} -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 {}".format(
    filepath_input,
    filepath_output
  ))
  print("Animation finished: " + filepath_output)
  print(" ")


def plotImplicitShape(
    ax, implicit_func,
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
    X = implicit_func(coord, A1, A2)
    Y = implicit_func(A1, coord, A2)
    Z = implicit_func(A1, A2, coord)
    ## plot contours in the YZ plane
    try: ax.contour(X+coord, A1, A2, [coord], zdir="x", colors=color, alpha=alpha, linewidth=2, linestyle="-")
    except UserWarning: continue
    ## plot contours in the XZ plane
    try: ax.contour(A1, Y+coord, A2, [coord], zdir="y", colors=color, alpha=alpha, linewidth=2, linestyle="-")
    except UserWarning: continue
    ## plot contours in the XY plane
    try: ax.contour(A1, A2, Z+coord, [coord], zdir="z", colors=color, alpha=alpha, linewidth=2, linestyle="-")
    except UserWarning: continue


## ###############################################################
## END OF LIBRARY
## ###############################################################