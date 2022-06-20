## ###############################################################
## MODULES
## ###############################################################
import os
import numpy as np

from MyLibrary import UsefulFuncs


## ###############################################################
## ANIMATING PNG-FRAMES INTO MP4
## ###############################################################
def animateFrames(filepath_frames, shape_name):
  print("Animating plots...")
  ## create filepath to where plots are saved
  filepath_input = UsefulFuncs.createFilePath([filepath_frames, shape_name + "_%*.png"])
  ## create filepath to where animation should be saved
  filepath_output = UsefulFuncs.createFilePath([filepath_frames, "..", shape_name + ".mp4"])
  ## animate plot frames
  os.system(f"ffmpeg -y -start_number 0 -i {filepath_input} -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 {filepath_output}")
  print("Animation finished: " + filepath_output)
  print(" ")


## ###############################################################
## PLOT IMPLICITLY DEFINED SURFACE
## ###############################################################
## from: https://stackoverflow.com/a/4687582
def plotImplicit(
    ax, func_shape,
    bbox        = (-1, 1),
    res_contour = 70,
    res_slices  = 30,
    color       = "black",
    alpha       = 0.75
  ):
  ## define plot style
  args = {"colors":color, "alpha":alpha, "zorder":3}
  ## extract box domain bounds
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
    X = func_shape(coord, A1, A2)
    Y = func_shape(A1, coord, A2)
    Z = func_shape(A1, A2, coord)
    ## plot contours in the YZ plane
    try: ax.contour(X+coord, A1, A2, [coord], zdir="x", **args)
    except UserWarning: continue
    ## plot contours in the XZ plane
    try: ax.contour(A1, Y+coord, A2, [coord], zdir="y", **args)
    except UserWarning: continue
    ## plot contours in the XY plane
    try: ax.contour(A1, A2, Z+coord, [coord], zdir="z", **args)
    except UserWarning: continue


## END OF LIBRARY