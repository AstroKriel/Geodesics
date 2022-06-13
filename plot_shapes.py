#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
from turtle import Shape
import warnings
import functools
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

## import user libraries
from MyLibrary import MyTools, PlotFuncs, UsefulFuncs


## ###############################################################
## PREPARE TERMINAL / WORKSPACE / CODE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend('agg') # use a non-interactive plotting backend
warnings.simplefilter('ignore', UserWarning) # hide warnings


## ###############################################################
## PLOTTING FUNCTION
## ###############################################################
def funcPlotShape(
    filepath_plot,
    shape_name, shape_func,
    list_angles = [0],
    bbox        = (-1, 1),
    num_cols    = 15,
    res_contour = 70,
    res_slices  = 30,
    bool_dark_theme  = True,
    bool_multicolors = True,
    col_map_str = "cmr.tropical",
    bool_plot_x = True,
    bool_plot_y = True,
    bool_plot_z = True
  ):
  ## #################
  ## INITIALISE FIGIRE
  ## #################
  fig = plt.figure(figsize=(7.5, 7.5), constrained_layout=True)
  ax = plt.axes(projection="3d", proj_type="ortho")
  ## dark theme
  if bool_dark_theme:
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
  ## get box domain
  ax_min, ax_max = bbox
  ## create colormap from the Cmasher library
  cmr_colormap = plt.get_cmap(col_map_str, num_cols)
  domain_vals  = np.linspace(ax_min, ax_max, num_cols)
  my_colormap  = cmr_colormap(domain_vals)
  ## initialise the coordinates where contours are evaluated
  coords_contour = np.linspace(ax_min, ax_max, res_contour) 
  ## initialise the slice coordinates (along each direction)
  coords_slice = np.linspace(ax_min, ax_max, res_slices)
  ## initialise the grid for plotting contours
  A1, A2 = np.meshgrid(coords_contour, coords_contour)
  ## rotate the figure
  for ang_val, ang_index in UsefulFuncs.loopListWithUpdates(list_angles):
    ## #################
    ## PLOT EACH CONTOUR
    ## #################
    for coord in coords_slice:
      ## evaluate the function
      X = shape_func(coord, A1, A2)
      Y = shape_func(A1, coord, A2)
      Z = shape_func(A1, A2, coord)
      ## get contour color
      if bool_multicolors:
        color = [ tuple(my_colormap[
          UsefulFuncs.getIndexClosestValue(domain_vals, coord)
        ]) ]
      else: color = "black"
      ## plot contours
      if bool_plot_x: # YZ plane
        try: ax.contour(X+coord, A1, A2, [coord], zdir="x", colors=color, linewidth=2, linestyle="-")
        except UserWarning: continue
      if bool_plot_y: # XZ plane
        try: ax.contour(A1, Y+coord, A2, [coord], zdir="y", colors=color, linewidth=2, linestyle="-")
        except UserWarning: continue
      if bool_plot_z: # XY plane
        try: ax.contour(A1, A2, Z+coord, [coord], zdir="z", colors=color, linewidth=2, linestyle="-")
        except UserWarning: continue
    ## #############
    ## ADJUST FIGURE
    ## #############
    ## set figure window bounds
    ax.set_xlim3d(ax_min, ax_max)
    ax.set_ylim3d(ax_min, ax_max)
    ax.set_zlim3d(ax_min, ax_max)
    ## set axis equal
    ax.set_box_aspect([1, 1, 1])
    ## remove axis labels
    ax.set_axis_off()
    ## change view axis
    ax.view_init(
      elev = 35 * np.cos((ang_val % 360) * np.pi / 180),
      azim = 45 + ang_val
    )
    ## ###########
    ## SAVE FIGURE
    ## ###########
    if len(list_angles) > 3:
      fig_name = shape_name + "_{0:03}".format(int(ang_index)) + ".png"
    else:
      fig_name = shape_name + ".png"
      print("Saved figure:", fig_name)
    fig_filepath = UsefulFuncs.createFilePath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    ## clear axis
    ax.clear()
  ## once plotting had finished - close figure
  plt.close()
  ## animate frames
  if len(list_angles) > 3:
    PlotFuncs.animateFrames(filepath_plot, shape_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## ##################
  ## INITIALISE PROGRAM
  ## ##################
  ## current working directory
  filepath_plot = os.getcwd() + "/RotatingTools.Shapes"
  UsefulFuncs.createFolder(filepath_plot)
  ## program workflow
  bool_plot_frame   = 1
  bool_create_video = 1
  ## colormap: guppy, pride, fusion, viola, waterlily, sunburst_r, jungle, rainforest, tropical
  col_map = "tropical_r"

  ## ##############################
  ## DEFINE SHAPE + PLOT PROPERTIES
  ## ##############################
  ## gyroid
  plot_args = {
      "bbox":(-1.0, 1.0),
      "shape_name":"gyroid_sphere_"+col_map,
      "shape_func":MyTools.Opperations.intersect(
          MyTools.Shapes.gyroid,
          functools.partial(MyTools.Shapes.sphere, c=1.25)
      ),
      "res_slices":30,
      "bool_dark_theme":False,
      "bool_multicolors":True,
      "col_map_str":"cmr."+col_map,
      "bool_plot_x":True,
      "bool_plot_y":True,
      "bool_plot_z":True
  }
  # ## tangle cube
  # plot_args = {
  #     "bbox":(-2.5, 2.5),
  #     "shape_name":("gt_sphere_" + col_map),
  #     "shape_func":Tools.Opperations.subtract(
  #         Tools.Shapes.goursatTangle,
  #         functools.partial(Tools.Shapes.sphere, c=2.25)
  #     ),
  #     "res_slices":30,
  #     "bool_dark_theme":False,
  #     "bool_multicolors":True,
  #     "col_map_str":("cmr." + col_map),
  #     "bool_plot_x":True,
  #     "bool_plot_y":True,
  #     "bool_plot_z":True
  # }

  ## ##############
  ## PLOT THE SHAPE
  ## ##############
  ## plot a single frame
  if bool_plot_frame:
    funcPlotShape(
      filepath_plot = filepath_plot,
      list_angles = [ 0 ],
      **plot_args
    )
  ## pot and animate the shape rotating
  if bool_create_video:
    ## create a folder where plots of the shape will be saved
    filepath_frames = UsefulFuncs.createFilePath([
      filepath_plot,
      plot_args["shape_name"]
    ])
    UsefulFuncs.createFolder(filepath_frames)
    funcPlotShape(
      filepath_plot = filepath_frames,
      list_angles   = np.linspace(0, 360, 100),
      **plot_args
    )


## ###############################################################
## RUN MAIN
## ###############################################################
if __name__ == "__main__":
  main()


## ###############################################################
## END OF PROGRAM
## ###############################################################