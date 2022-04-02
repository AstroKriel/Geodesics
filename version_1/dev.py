#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import re
import warnings
import functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr # https://cmasher.readthedocs.io/user/introduction.html#introduction
## cmr sequential maps: tropical, ocean, arctic, bubblegum, lavender
## cmr diverging maps: iceburn, wildfire, fusion

from tqdm.auto import tqdm

## ###############################################################
## PREPARE TERMINAL/WORKSPACE/CODE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend('agg') # use a non-interactive plotting backend
warnings.simplefilter('ignore', UserWarning) # hide warnings


## ###############################################################
## WORKING WITH FOLDERS
## ###############################################################
def createFolder(folder_name):
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print("SUCCESS: Folder created. \n\t" + folder_name + "\n")
    else: print("WARNING: Folder already exists (folder not created). \n\t" + folder_name + "\n")

def createFilePath(folder_names):
    return re.sub( '/+', '/', "/".join([ folder for folder in folder_names if folder != "" ]) )


## ###############################################################
## WORKING WITH LISTS / ARRAYS
## ###############################################################
def loopListWithUpdates(input_list, bool_hide_updates=False):
    lst_len = len(input_list)
    return zip(
        input_list,
        tqdm(
            range(lst_len),
            total   = lst_len - 1,
            disable = (lst_len < 3) or bool_hide_updates
        )
    )

def getIndexClosestValue(input_vals, target_value):
    return np.argmin(np.abs(np.array(input_vals) - target_value))


## ###############################################################
## ANIMATING
## ###############################################################
def animateFrames(filepath_frames, shape_name):
    print("Animating plots...")
    ## create filepath to where plots are saved
    filepath_input = createFilePath([filepath_frames, shape_name + "_%*.png"])
    ## create filepath to where animation should be saved
    filepath_output = createFilePath([filepath_frames, shape_name + ".mp4"])
    ## animate plot frames
    os.system(
        "ffmpeg -y -start_number 0 -i " + filepath_input +
        " -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 " + filepath_output
    )
    print("Animation finished: " + filepath_output)
    print(" ")


## ###############################################################
## PLOTTING
## ###############################################################
def plotShape(
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
    ## ##########
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
    ## create from Cmasher's library
    cmr_colormap = plt.get_cmap(col_map_str, num_cols)
    domain_vals  = np.linspace(ax_min, ax_max, num_cols)
    my_colormap  = cmr_colormap(domain_vals)
    ## initialise the coordinates where contours are evaluated
    coords_contour = np.linspace(ax_min, ax_max, res_contour) 
    ## initialise the slice coordinates (along each direction)
    coords_slice = np.linspace(ax_min, ax_max, res_slices)
    ## initialise the grid for plotting contours
    A1, A2 = np.meshgrid(coords_contour, coords_contour)
    ## initialise plot arguments
    plot_args = {
        "alpha":0.75,
        "linewidths":2,
        "linestyles":"-"
    }
    ## rotate the figure
    for ang_val, ang_index in loopListWithUpdates(list_angles):
        ## #################
        ## PLOT EACH CONTOUR
        ## #####
        for coord in coords_slice:
            ## evaluate the function
            X = shape_func(coord, A1, A2)
            Y = shape_func(A1, coord, A2)
            Z = shape_func(A1, A2, coord)
            ## get contour color
            if bool_multicolors:
                color = [ tuple(my_colormap[
                    getIndexClosestValue(domain_vals, coord)
                ]) ]
            else: color = "black"
            ## plot contours
            if bool_plot_x: # YZ plane
                try: ax.contour(X+coord, A1, A2, [coord], zdir="x", colors=color, **plot_args)
                except UserWarning: continue
            if bool_plot_y: # XZ plane
                try: ax.contour(A1, Y+coord, A2, [coord], zdir="y", colors=color, **plot_args)
                except UserWarning: continue
            if bool_plot_z: # XY plane
                try: ax.contour(A1, A2, Z+coord, [coord], zdir="z", colors=color, **plot_args)
                except UserWarning: continue
        ## #################
        ## ADJUST FIGURE
        ## ######
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
        ## #################
        ## SAVE FIGURE
        ## #####
        if len(list_angles) > 3:
            fig_name = shape_name + "_{0:03}".format(int(ang_index)) + ".png"
        else:
            fig_name = shape_name + ".pdf"
            print("Saved figure:", fig_name)
        fig_filepath = createFilePath([filepath_plot, fig_name])
        plt.savefig(fig_filepath)
        ## clear axis
        ax.clear()
    ## once plotting had finished - close figure
    plt.close()
    ## animate frames
    if len(list_angles) > 3:
        animateFrames(filepath_plot, shape_name)


## ###############################################################
## SHAPES
## ###############################################################
def gyroid(x, y, z):
    a0, a1 = 5.5, 2.0
    return (
        np.sin(a0*x + a1) * np.cos(a0*y + a1) +
        np.sin(a0*y + a1) * np.cos(a0*z + a1) +
        np.sin(a0*z + a1) * np.cos(a0*x + a1)
    )

def goursatTangle(x, y, z):
    a0, a1, a2 = 0.0, -5.0, 11.8
    return (
        x**4 + y**4 + z**4 +
        a0*(x**2 + y**2 + z**2)**2 +
        a1*(x**2 + y**2 + z**2) +
        a2
    )

def sphere(x, y, z, c):
    return x**2 + y**2 + z**2 - c**2

def shape_1(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)

def shape_2(x, y, z):
    return (
        np.sin(x)*np.sin(y)*np.sin(z) + 
        np.sin(x)*np.cos(y)*np.cos(z) + 
        np.cos(x)*np.sin(y)*np.cos(z) + 
        np.cos(x)*np.cos(y)*np.sin(z)
    )

def shape_3(x, y, z):
    return np.cos(x)*np.cos(y) + np.cos(y)*np.sin(z) + np.cos(z)*np.cos(x)

def shape_4(x, y, z):
    return np.cos(x)*np.sin(y) + np.cos(y)*np.sin(z) + np.cos(z)*np.sin(x)

## ###############################################################
## OPPERATORS
## ###############################################################
def translate(fn,x,y,z):
    return lambda a,b,c: fn(x-a, y-b, z-c)

def union(*fns):
    return lambda x,y,z: np.min(
        [ fn(x,y,z) for fn in fns ], 0)

def intersect(*fns):
    return lambda x,y,z: np.max(
        [ fn(x,y,z) for fn in fns ], 0)

def subtract(fn1, fn2):
    return intersect(fn1, lambda *args:-fn2(*args))


## ###############################################################
## RUNNING CODE
## ###############################################################
## get filepath to the current working directory
filepath_directory = os.getcwd()
## choose what to do in program
bool_plot_frame   = 1
bool_create_video = 0

## choose colormap (i.e. guppy, pride, fusion, viola, waterlily, sunburst_r, jungle, rainforest, tropical)
col_map = "tropical_r"

## another shape
plot_args = {
    "bbox":(-10.0, 10.0),
    "shape_name":"shape_1",
    "shape_func":shape_1,
    "res_slices":30,
    "bool_dark_theme":False,
    "bool_multicolors":True,
    "col_map_str":"cmr."+col_map,
    "bool_plot_x":1,
    "bool_plot_y":1,
    "bool_plot_z":1
}

# ## gyroid
# plot_args = {
#     "bbox":(-1.0, 1.0),
#     "shape_name":"gyroid_sphere_"+col_map,
#     "shape_func":intersect(
#         gyroid,
#         functools.partial(sphere, c=1.25)
#     ),
#     "res_slices":30,
#     "bool_dark_theme":False,
#     "bool_multicolors":True,
#     "col_map_str":"cmr."+col_map,
#     "bool_plot_x":True,
#     "bool_plot_y":True,
#     "bool_plot_z":True
# }

# ## tangle cube
# plot_args = {
#     "bbox":(-2.5, 2.5),
#     "shape_name":"gt_sphere_"+col_map,
#     "shape_func":subtract(
#         goursatTangle,
#         functools.partial(sphere, c=2.25)
#     ),
#     "res_slices":30,
#     "bool_dark_theme":False,
#     "bool_multicolors":True,
#     "col_map_str":"cmr."+col_map,
#     "bool_plot_x":True,
#     "bool_plot_y":True,
#     "bool_plot_z":True
# }


## ###############################################################
## PLOT SINGLE SLICE
## ###############################################################
if bool_plot_frame:
    plotShape(
        filepath_plot = filepath_directory,
        list_angles = [ 0 ],
        **plot_args
    )


## ###############################################################
## PLOT/ANIMATE ROTATING SHAPE
## ###############################################################
## initialise set of viewing angles
list_angles = np.linspace(0, 360, 100)
## plot and animate rotating shape
if bool_create_video:
    ## create a folder where plots of the shape will be saved
    filepath_plot = createFilePath([
        filepath_directory,
        plot_args["shape_name"]
    ])
    createFolder(filepath_plot)
    ## plot/animate shape
    plotShape(
        filepath_plot = filepath_plot,
        list_angles = list_angles,
        **plot_args
    )


## ###############################################################
## END OF PROGRAM
## ###############################################################