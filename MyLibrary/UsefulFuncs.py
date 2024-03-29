## ###############################################################
## MODULES
## ###############################################################
import os, re, tqdm
import numpy as np


## ###############################################################
## DEFINING SHAPES
## ###############################################################
class Shapes():
  def sphere(x, y, z, r=1):
    return x*x + y*y + z*z - r*r

  def gyroid(x, y, z, a0=5.5, a1=2.0):
    return (
      np.sin(a0*x + a1) * np.cos(a0*y + a1) +
      np.sin(a0*y + a1) * np.cos(a0*z + a1) +
      np.sin(a0*z + a1) * np.cos(a0*x + a1)
    )

  def goursatTangle(x, y, z, a0=0.0, a1=-5.0, a2=11.8):
    return (
      x**4 + y**4 + z**4 +
      a0*(x**2 + y**2 + z**2)**2 +
      a1*(x**2 + y**2 + z**2) +
      a2
    )

  def sphere(x, y, z, c):
    return x**2 + y**2 + z**2 - c**2

  def tunnels(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)


## ###############################################################
## DEFINING SHAPE OPPERATORS
## ###############################################################
class Opperations():
  def translate(fn, x, y, z):
    return lambda a,b,c: fn(x-a, y-b, z-c)

  def union(*fns):
    return lambda x, y, z: np.min(
      [ fn(x,y,z) for fn in fns ], 0)

  def intersect(*fns):
    return lambda x, y, z: np.max(
      [ fn(x,y,z) for fn in fns ], 0)

  def subtract(fn1, fn2):
    return Opperations.intersect(fn1, lambda *args: -fn2(*args))


## ###############################################################
## WORKING WITH FOLDERS
## ###############################################################
def createFolder(folder_name):
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print("SUCCESS: Folder created. \n\t" + folder_name + "\n")
    else: print("WARNING: Folder already exists (folder not created). \n\t" + folder_name + "\n")

def createFilePath(folder_names):
    return re.sub( '/+', '/', "/".join([
      folder for folder in folder_names if folder != ""
    ]) )


## ###############################################################
## WORKING WITH LISTS / ARRAYS
## ###############################################################
def loopListWithUpdates(input_list, bool_hide_updates=False):
    lst_len = len(input_list)
    return zip(
        input_list,
        tqdm.auto.tqdm(
            range(lst_len),
            total   = lst_len - 1,
            disable = (lst_len < 3) or bool_hide_updates
        )
    )

def getIndexClosestValue(input_vals, target_value):
    return np.argmin(np.abs( np.array(input_vals) - target_value ))


## END OF LIBRARY