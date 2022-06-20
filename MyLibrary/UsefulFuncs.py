## ###############################################################
## MODULES
## ###############################################################
import os, re
import numpy as np

from tqdm.auto import tqdm


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
        tqdm(
            range(lst_len),
            total   = lst_len - 1,
            disable = (lst_len < 3) or bool_hide_updates
        )
    )

def getIndexClosestValue(input_vals, target_value):
    return np.argmin(np.abs( np.array(input_vals) - target_value ))


## END OF LIBRARY