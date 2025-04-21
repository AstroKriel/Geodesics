## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
from pathlib import Path


## ###############################################################
## FUNCTIONS
## ###############################################################
def animate_images(directory, shape_name):
  print("Animating plots...")
  directory = Path(directory)
  file_path_input  = directory / f"{shape_name}_%*.png"
  file_path_output = (directory.parent / f"{shape_name}.mp4").resolve()
  command = (
    f"ffmpeg -y -start_number 0 -i '{file_path_input}' "
    f"-vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 '{file_path_output}'"
  )
  os.system(command)
  print("Animation finished:", file_path_output)
  print(" ")


## END OF MODULE