## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def translate(func, x_pos, y_pos, z_pos):
  return lambda x_shift, y_shift, z_shift: func(x_pos-x_shift, y_pos-y_shift, z_pos-z_shift)

def union(*funcs):
  return lambda x_pos, y_pos, z_pos: numpy.min([ func(x_pos,y_pos,z_pos) for func in funcs ], 0)

def intersect(*funcs):
  return lambda x_pos, y_pos, z_pos: numpy.max([ func(x_pos,y_pos,z_pos) for func in funcs ], 0)

def subtract(func1, func2):
  return intersect(func1, lambda *args: -func2(*args))


## END OF MODULE