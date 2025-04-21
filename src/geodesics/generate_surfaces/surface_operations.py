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
  return lambda x_pos, y_pos, z_pos: numpy.min([ func(x_pos,y_pos,z_pos) for func in funcs ], axis=0)

def intersect(*funcs):
  return lambda x_pos, y_pos, z_pos: numpy.max([ func(x_pos,y_pos,z_pos) for func in funcs ], axis=0)

def subtract(func1, func2):
  return intersect(func1, lambda *args: -func2(*args))

def rotate_xy(func, angle_degrees):
  angle_radians = numpy.pi * angle_degrees / 180
  cos_theta = numpy.cos(angle_radians)
  sin_theta = numpy.sin(angle_radians)
  return lambda x_pos, y_pos, z_pos: func(
    cos_theta * x_pos - sin_theta * y_pos,
    sin_theta * x_pos + cos_theta * y_pos,
    z_pos
  )

def rotate_xz(func, angle_degrees):
  angle_radians = numpy.pi * angle_degrees / 180
  cos_theta = numpy.cos(angle_radians)
  sin_theta = numpy.sin(angle_radians)
  return lambda x_pos, y_pos, z_pos: func(
    cos_theta * x_pos - sin_theta * z_pos,
    y_pos,
    sin_theta * x_pos + cos_theta * z_pos
  )

def rotate_yz(func, angle_degrees):
  angle_radians = numpy.pi * angle_degrees / 180
  cos_theta = numpy.cos(angle_radians)
  sin_theta = numpy.sin(angle_radians)
  return lambda x_pos, y_pos, z_pos: func(
    x_pos,
    cos_theta * y_pos - sin_theta * z_pos,
    sin_theta * y_pos + cos_theta * z_pos
  )


## END OF MODULE