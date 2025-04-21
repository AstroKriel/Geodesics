## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def sphere(x_pos, y_pos, z_pos, radius=1):
  return x_pos*x_pos + y_pos*y_pos + z_pos*z_pos - radius*radius

def gyroid(x_pos, y_pos, z_pos, a0=5.5, a1=2.0):
  return (
    numpy.sin(a0*x_pos + a1) * numpy.cos(a0*y_pos + a1) +
    numpy.sin(a0*y_pos + a1) * numpy.cos(a0*z_pos + a1) +
    numpy.sin(a0*z_pos + a1) * numpy.cos(a0*x_pos + a1)
  )

def goursat_tangle(x_pos, y_pos, z_pos, a0=0.0, a1=-5.0, a2=11.8):
  return (
    x_pos**4 + y_pos**4 + z_pos**4 +
    a0*(x_pos**2 + y_pos**2 + z_pos**2)**2 +
    a1*(x_pos**2 + y_pos**2 + z_pos**2) +
    a2
  )

def tunnels(x_pos, y_pos, z_pos):
  return numpy.cos(x_pos) + numpy.cos(y_pos) + numpy.cos(z_pos)


## END OF MODULE