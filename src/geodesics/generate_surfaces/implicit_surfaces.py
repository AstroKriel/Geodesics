## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def sphere(radius=1.0):
  return lambda x, y, z: x**2 + y**2 + z**2 - radius**2

def box(size=1.0):
  return lambda x, y, z: numpy.max(numpy.stack([
    numpy.abs(x) - size,
    numpy.abs(y) - size,
    numpy.abs(z) - size
  ]), axis=0)

def gyroid(size=0.5, shift=0.0):
  return lambda x, y, z: (
    numpy.sin(size * x + shift) * numpy.cos(size * y + shift) +
    numpy.sin(size * y + shift) * numpy.cos(size * z + shift) +
    numpy.sin(size * z + shift) * numpy.cos(size * x + shift)
  )

def goursat_tangle(a0=0.0, a1=-5.0, a2=11.8):
  return lambda x, y, z: (
    1.0 * (x**4 + y**4 + z**4) +
    a0  * (x**2 + y**2 + z**2)**2 +
    a1  * (x**2 + y**2 + z**2) +
    a2
  )

def tunnels():
  return lambda x, y, z: numpy.cos(x) + numpy.cos(y) + numpy.cos(z)


## END OF MODULE