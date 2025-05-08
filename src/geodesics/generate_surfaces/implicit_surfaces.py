## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy


## ###############################################################
## IMPLICIT SURFACES
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

def mandelbulb(exponent=8, num_iterations=10):
  def distance_estimate(x, y, z):
    ## initialise arrays for vectorised compute
    input_position    = numpy.stack([x, y, z], axis=-1) # shape (N,N,N,3)
    iter_position     = input_position.copy()
    radius_derivative = numpy.ones_like(x)
    radius            = numpy.zeros_like(x)
    active_mask       = numpy.ones_like(x, dtype=bool)
    for _ in range(num_iterations):
      ## only process points that have not escaped yet
      if not numpy.any(active_mask): break
      ## calculate the current radius for active points
      radius[active_mask] = numpy.linalg.norm(
        iter_position[active_mask], axis=-1
      )
      ## update the escape condition mask
      active_mask &= (radius <= 2.0)
      ## calculate spherical coordinates for active points
      polar_angle = numpy.arccos(
        iter_position[active_mask, 2] / radius[active_mask]
      )
      azimuthal_angle = numpy.arctan2(
        iter_position[active_mask, 1], iter_position[active_mask, 0]
      )
      ## update derivative and position for active points
      radius_derivative[active_mask] = (
        exponent * numpy.power(radius[active_mask], exponent-1) 
        * radius_derivative[active_mask] + 1.0
      )
      r_pow = numpy.power(radius[active_mask], exponent)
      iter_position[active_mask] = input_position[active_mask] + r_pow[..., None] * numpy.stack([
        numpy.sin(polar_angle * exponent) * numpy.cos(azimuthal_angle * exponent),
        numpy.sin(polar_angle * exponent) * numpy.sin(azimuthal_angle * exponent),
        numpy.cos(polar_angle * exponent)
      ], axis=-1)
    return 0.5 * numpy.log(radius) * radius / radius_derivative
  return distance_estimate



## END OF MODULE