import numpy as np

def sphere(x, y, z, r=1):
  return x*x + y*y + z*z - r*r

def sphere_z(x, y, r=1):
  val_to_sqrt = r*r - x*x - y*y
  if val_to_sqrt < 0: return np.inf
  else: z = np.sqrt( val_to_sqrt )
  return z

def torus(x, y, z, r1=2, r2=1):
  return z*z - r2*r2 + ( r1 - np.sqrt(x*x + y*y) )**2

def torus_z(x, y, r1=2, r2=1):
  val_to_sqrt = r2*r2 - ( r1 - np.sqrt(x*x + y*y) )**2
  if val_to_sqrt < 0: return np.inf
  else: z = np.sqrt( val_to_sqrt )
  return z

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

def tunnels(x, y, z):
  return np.cos(x) + np.cos(y) + np.cos(z)

