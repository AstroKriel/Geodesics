import numpy as np

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

