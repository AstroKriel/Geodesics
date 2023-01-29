## ###############################################################
## MODULES
## ###############################################################
import skimage
import numpy as np
import matplotlib.pyplot as plt


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def gyroid(x, y, z):
    return np.cos(x) * np.sin(y) + np.cos(y) * np.sin(z) + np.cos(z) * np.sin(x)


## ###############################################################
## PROGRAM MAIN
## ###############################################################
def main():
  # Create a figure and an axes
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  # Define the range of x, y, and z
  xmin, xmax, ymin, ymax, zmin, zmax = -np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi
  # Generate a grid of points
  x, y, z = np.meshgrid(
    np.linspace(xmin, xmax, 100),
    np.linspace(ymin, ymax, 100),
    np.linspace(zmin, zmax, 100)
  )
  # Evaluate gyroid at each point in domain
  F = gyroid(x, y, z)
  # Extract the isosurface for the given isovalue
  verts, faces, normals, values = skimage.measure.marching_cubes(F, 0)
  # Extrapolate along normal
  offset = 3.0
  # Plot the isosurface
  ax.plot(verts[:,0], verts[:,1], verts[:,2], "b.", ms=0.1)
  # Set the plot labels
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  # Save the plot
  ax.view_init(30, 26)
  plt.show()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM