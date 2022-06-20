## from: https://gist.github.com/glederrey
import time
import numpy as np
import numdifftools as nd


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def gradient(x, func, eps=1e-6):
  """
  Central finite differences for computing gradient
  INPUT:
    x: vector of parameters with shape (n, 1)
    func: function to compute the gradient on. Take x as a parameter.
    eps: parameter for the central differences
  OUTPUT:
    grad: vector with the values of the gradient with shape (n, 1)
  """
  # Size the problem, i.e. nbr of parameters
  n = len(x)
  # Prepare the vector for the gradient
  grad = np.zeros(n)
  # Prepare the array to add epsilon to.
  dx = np.zeros(n)
  # Go through all parameters
  for i in range(len(x)):
    # Add epsilon to variate a parameter
    dx[i] += eps
    # Central finite differences
    grad[i] = -(func(x+dx) - func(x-dx))/(2*eps)
    # Set back to 0
    dx[i] = 0
  return grad


def hessian(x, func, eps=1e-6):
  """
  Central finite differences for computing the hessian
  INPUT:
    x: vector of parameters with shape (n, 1)
    func: function to compute the gradient on. Take x as a parameter.
    eps: parameter for the central differences
  OUTPUT:
    grad: vector with the values of the gradient with shape (n, 1)
  """
  # Size the problem, i.e. nbr of parameters
  n = len(x)
  # Prepare the vector for the gradient
  hess = np.zeros((n,n))
  # Prepare the array to add epsilon to.
  dx = np.zeros(n)
  # Go through all parameters
  for i in range(n):
    # Add epsilon to variate a parameter
    dx[i] += eps
    # Compute the gradient with forward and backward difference
    grad_plus = gradient(x+dx, func, eps)
    grad_minus = gradient(x-dx, func, eps)
    # Central finite difference
    hess[i,:] = -(grad_plus - grad_minus)/(2*eps)
    # Set back to 0
    dx[i] = 0
  return hess


def conjgrad(A, b, x):
  """
  A function to solve [A]{x} = {b} linear equation system with the 
  conjugate gradient method.
  More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
  ========== Parameters ==========
  A : matrix 
    A real symmetric positive definite matrix.
  b : vector
    The right hand side (RHS) vector of the system.
  x : vector
    The starting guess for the solution.
  """  
  r = b - np.dot(A, x)
  p = r
  rsold = np.dot(np.transpose(r), r)
  for i in range(len(b)):
    Ap = np.dot(A, p)
    alpha = rsold / np.dot(np.transpose(p), Ap)
    x = x + np.dot(alpha, p)
    r = r - np.dot(alpha, Ap)
    rsnew = np.dot(np.transpose(r), r)
    if np.sqrt(rsnew) < 1e-8:
      break
    p = r + (rsnew/rsold)*p
    rsold = rsnew
  return x


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  # Create a function to test our implementation
  func = lambda x: 3*x[0]**2 + 2*x[1]**3 + x[0]*x[1] + x[2]
  # Parameter
  x = [1,1,1]
  print("Func({}): {}".format(
    ", ".join(map(str, x)),
    func(x)
  ))
  # Compute the gradient
  start = time.time()
  grad = gradient(x, func, eps=1e-4)
  stop = time.time()
  print("Gradient: ({}). Computed in: {:.2f}µs.".format(
    ", ".join(map(str, grad)),
    1e6*(stop-start)
  ))
  # Compute the hessian
  start = time.time()
  hess = hessian(x, func, eps=1e-4)
  print(hess)
  stop = time.time()
  print("Hessian: ({}). Computed in: {:.2f}µs.".format(
    ", ".join(map(str, hess)),
    1e6*(stop-start)
  ))
  # Compare our implementation with numdifftools
  start = time.time()
  nd_hess = nd.Hessian(func)
  nd_hess_vals = nd_hess(x)
  stop = time.time()
  print("Hessian (numdifftools): ({}). Computed in: {:.2f}µs.".format(
    ", ".join(map(str, nd_hess_vals)),
    1e6*(stop-start)
  ))
  print("Norm of the difference between the two hessians: {:.2E}".format(
    np.linalg.norm(hess - nd_hess_vals)
  ))


## ###############################################################
## RUN MAIN
## ###############################################################
if __name__ == "__main__":
  main()

## END OF DEMO