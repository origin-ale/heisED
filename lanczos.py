import numpy as np
from numpy.random import default_rng

from scipy.linalg import norm

from build_hamiltonian import build_hamiltonian

def normalize(v: np.ndarray):
  """
  Returns v normalized and the norm of v
  """
  return v/norm(v), norm(v)

def spinH_action(H: np.ndarray, psi: np.ndarray):
  """
  Compute action of H on psi, optimized for spin hamiltonians which are sparse (Sandvik 2010 §4.2.3)
  """
  return H @ psi # placeholder basic matrix multiplication

def spinH_lanczos(H: np.ndarray, max_size = 100000):
  """
  Tridiagonalize a spin hamiltonian using the Lanczos algorithm (Sandvik 2010 §4.2.3)

  Parameters
  ---
  H: np.ndarray
    The spin hamiltonian to tridiagonalize
  max_size: int
    The maximum size of the Lanczos basis
  """
  N = len(H) # Get system size from hamiltonian
  phi = [] # List of normalized Lanczos vectors
  norm = [] # List of Lanczos vector norms
  a = [] # List of Lanczos a coefficients
  seed = 42

  new_phi, new_norm = normalize(default_rng(seed).random(N)) # phi0 is a random normalized vector
  phi.append(new_phi)
  norm.append(new_norm)
  
  latest_Hphi = spinH_action(H, phi[0])
  a.append(phi[0] @ latest_Hphi)
  unnorm_new_phi = latest_Hphi - a[0]*phi[0]
  new_phi, new_norm = normalize(unnorm_new_phi)
  phi.append(new_phi)
  norm.append(new_norm)

  tridiag_size = min(max_size, N)
  for m in range(1, tridiag_size):
    latest_Hphi = spinH_action(H, phi[m])
    a.append(phi[m] @ latest_Hphi)
    unnorm_new_phi = latest_Hphi - a[m]*phi[m] - norm[m]*phi[m-1]
    new_phi, new_norm = normalize(unnorm_new_phi)
    phi.append(new_phi)
    norm.append(new_norm)
  phi.pop() # remove extra elements given by last iteration
  norm.pop()

  H_tridiag = np.zeros((tridiag_size, tridiag_size))
  H_tridiag[0,0] = a[0]
  H_tridiag[1, 0] = norm[1]
  for m in range(1, tridiag_size-1):
    H_tridiag[m-1, m] = norm[m]
    H_tridiag[m,m] = a[m]
    H_tridiag[m+1, m] = norm[m+1]
  H_tridiag[tridiag_size-2, tridiag_size-1] = norm[tridiag_size-1]
  H_tridiag[tridiag_size-1,tridiag_size-1] = a[tridiag_size-1]
  return H_tridiag

np.set_printoptions(precision = 3, suppress = True)

H = build_hamiltonian(2, 1.)
print(f'Hamiltonian is \n{H}')

tridiag = spinH_lanczos(H)
print(f'Tridiagonal Hamiltonian is \n{tridiag}')