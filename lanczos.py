import numpy as np
from numpy.random import default_rng

from scipy.linalg import norm
from scipy.linalg import eigh_tridiagonal

from build_hamiltonian import build_hamiltonian

def normalize(v: np.ndarray):
  """
  Returns v normalized and the norm of v
  """
  return v/norm(v), norm(v)

def save_nonzero(H: np.ndarray):
  """
  Input: one matrix H

  Returns
  ---
  nonzero_elements : list of int
    The number of nonzero elements in each row of H
  nze_locations : list of int
    The locations of nonzero elements in each row. First `nonzero_elements[0]` entries refer to row `0`, and so on
  nze_values : list of float
    The nonzero elements themselves
  """
  N = len(H)
  nonzero_elements = [] # Number of nonzero elements in each row
  nze_locations = [] # Locations of nonzero elements in each row
  nze_values = [] # Values of each nonzero element
  for a in range(0, N):
    nze_a = 0
    for b in range(0, N):
      if H[a,b] != 0.:
        nze_a += 1
        nze_locations.append(b)
        nze_values.append(float(H[a,b])) # Convert from np.float64 for nicer printing in debug
    nonzero_elements.append(nze_a)
  return nonzero_elements, nze_locations, nze_values

def spinH_action(psi: np.ndarray, nonzero_elements: list, nze_locations: list, nze_values: list):
  """
  Act with a spin hamiltonian given by compact information on a vector (Sandvik 2010 §4.2.3)

  Parameters
  ---
  psi : np.ndarray
    The vector to act on
  nonzero_elements : list of int
    The number of nonzero elements in each row of the spin hamiltonian
  nze_locations : list of int
    The locations of nonzero elements in each row. First `nonzero_elements[0]` entries refer to row `0`, and so on
  nze_values : list of float
    The nonzero elements themselves
  """
  N = len(nonzero_elements) # Number of rows in spin hamiltonian
  Hpsi = np.zeros(N)
  i = 0 # Tracker for position in nze_locations and nze_values
  for a in range(0, N):
    # print(f'----------\na = {a}')
    for j in range(0, nonzero_elements[a]):
      # print(f'i = {i}')
      # print(f'nze_locations[i] = {nze_locations[i]}\nnze_values[i] = {nze_values[i]}')
      Hpsi[nze_locations[i]] += nze_values[i]*psi[a]
      Hpsi[a] += nze_values[i] * psi[nze_locations[i]]
      i += 1
  return Hpsi/2. # Correct for double counting

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
  compact_H = save_nonzero(H)

  """Initialize algorithm"""
  new_phi, new_norm = normalize(default_rng(seed).random(N)) # phi0 is a random normalized vector
  phi.append(new_phi)
  norm.append(new_norm)
  latest_Hphi = spinH_action(phi[0], compact_H[0], compact_H[1], compact_H[2])
  a.append(phi[0] @ latest_Hphi)

  """Special case new = 1"""
  unnorm_new_phi = latest_Hphi - a[0]*phi[0]
  new_phi, new_norm = normalize(unnorm_new_phi)
  phi.append(new_phi)
  norm.append(new_norm)
  latest_Hphi = spinH_action(phi[1], compact_H[0], compact_H[1], compact_H[2]) 
  # this is technically a waste but it lets the loop start with 2 elements in everything 
  a.append(phi[1] @ latest_Hphi)

  """Main loop 2 <= new <= size-1"""
  tridiag_size = min(max_size, N)
  for m in range(2, tridiag_size+1):
    latest_Hphi = spinH_action(phi[m-1], compact_H[0], compact_H[1], compact_H[2])
    unnorm_new_phi = latest_Hphi - a[m-1]*phi[m-1] - norm[m-1]*phi[m-2]
    new_phi, new_norm = normalize(unnorm_new_phi)
    phi.append(new_phi)
    norm.append(new_norm)
    latest_Hphi = spinH_action(phi[m], compact_H[0], compact_H[1], compact_H[2]) 
    a.append(phi[m] @ latest_Hphi)

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

def test_spinH_action(H, compact_H):
  print(f'Hamiltonian is \n{H}')
  N = len(H)
  successes = 0
  for test in range(0, 10):
    psi = default_rng().random(2**N)
    Hpsi_efficient = spinH_action(psi, compact_H[0], compact_H[1], compact_H[2])
    Hpsi_naive = H @ psi
    print(f'''
          Efficient: {Hpsi_efficient}
          Naive:     {Hpsi_naive}''')
  return
