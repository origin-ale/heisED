import numpy as np
from numpy.random import default_rng

from scipy.linalg import norm
from scipy.linalg import eigh_tridiagonal

import mytiming as mt

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

def spinH_lanczos(H: np.ndarray, max_size = 100000, reortho = False):
  """
  Tridiagonalize a spin hamiltonian using the Lanczos algorithm (Sandvik 2010 §4.2.3)

  Parameters
  ---
  H: np.ndarray
    The spin hamiltonian to tridiagonalize
  max_size: int
    The maximum size of the Lanczos basis
  reortho: bool
    Whether to explicitly reorthogonalize each Lanczos vector to prevent degeneracy. Significantly increases computation time!
  """
  st = mt.perf_counter() # Function start time

  N = len(H) # Get system size from hamiltonian
  phi = [] # List of normalized Lanczos vectors
  norm = [] # List of Lanczos vector norms
  a = [] # List of Lanczos a coefficients
  seed = 42
  compact_H = save_nonzero(H)
  mt.timeprint(st, "Compact hamiltonian saved")

  """Initialize algorithm"""
  new_phi, new_norm = normalize(default_rng(seed).random(N)) # phi0 is a random normalized vector
  phi.append(new_phi)
  norm.append(new_norm)
  latest_Hphi = spinH_action(phi[0], compact_H[0], compact_H[1], compact_H[2])
  a.append(phi[0] @ latest_Hphi)
  mt.timeprint(st, "Lanczos initialized, basis size 1")

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

    """Ensure orthogonality""" # Comment out for slight performance improvements over reortho = False
    if reortho:
      for i in range(0,m):
        q = phi[i] @ new_phi
        new_phi = (new_phi - q*phi[i])/(1-q*q)

    new_phi, new_norm = normalize(unnorm_new_phi)
    phi.append(new_phi)
    norm.append(new_norm)
    latest_Hphi = spinH_action(phi[m], compact_H[0], compact_H[1], compact_H[2]) 
    a.append(phi[m] @ latest_Hphi)
    if len(phi) % 500 == 0: mt.timeprint(st, f"Lanczos basis size {len(phi)}")

  mt.timeprint(st, "Lanczos basis built")
  H_tridiag = np.zeros((tridiag_size, tridiag_size))
  H_tridiag[0,0] = a[0]
  H_tridiag[1, 0] = norm[1]
  for m in range(1, tridiag_size-1):
    H_tridiag[m-1, m] = norm[m]
    H_tridiag[m,m] = a[m]
    H_tridiag[m+1, m] = norm[m+1]
  H_tridiag[tridiag_size-2, tridiag_size-1] = norm[tridiag_size-1]
  H_tridiag[tridiag_size-1,tridiag_size-1] = a[tridiag_size-1]
  mt.timeprint(st, "Tridiagonal matrix built")
  return H_tridiag

def spinH_lanczos_gs(H: np.ndarray, tol = 1e-7, reortho = False):
  """
  Get the ground state of a spin hamiltonian using the Lanczos algorithm, only using as many basis vectors as necessary (Sandvik 2010 §4.2.3)

  Parameters
  ---
  H: np.ndarray
    The spin hamiltonian to tridiagonalize
  tol: float
    The tolerance for the GS to be considered converged
  reortho: bool
    Whether to explicitly reorthogonalize each Lanczos vector to prevent degeneracy. Significantly increases computation time!
  """
  st = mt.perf_counter() # Function start time

  N = len(H) # Get system size from hamiltonian
  phi = [] # List of normalized Lanczos vectors
  norm = [] # List of Lanczos vector norms
  a = [] # List of Lanczos a coefficients
  seed = 42
  compact_H = save_nonzero(H)
  delta_gs = np.inf
  prev_gs = 0.
  
  lH_diag = []
  lH_odiag = []

  mt.timeprint(st, "Compact hamiltonian saved")

  """Initialize algorithm"""
  m = 0
  new_phi, new_norm = normalize(default_rng(seed).random(N)) # phi0 is a random normalized vector
  phi.append(new_phi)
  norm.append(new_norm)
  latest_Hphi = spinH_action(phi[0], compact_H[0], compact_H[1], compact_H[2])
  a.append(phi[0] @ latest_Hphi)
  lH_diag.append(a[0])
  mt.timeprint(st, "Lanczos initialized, basis size 1")

  """Special case new = 1"""
  m = 1
  unnorm_new_phi = latest_Hphi - a[0]*phi[0]
  new_phi, new_norm = normalize(unnorm_new_phi)
  phi.append(new_phi)
  norm.append(new_norm)
  latest_Hphi = spinH_action(phi[1], compact_H[0], compact_H[1], compact_H[2]) 
  # this is technically a waste but it lets the loop start with 2 elements in everything 
  a.append(phi[1] @ latest_Hphi)

  lH_diag.append(a[1])
  lH_odiag.append(norm[1])

  """Main loop 2 <= new <= size-1"""
  m = 2
  while delta_gs > tol:
    latest_Hphi = spinH_action(phi[m-1], compact_H[0], compact_H[1], compact_H[2])
    unnorm_new_phi = latest_Hphi - a[m-1]*phi[m-1] - norm[m-1]*phi[m-2]
    new_phi, new_norm = normalize(unnorm_new_phi)

    """Ensure orthogonality""" # Comment out for slight performance improvements over reortho = False
    if reortho:
      for i in range(0,m):
        q = phi[i] @ new_phi
        new_phi = (new_phi - q*phi[i])/(1-q*q)

    new_phi, new_norm = normalize(unnorm_new_phi)
    phi.append(new_phi)
    norm.append(new_norm)
    latest_Hphi = spinH_action(phi[m], compact_H[0], compact_H[1], compact_H[2]) 
    a.append(phi[m] @ latest_Hphi)
   
    lH_diag.append(a[m])
    lH_odiag.append(norm[m])

    if len(phi) % 50 == 0: mt.timeprint(st, f"Lanczos basis size {len(phi)}")

    gs_energy = eigh_tridiagonal(lH_diag, 
                              lH_odiag, 
                              eigvals_only=True, 
                              select='i', 
                              select_range=[0,0])
    gs = gs_energy[0]

    delta_gs = abs(prev_gs - gs)
    prev_gs = gs
    m += 1
  mt.timeprint(st, f"Lanczos basis built with {m} elements")
  return gs


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
