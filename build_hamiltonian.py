import numpy as np
import mytiming as mt

def get_bit(n: int, i: int):
  """
  Get the ith bit of an int n
  """
  return n >> i & 1

def flip_bits(n: int, i: int, j: int):
  """
  Flip (not switch) the ith and jth bits of an int n
  """
  return n ^ (2**i + 2**j)

def build_hamiltonian(N: int, J: float):
  """
  Build the Hamiltonian matrix for a Heisenberg spin chain (Sandvik 2010, §4.1.2)

  Parameters
  ----------
  N : int
    The length of the spin chain (number of spins).
  J : float
    The coupling constant for spin-spin interactions.
  """
  st = mt.perf_counter()
  state_n = 2**N
  H = np.zeros((state_n, state_n))
  for s in range(0, state_n):
    for i in range(0, N):
      j = (i+1)%N # rightward nearest neighbor of i, with periodicity enforced
      if get_bit(s,i) == get_bit(s,j): 
        H[s,s] += .25
      else:
        H[s,s] += -.25
        sp = flip_bits(s,i,j)
        H[s,sp] = .5
    if s%5000 == 0: mt.timeprint(st, f"H building {(s/state_n * 100):.2f}% done")
  mt.timeprint(st, f"H building done!")
  return J*H

def build_compressed_hamiltonian(N: int, J: float):
  """
  Build the Hamiltonian matrix for a Heisenberg spin chain (Sandvik 2010, §4.1.2) and return it in a compressed form suitable for fast action on vectors

  Parameters
  ---
  N : int
    The length of the spin chain (number of spins).
  J : float
    The coupling constant for spin-spin interactions.
  Returns
  ---
  nonzero_elements : list of int
    The number of nonzero elements in each row of the hamiltonian
  nze_locations : list of int
    The locations of nonzero elements in each row. First `nonzero_elements[0]` entries refer to row `0`, and so on
  nze_values : list of float
    The nonzero elements themselves
  """
  st = mt.perf_counter()
  state_n = 2**N
  nonzero_elements = [] # Number of nonzero elements in each row
  nze_locations = [] # Locations of nonzero elements in each row
  nze_values = [] # Values of each nonzero element
  for s in range(0, state_n):
    """ Build row s of the Hamiltonian """
    row = np.zeros(state_n)
    for i in range(0, N):
      j = (i+1)%N # rightward nearest neighbor of i, with periodicity enforced
      if get_bit(s,i) == get_bit(s,j): 
        row[s] += .25
      else:
        row[s] += -.25
        sp = flip_bits(s,i,j)
        row[sp] = .5
    """ Compress & save row s """
    nze_s = 0
    for i in range(0,state_n):
      if row[i] != 0:
        nze_s += 1
        nze_locations.append(i)
        nze_values.append(float(row[i]))
    nonzero_elements.append(nze_s)
    
    if s%1000 == 0: mt.timeprint(st, f"Compressed H building {(s/state_n * 100):.2f}% done")
  for nze in nze_values: nze *= J
  mt.timeprint(st, f"Compressed H building done!")
  return nonzero_elements, nze_locations, nze_values
