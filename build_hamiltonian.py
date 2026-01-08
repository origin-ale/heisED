import numpy as np

def get_bit(n: int, i: int):
  """
  Get the ith bit of an int n
  """
  return n >> i & 1

def flip_bits(n: int, i: int, j:int):
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
  state_n = 2**N
  H = np.zeros((state_n, state_n))
  for s in range(0, state_n-1):
    for i in range(0, N-1):
      j = (i+1)/N # rightward nearest neighbor of i, with periodicity enforced
      if get_bit(s,i) == get_bit(s,j): 
        H[s,s] += .25
      else:
        H[s,s] += -.25
        sp = flip_bits(s,i,j)
        H[s,sp] = .5

  return