import numpy as np
from scipy.linalg import eigh_tridiagonal

from build_hamiltonian import build_hamiltonian
from lanczos import spinH_lanczos, spinH_lanczos_gs
from file_output import grounds_to_file
import mytiming as mt

J = .5
L = 1

iter_duration = 0. # How long the last iteration lasted
max_duration = 300 # Maximum time for an iteration before program is set to stop, in seconds

lengths = []
energies = []
en_lens = []
durations = []

print(16*'=' + f" Running with J = {J} until iterations take {max_duration} s " + 16*'=')
while(iter_duration < max_duration):
  iter_start = mt.perf_counter()

  print(16*'=' + f" L = {L} " + 16*'=')
  H = build_hamiltonian(L, J)

  # tridiag = spinH_lanczos(H)
  # gs_energy = eigh_tridiagonal(tridiag.diagonal(), 
  #                             tridiag.diagonal(offset=1), 
  #                             eigvals_only=True, 
  #                             select='i', 
  #                             select_range=[0,0])
  # gs = gs_energy[0]
  gs = spinH_lanczos_gs(H)

  print(f'{L} spins with PBCs, J={J}:\n\tE0 {gs:.7f}\n\tE0/L {gs/L:.7f}')

  iter_end = mt.perf_counter()
  iter_duration = iter_end - iter_start

  lengths.append(L)
  energies.append(gs)
  en_lens.append(gs/L)
  durations.append(iter_duration)

  L += 1

print(16*'=' + f" Maximum duration {max_duration} s passed at L = {L-1} " + 16*'=')
grounds_to_file(lengths, energies, en_lens, durations, J)