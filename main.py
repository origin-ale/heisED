import numpy as np
from scipy.linalg import eigh_tridiagonal

from build_hamiltonian import build_compressed_hamiltonian
from lanczos import spinH_compr_lanczos_gs
from file_output import grounds_to_file
import mytiming as mt

J = 1.
L = 2

iter_duration = 0. # How long the previous iteration lasted
max_duration = 300 # Maximum time for an iteration before program is set to stop, in seconds

lengths = []
energies = []
en_lens = []
durations = []

print(16*'=' + f" Running with J = {J} until iterations take {max_duration} s " + 16*'=')
while(iter_duration < max_duration):
  iter_start = mt.perf_counter()

  print(16*'=' + f" L = {L} " + 16*'=')
  compact_H = build_compressed_hamiltonian(L, J)
  gs = spinH_compr_lanczos_gs(compact_H, L)

  iter_end = mt.perf_counter()
  iter_duration = iter_end - iter_start

  print(f'{L} spins with PBCs, J={J}:\n\tE0 {gs:.7f}\n\tE0/L {gs/L:.7f}\nIteration took {iter_duration:.3f} s')

  lengths.append(L)
  energies.append(gs)
  en_lens.append(gs/L)
  durations.append(iter_duration)
  grounds_to_file(lengths, energies, en_lens, durations, J)

  L += 2 # Only use even site numbers to avoid oscillations

print(16*'=' + f" Maximum duration {max_duration} s passed at L = {L-2} " + 16*'=')