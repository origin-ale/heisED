import numpy as np
from scipy.linalg import eigh_tridiagonal

from build_hamiltonian import build_hamiltonian
from lanczos import spinH_lanczos

# np.set_printoptions(precision = 3, suppress = True, legacy='1.25')

L = 12
J = 1.
H = build_hamiltonian(L, J)

tridiag = spinH_lanczos(H)

gs_energy = eigh_tridiagonal(tridiag.diagonal(), 
                             tridiag.diagonal(offset=1), 
                             eigvals_only=True, 
                             select='i', 
                             select_range=[0,0])

print(f'Ground state energy for {L} spins with PBCs in J={J} is {gs_energy[0]}, that is {gs_energy[0]/L:.3f} per site')