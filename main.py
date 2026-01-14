import numpy as np
from scipy.linalg import eigh_tridiagonal

from build_hamiltonian import build_hamiltonian
from lanczos import spinH_lanczos
import mytiming as mt

L = 12
J = 1.

H = build_hamiltonian(L, J)

tridiag = spinH_lanczos(H)

gs_energy = eigh_tridiagonal(tridiag.diagonal(), 
                             tridiag.diagonal(offset=1), 
                             eigvals_only=True, 
                             select='i', 
                             select_range=[0,0])

print(f'{L} spins with PBCs, J={J}:\n\tE0 {gs_energy[0]:.7f}\n\tE0/L {gs_energy[0]/L:.7f}')