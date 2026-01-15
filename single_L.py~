from build_hamiltonian import build_hamiltonian
from lanczos import spinH_lanczos, spinH_lanczos_gs

J = 1
L = 13

H = build_hamiltonian(L, J)
gs = spinH_lanczos_gs(H)

print(f'{L} spins with PBCs, J={J}:\n\tE0 {gs:.7f}\n\tE0/L {gs/L:.7f}')