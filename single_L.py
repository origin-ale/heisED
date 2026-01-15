from build_hamiltonian import build_compressed_hamiltonian
from lanczos import spinH_lanczos, spinH_lanczos_gs, spinH_compr_lanczos_gs

J = 1
L = 18

compact_H = build_compressed_hamiltonian(L, J)
gs = spinH_compr_lanczos_gs(compact_H, L)

print(f'{L} spins with PBCs, J={J}:\n\tE0 {gs:.7f}\n\tE0/L {gs/L:.7f}')