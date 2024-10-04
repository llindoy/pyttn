import sys
sys.path.append("../")
from pyttn import *

import numpy as np
import time
import h5py
import scipy
import copy
from numba import jit

# set up Hamiltonian

N = 16
J = 1
h = 1

H = SOP(N)
for i in range(N-1):
    H += -4.0*J*sOP("sz",i)*sOP("sz",i+1)
    H += -2.0*h*sOP("sx",i)
H += -2.0*h*sOP("sx",N-1)
print(H)


# set up TTN
chi = 8
dims = [2 for i in range(N)]

sysinf = system_modes(N)
for i in range(N):
    sysinf[i] = spin_mode(2)

topo = ntreeBuilder.mps_tree(dims, chi)

print(topo)

ntreeBuilder.insert_basis_nodes(topo)
ntreeBuilder.collapse_bond_matrices(topo)
ntreeBuilder.sanitise_bond_dimensions(topo)

print(topo)

A = ttn(topo, dtype=np.complex128)
A.random()

h = sop_operator(H, A, sysinf, compress=True)

mel = matrix_element(A)

print(sysinf.nmodes())

# set up simulation

ndmrg = 4

dmrg_sweep = dmrg(A, h, krylov_dim=8)
dmrg_sweep.prepare_environment(A, h)
dmrg_sweep.restarts = 1

for i in range(ndmrg):
    dmrg_sweep.step(A,h)
    print(i, dmrg_sweep.E()/N)
