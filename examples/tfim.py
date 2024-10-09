import sys
sys.path.append("../")
from pyttn import *

from pyttn._pyttn import apply_sop_to_ttn
from pyttn.utils import visualise_tree

import matplotlib.pyplot as plt

import numpy as np
import random
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
Hsop = sSOP()
for i in range(N-1):
    H += J*sOP("z",i)*sOP("z",i+1)
    Hsop += J*sOP("z",i)*sOP("z",i+1)
    H += -h*sOP("x",i)
    Hsop += -h*sOP("x",i)

H += -h*sOP("x",N-1)
Hsop += -h*sOP("x",N-1)
H2sop = Hsop*Hsop

H2 = SOP(N)
for t in H2sop:
    H2 += t
print(H2)


H3 = SOP(N)

inds = ['x', 'y', 'z']
for i in range(16):
    ret = sPOP()
    for ind in range(N):
        ret *= sOP(inds[random.randint(0, 2)], ind)
    H3 += random.uniform(-1, 1)*ret

# set up TTN
chi =16
dims = [2 for i in range(N)]

sysinf = system_modes(N)
for i in range(N):
    sysinf[i] = qubit_mode()


topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi, [i+2 for i in range(N)])

visualise_tree(topo)
plt.show()

print(topo)
exit()

ntreeBuilder.insert_basis_nodes(topo)
ntreeBuilder.collapse_bond_matrices(topo)
ntreeBuilder.sanitise_bond_dimensions(topo)

print(topo)

A = ttn(topo, dtype=np.complex128)
A.set_state([0 for i in range(N)])
A.random()

h = sop_operator(H, A, sysinf, compress=True)
h2 = sop_operator(H2, A, sysinf, compress=True)
op = sop_operator(H3, A, sysinf, compress=True)
mel = matrix_element(A)

print(sysinf.nmodes())

# set up simulation

ndmrg = 20

dmrg_sweep = dmrg(A, h, krylov_dim=8)
dmrg_sweep.prepare_environment(A, h)
dmrg_sweep.restarts = 1

B = ttn(topo, dtype=np.complex128)
C = ttn(topo, dtype=np.complex128)
D = ttn(topo, dtype=np.complex128)

#apply it using the apply function.  This takes an optional coefficient that scales the result
apply_sop_to_ttn(h, A, B, coeff=1.0)
apply_sop_to_ttn(h2, A, C)
apply_sop_to_ttn(op, A, D)

print("H: ", H)
print("H^2: ", H2)
print("op: ", H3)

print(A.maximum_bond_dimension(), B.maximum_bond_dimension(), C.maximum_bond_dimension(), D.maximum_bond_dimension())
print("<H>: ", mel(A, B), mel(h, A), dmrg_sweep.E())
print("<H^2>: ", mel(A, C), mel(B, B), mel(h2, A))
print("<O>: ", mel(A, D), mel(op, A))

for i in range(ndmrg):
    dmrg_sweep.step(A,h)

    B = h@A
    C = h2@A
    D = op@A
    print(A.maximum_bond_dimension(), B.maximum_bond_dimension(), C.maximum_bond_dimension(), D.maximum_bond_dimension())
    print("<H>: ", mel(A, B), mel(h, A), dmrg_sweep.E())
    print("<H^2>: ", mel(A, C), mel(B, B), mel(h2, A))
    print("<O>: ", mel(A, D), mel(op, A))

    B.truncate(nchi=16)
    #B.truncate(tol=1e-12, nbond)
    C.truncate(nchi=16)
    #C.truncate(tol=1e-10)
    D.truncate(nchi=16)
    #D.truncate(tol=1e-10)
    
    print(A.maximum_bond_dimension(), B.maximum_bond_dimension(), C.maximum_bond_dimension(), D.maximum_bond_dimension())
    print("<H>: ", mel(A, B), mel(h, A), dmrg_sweep.E())
    print("<H^2>: ", mel(A, C), mel(B, B), mel(h2, A))
    print("<O>: ", mel(A, D), mel(op, A))


