import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../")
import pyttn
from pyttn import *
from pyttn import oqs, utils

# Number of physical degrees of freedom
N = 10

# Maximum bond dimension
chi = 8

# Vector of dimensions of the local Hilbert space in the physical nodes
dims = [2 for _ in range(N)]

# Vector of dimensions of the basis transformation nodes
basis_nodes_dims = [2 for _ in range(N)]

# Max number of child nodes for each node. Always larger than 1
degree = 2


# Set up the tree topology
topo = ntreeBuilder.mlmctdh_tree(dims, degree, chi, basis_nodes_dims)

# Initialise system information variable
sysinf = system_modes(N)

# Specify that all modes are bosonic, with Hilbert space dimensions defined above.
for i in range(N):
    sysinf[i] = fermion_mode()


# Create TTN and initialise to a random state
A = ttn(topo)
A.random()

# Create one-site operator
op = product_operator(fOP("n",4), sysinf)

# Create B = n_4 @ A(t), copy of A that will be time evolved
B = copy.deepcopy(A)
B @= op


# Calculate matrix element
mel = matrix_element()
print("<A|n_4(0)n_4(0)|A>:   ", mel(op, A, B))



# Create Hamiltonian for time evolution
H = SOP(N)

H += 2*fOP("n", 4)
H += 2*fOP("n", 5)
H += 2*fOP("adag", 5)
H += 2*fOP("a", 5)
H += 5*fOP("a", 4)*fOP("adag",5)

#H.jordan_wigner(sysinf)

h = sop_operator(H, A, sysinf)



# Initialise time-evolution engine
sweep = tdvp(B, h, krylov_dim = 12, expansion='subspace')
sweep.coefficient = -1.0j
sweep.dt = 1e-2

# Perform time evolution
for i in range(1,10):
    sweep.step(B, h)
    print("<A|n_4(0)n_4(%i*dt)|A>:" % i, mel(op, A, B))
