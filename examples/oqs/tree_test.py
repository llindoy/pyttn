import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
import pyttn
from pyttn import *
from pyttn import oqs
from numba import jit


N=10
# set up TTN
chi = 8
dims = [2 for i in range(N-1)]

sysinf = system_modes(N)
for i in range(N):
    sysinf[i] = spin_mode(2)

topo = ntree("(1(2(2))(2))")
ntreeBuilder.mps_subtree(topo()[1], dims, chi)
topo().insert(2)
topo().insert(2)
topo().insert(2)
topo().insert(2)
print(topo)
ntreeBuilder.insert_basis_nodes(topo)
print(topo)


