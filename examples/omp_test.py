import sys
sys.path.append("../")
from pyttn import *

import pyttn
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
for i in range(N-1):
    H += J*sOP("z",i)*sOP("z",i+1)
    H += -h*sOP("x",i)

H += -h*sOP("x",N-1)

# set up TTN
chi = 256
dims = [2 for i in range(N)]

sysinf = system_modes(N)
for i in range(N):
    sysinf[i] = qubit_mode()


topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi, [i+2 for i in range(N)])

A = multiset_ttn(topo, 48, dtype=np.complex128)
A.nthreads = 1
for i in range(100):
    A.random()
    A.reset_orthogonality_centre()
    t1 = time.time()
    A.orthogonalise()
    t2 = time.time()
    print(i, t2-t1)
