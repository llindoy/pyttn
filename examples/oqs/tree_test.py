import numpy as np
import time
import sys
import h5py
import scipy
import copy
import random

sys.path.append("../../")
import pyttn
from pyttn import *
from pyttn.utils import visualise_tree
from numba import jit

N=10
# set up TTN
chi = 8
dims = [2 for i in range(N-1)]

sysinf = system_modes(N)
for i in range(N):
    sysinf[i] = spin_mode(2)

topo = ntree("(1(2))")
ntreeBuilder.mps_subtree(topo(), dims, chi)
topo().insert(2)
topo().insert(2)
topo().insert(2)
topo().insert(2)

for i in range(100):
    r = 0.05
    for node in topo:
        if random.uniform(0, 1)<r:
            node.insert(random.randint(2,9))
            break



import matplotlib.pyplot as plt
plt.figure(1)
visualise_tree(topo)

plt.figure(2)
ntreeBuilder.sanitise(topo)
visualise_tree(topo)

plt.show()
