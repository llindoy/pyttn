import numpy as np
import time
import sys
import h5py
import scipy
import copy
import random

sys.path.append("../")
from pyttn import *
from pyttn.utils import visualise_tree
from numba import jit

N=8
# set up TTN
chi = 8
dims = [2 for i in range(N)]

topo = ntree("(1(2))")
ntreeBuilder.mps_subtree(topo(), dims, chi)

import matplotlib.pyplot as plt
plt.figure(1)
visualise_tree(topo)

topo = ntree("(1(2))")
ntreeBuilder.mlmctdh_subtree(topo(), dims, 2, chi, include_local_basis_transformation=False)
plt.figure(2)
visualise_tree(topo)

topo = ntree("(1(2))")
ntreeBuilder.mlmctdh_subtree(topo(), dims, 3, chi, include_local_basis_transformation=False)
plt.figure(3)
visualise_tree(topo)

plt.show()
