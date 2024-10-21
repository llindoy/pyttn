import numpy as np
import time
import sys
import h5py
import scipy
import copy
from numba import jit
import matplotlib.pyplot as plt 

import sys
sys.path.append("../")
from pyttn import *
from pyttn.utils import visualise_tree


N = 10
chi = 8
dims = [15+i for i in range(N)]
degree = 2

topo = ntreeBuilder.mlmctdh_tree(dims, degree, chi)
print(topo)
visualise_tree(topo)
plt.show()


A = ttn(topo, dtype = np.complex128)
A.random()


sysinf = system_modes(N)
for i in range(N):
    sysinf[i] = boson_mode(dims[i])
    
B = copy.deepcopy(A)

adag_2 = site_operator_complex(sOP("adag", 2), sysinf)
A.apply_one_body_operator(adag_2)
print(adag_2)

adag_2 = site_operator(sOP("adag", 2), sysinf)

print(adag_2)
B.apply_one_body_operator(adag_2)

mel = matrix_element(A)
print(mel(A, A), mel(B, B), mel(A, B))
exit()


ad = sOP("adag",3)*sOP("a",5)*sOP("adag",8)
adag_3_a_5_adag8 = sop_operator(ad, sysinf)
#A.apply_product_operator(adag_3_a_5_adag8)
