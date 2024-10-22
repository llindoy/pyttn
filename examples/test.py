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

adag_2 = site_operator(sOP("adag", 2), sysinf)
A.apply_one_body_operator(adag_2)
print(adag_2)

adag_2 = site_operator(sOP("adag", 2), sysinf)

print(adag_2)
B.apply_one_body_operator(adag_2)

mel = matrix_element(A)
print(mel(A, A), mel(B, B), mel(A, B))


adb = SOP(N)
adb += sOP("adag",3)*sOP("a",3)#*sOP("adag",8)
adag_3_a_5_adag8 = sop_operator(adb, A, sysinf)
print("A", mel(adag_3_a_5_adag8, A), mel(B, B), mel(A, B))

ad = 1.0*sOP("n",3)
adag_3_a_5_adag8 = product_operator(ad, sysinf)
print("B", mel(adag_3_a_5_adag8, A), mel(B, B), mel(A, B))
A.apply_product_operator(adag_3_a_5_adag8)
print("C", mel(adag_3_a_5_adag8, B), mel(B, A), mel(A, B))
A = copy.deepcopy(B)

ad = sOP("n",3)
adag_3_a_5_adag8 = product_operator(ad, sysinf)
print("D", mel(adag_3_a_5_adag8, A), mel(B, B), mel(A, B))
A.apply_product_operator(adag_3_a_5_adag8)
print("E", mel(adag_3_a_5_adag8, B), mel(B, A), mel(A, B))
A = copy.deepcopy(B)

ad = sOP("adag",3)*sOP("a", 3)#*sOP("adag",4)*sOP("a", 4)
adag_3_a_5_adag8 = product_operator(ad, sysinf)
print("F", mel(adag_3_a_5_adag8, A), mel(B, B), mel(A, B))
A.apply_product_operator(adag_3_a_5_adag8)
print("G", mel(adag_3_a_5_adag8, B), mel(B, A), mel(A, B))
A = copy.deepcopy(B)
exit()
