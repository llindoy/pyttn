import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../")
from pyttn import *
from pyttn import oqs
from numba import jit

N=4
sysinf = system_modes(N)
sysinf[0] = generic_mode(2)
for i in range(N-1):
    sysinf[i+1] = boson_mode(10)
print(sysinf.liouville_space())
