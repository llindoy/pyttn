import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../../../")
sys.path.append("../")

from pyttn import *
from sbm_core import *

def bath_size_scaling_binary_ternary():
    chi = 20
    nbose = 10
    Nbs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    timings_binary = []
    stdevs_binary = []

    for nb in Nbs:
        print(nb)
        m, std = spin_boson_test(nb, 2.0, 25, 0.0, 1.0, chi, nbose, 0.001, nstep=1, compress = True)
        timings_binary.append(m)
        stdevs_binary.append(std)

    plt.figure(1)
    plt.loglog(Nbs, timings_binary)

    Nbs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    timings_3 = []
    stdevs_3 = []
    for nb in Nbs:
        print(nb)
        m, std = spin_boson_test(nb, 2.0, 25, 0.0, 1.0, chi, nbose, 0.001, nstep=1, degree = 3, compress = True)
        timings_3.append(m)
        stdevs_3.append(std)

    plt.loglog(Nbs, timings_3)
    plt.show()

bath_size_scaling_binary_ternary()
