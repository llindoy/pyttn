import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../../../")
sys.path.append("../")

from pyttn import *
from sbm_core import *

def bath_size_scaling_binary_ternary():
    chis = [4, 8, 16, 32, 48, 64, 96, 128]
    nbose = 20
    nb = 32
    timings_binary = []
    stdevs_binary = []

    for chi in chis:
        print(chi)
        m, std = spin_boson_test(nb, 2.0, 25, 0.0, 1.0, chi, nbose, 0.001, nstep=1, compress = True)
        timings_binary.append(m)
        stdevs_binary.append(std)

    plt.figure(1)
    plt.loglog(chis, timings_binary)

    chis = [4, 8, 16, 32, 48, 64]
    timings_3 = []
    stdevs_3 = []
    for chi in chis:
        print(chi)

        m, std = spin_boson_test(nb, 2.0, 25, 0.0, 1.0, chi, nbose, 0.001, nstep=1, degree = 3, compress = True)
        timings_3.append(m)
        stdevs_3.append(std)

    plt.loglog(chis, timings_3)
    plt.show()

bath_size_scaling_binary_ternary()
