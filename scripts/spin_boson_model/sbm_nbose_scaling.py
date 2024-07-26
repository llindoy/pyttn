import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../../")

from pyttn import *
from sbm_core import *

def bath_size_scaling_binary_ternary():
    chi = 20
    Nbose = [4, 8, 10, 16, 20, 32, 48, 64, 80, 96, 112, 128, 192, 256]
    nb = 64
    timings_binary = []
    stdevs_binary = []

    for nbose in Nbose:
        print(nbose)
        m, std = spin_boson_test(nb, 4.0, 25, 0.0, 2.0, chi, nbose, 0.001, nstep=1, compress = True)
        timings_binary.append(m)
        stdevs_binary.append(std)

    plt.figure(1)
    plt.loglog(Nbose, timings_binary)

    timings_3 = []
    stdevs_3 = []
    for nbose in Nbose:
        print(nbose)
        m, std = spin_boson_test(nb, 4.0, 25, 0.0, 2.0, chi, nbose, 0.001, nstep=1, degree = 3, compress = True)
        timings_3.append(m)
        stdevs_3.append(std)

    plt.loglog(Nbose, timings_3)
    plt.show()

bath_size_scaling_binary_ternary()
