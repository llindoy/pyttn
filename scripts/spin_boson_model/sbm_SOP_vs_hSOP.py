import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../../")

from pyttn import *
from sbm_core import *

def bath_size_scaling_SOP_vs_hSOP():
    chi = 20
    nbose = 10
    Nbs = [2, 4, 8, 16, 32, 64, 128, 256]
    timings_hSOP = []
    stdevs_hSOP = []

    timings_SOP = []
    stdevs_SOP = []
    for nb in Nbs:
        print(nb)
        m, std = spin_boson_test(nb, 4.0, 25, 0.0, 2.0, chi, nbose, 0.001, nstep=1, compress = True)
        timings_hSOP.append(m)
        stdevs_hSOP.append(std)

    for nb in Nbs:
        m, std = spin_boson_test(nb, 4.0, 25, 0.0, 2.0, chi, nbose, 0.001, nstep=1, compress = False)
        timings_SOP.append(m)
        stdevs_SOP.append(std)
        print(nb, timings_hSOP[-1], stdevs_hSOP[-1], timings_SOP[-1], stdevs_SOP[-1])

    plt.figure(1)
    plt.loglog(Nbs, timings_hSOP)
    plt.loglog(Nbs, timings_SOP)
    plt.show()

bath_size_scaling_SOP_vs_hSOP()
