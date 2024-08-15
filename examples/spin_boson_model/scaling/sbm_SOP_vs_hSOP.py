import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../../../")
sys.path.append("../")

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
    step = 20
    for nb in Nbs:
        print(nb)
        m, std = spin_boson_test(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, compress = True)
        timings_hSOP.append(m)
        stdevs_hSOP.append(std/np.sqrt(nstep))
    h5 = h5py.File("bath_size_scaling_hSOP.h5", 'w')
    h5.create_dataset('Nbs', data=np.array(Nbose))
    h5.create_dataset('mean', data=np.array(timings_hSOP))
    h5.create_dataset('stderr', data=np.array(stdevs_hSOP))
    h5.close()

    for nb in Nbs:
        m, std = spin_boson_test(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, compress = False)
        timings_SOP.append(m)
        stdevs_SOP.append(std/np.sqrt(nstep))
        print(nb, timings_hSOP[-1], stdevs_hSOP[-1], timings_SOP[-1], stdevs_SOP[-1])

    h5 = h5py.File("bath_size_scaling_SOP.h5", 'w')
    h5.create_dataset('Nbs', data=np.array(Nbose))
    h5.create_dataset('mean', data=np.array(timings_SOP))
    h5.create_dataset('stderr', data=np.array(stdevs_SOP))
    h5.close()

bath_size_scaling_SOP_vs_hSOP()
