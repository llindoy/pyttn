import numpy as np
import time
import h5py
import sys

from sbm_timing_helper import sbm_dynamics

def bath_size_scaling_SOP_vs_hSOP():
    chi = 20
    nbose = 10
    Nbs = [2, 4, 8, 16, 32, 64, 128, 256]
    timings_hSOP = []
    stdevs_hSOP = []

    timings_SOP = []
    stdevs_SOP = []
    nstep = 20
    for nb in Nbs:
        m, std = sbm_dynamics(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, compress = True)
        timings_hSOP.append(m)
        stdevs_hSOP.append(std/np.sqrt(nstep))
        print(nb, m, std/np.sqrt(1.0*nstep))

    h5 = h5py.File("bath_size_scaling_hSOP.h5", 'w')
    h5.create_dataset('Nbs', data=np.array(Nbs))
    h5.create_dataset('mean', data=np.array(timings_hSOP))
    h5.create_dataset('stderr', data=np.array(stdevs_hSOP))
    h5.close()

    for nb in Nbs:
        m, std = sbm_dynamics(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, compress = False)
        timings_SOP.append(m)
        stdevs_SOP.append(std/np.sqrt(nstep))
        print(nb, m, std/np.sqrt(1.0*nstep))

    h5 = h5py.File("bath_size_scaling_SOP.h5", 'w')
    h5.create_dataset('Nbs', data=np.array(Nbs))
    h5.create_dataset('mean', data=np.array(timings_SOP))
    h5.create_dataset('stderr', data=np.array(stdevs_SOP))
    h5.close()

bath_size_scaling_SOP_vs_hSOP()
