import numpy as np
import time
import sys
import h5py

from sbm_timing_helper import sbm_dynamics

def bath_size_scaling_binary_ternary():
    chi = 20
    nbose = 10
    Nbs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    timings_mps = []
    stdevs_mps = []

    nstep = 20
    for nb in Nbs:
        m, std = sbm_dynamics(nb, 1.0, 5.0, 1.0, 0.0, 1.0, chi, nbose, 0.01, degree =1, nstep=nstep, compress = True)
        timings_mps.append(m)
        stdevs_mps.append(std/np.sqrt(nstep))
        print(nb, m, std/np.sqrt(1.0*nstep))

    h5 = h5py.File("Nb_scaling_mps.h5", 'w')
    h5.create_dataset('chis', data=np.array(Nbs))
    h5.create_dataset('mean', data=np.array(timings_mps))
    h5.create_dataset('stderr', data=np.array(stdevs_mps))
    h5.close()

    timings_binary = []
    stdevs_binary = []

    for nb in Nbs:
        m, std = sbm_dynamics(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, compress = True)
        timings_binary.append(m)
        stdevs_binary.append(std/np.sqrt(nstep))
        print(nb, m, std/np.sqrt(1.0*nstep))

    h5 = h5py.File("Nb_scaling_binary.h5", 'w')
    h5.create_dataset('chis', data=np.array(Nbs))
    h5.create_dataset('mean', data=np.array(timings_binary))
    h5.create_dataset('stderr', data=np.array(stdevs_binary))
    h5.close()

    Nbs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    timings_3 = []
    stdevs_3 = []
    for nb in Nbs:
        m, std = sbm_dynamics(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, degree = 3, compress = True)
        timings_3.append(m)
        stdevs_3.append(std/np.sqrt(nstep))
        print(nb, m, std/np.sqrt(1.0*nstep))

    h5 = h5py.File("Nb_scaling_ternary.h5", 'w')
    h5.create_dataset('chis', data=np.array(Nbs))
    h5.create_dataset('mean', data=np.array(timings_3))
    h5.create_dataset('stderr', data=np.array(stdevs_3))
    h5.close()

bath_size_scaling_binary_ternary()
