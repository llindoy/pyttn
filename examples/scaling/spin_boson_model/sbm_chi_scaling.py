import numpy as np
import time
import sys
import h5py

from sbm_timing_helper import sbm_dynamics

def chi_scaling_mps_binary_ternary(adaptive = False):
    nbose = 10
    nb = 32
    nstep = 20

    jtype=""
    if(adaptive):
        jtype="_subspace"

    chis = [2, 4, 8, 16, 32, 48, 64]#, 96, 128, 160, 192, 256, 384]
    timings_mps = []
    stdevs_mps = []

    chivs = []
    for chi in chis:
        m, std = sbm_dynamics(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, degree = 1, compress = True, adaptive=adaptive)
        print(chi, m, std/np.sqrt(1.0*nstep))
        chivs.append(chi)
        timings_mps.append(m)
        stdevs_mps.append(std/np.sqrt(1.0*nstep))
        h5 = h5py.File("chi_scaling_mps"+jtype +".h5", 'w')
        h5.create_dataset('chis', data=np.array(chivs))
        h5.create_dataset('mean', data=np.array(timings_mps))
        h5.create_dataset('stderr', data=np.array(stdevs_mps))
        h5.close()


    timings_binary = []
    stdevs_binary = []
    chivs = []

    chis = [2, 4, 8, 16, 32]#, 48, 64, 96, 128, 160, 192]
    for chi in chis:
        m, std = sbm_dynamics(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, degree = 2, compress = True, adaptive=adaptive)
        chivs.append(chi)
        timings_binary.append(m)
        stdevs_binary.append(std/np.sqrt(1.0*nstep))
        print(chi, m, std/np.sqrt(1.0*nstep))

        h5 = h5py.File("chi_scaling_binary"+jtype +".h5", 'w')
        h5.create_dataset('chis', data=np.array(chivs))
        h5.create_dataset('mean', data=np.array(timings_binary))
        h5.create_dataset('stderr', data=np.array(stdevs_binary))
        h5.close()


    nb = 27
    chis = [2, 4, 8, 16]#, 24, 32, 40, 48, 56]
    timings_3 = []
    stdevs_3 = []
    chivs = []
    for chi in chis:
        m, std = sbm_dynamics(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, degree = 3, compress = True, adaptive=adaptive)
        chivs.append(chi)
        timings_3.append(m)
        stdevs_3.append(std/np.sqrt(1.0*nstep))

        print(chi, m, std/np.sqrt(1.0*nstep))

        h5 = h5py.File("chi_scaling_ternary"+jtype +".h5", 'w')
        h5.create_dataset('chis', data=np.array(chivs))
        h5.create_dataset('mean', data=np.array(timings_3))
        h5.create_dataset('stderr', data=np.array(stdevs_3))
        h5.close()


chi_scaling_mps_binary_ternary(adaptive=False)
#chi_scaling_mps_binary_ternary(adaptive=True)
