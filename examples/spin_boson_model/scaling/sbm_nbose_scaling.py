import numpy as np
import time
import sys
sys.path.append("../../../")
sys.path.append("../")

from pyttn import *
from sbm_core import *

def bath_size_scaling_binary_ternary():
    chi = 20
    Nbose = [4, 8, 10, 16, 20, 32, 48, 64]

    nb = 64
    timings_mps = []
    stdevs_mps = []

    nstep = 20
    for nbose in Nbose:
        m, std = spin_boson_test(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, degree = 1, compress = True)
        timings_mps.append(m)
        stdevs_mps.append(std/np.sqrt(1.0*nstep))

    h5 = h5py.File("bath_size_scaling_mps.h5", 'w')
    h5.create_dataset('Nbose', data=np.array(Nbose))
    h5.create_dataset('mean', data=np.array(timings_mps))
    h5.create_dataset('stderr', data=np.array(stdevs_mps))
    h5.close()

    nb = 64
    timings_binary = []
    stdevs_binary = []

    for nbose in Nbose:
        print(nbose)
        m, std = spin_boson_test(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, degree = 2, compress = True)
        timings_binary.append(m)
        stdevs_binary.append(std/np.sqrt(1.0*nstep))

    h5 = h5py.File("bath_size_scaling_binary.h5", 'w')
    h5.create_dataset('Nbose', data=np.array(Nbose))
    h5.create_dataset('mean', data=np.array(timings_binary))
    h5.create_dataset('stderr', data=np.array(stdevs_binary))
    h5.close()
    timings_3 = []
    stdevs_3 = []
    for nbose in Nbose:
        print(nbose)
        m, std = spin_boson_test(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, nstep=nstep, degree = 3, compress = True)
        timings_3.append(m)
        stdevs_3.append(std/np.sqrt(1.0*nstep))

    h5 = h5py.File("bath_size_scaling_ternary.h5", 'w')
    h5.create_dataset('Nbose', data=np.array(Nbose))
    h5.create_dataset('mean', data=np.array(timings_3))
    h5.create_dataset('stderr', data=np.array(stdevs_3))
    h5.close()

bath_size_scaling_binary_ternary()
