import numpy as np
import time
import h5py
import sys
sys.path.append("../../../")
sys.path.append("../")

from pyttn import *
from sbm_core import *

def bath_size_scaling_identity(degree, identity_opt=True):
    
    chi = 20
    if degree == 3:
        chi = 16
    if degree == 4:
        chi = 10
    nbose = 10
    Nbs = None
    if degree < 4:
        Nbs = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    else:
        Nbs = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    timings_hSOP = []
    stdevs_hSOP = []

    timings_SOP = []
    stdevs_SOP = []
    nstep = 10

    fname = "bath_size_scaling"
    if degree == 1:
        fname = fname +"_mps"
    elif degree == 2:
        fname = fname +"_binary"
    elif degree == 3:
        fname = fname +"_ternary"
    elif degree == 4:
        fname = fname +"_quaternary"

    if(identity_opt):
        fname = fname+"_id.h5"
    else:
        fname = fname+"_no_id.h5"
    for nb in Nbs:
        m, std = spin_boson_test(nb, 1.0, 5, 1.0, 0.0, 1.0, chi, nbose, 0.01, degree = degree, nstep=nstep, identity_opt = identity_opt, compress=True)
        timings_hSOP.append(m)
        stdevs_hSOP.append(std/np.sqrt(nstep))
        print(nb, timings_hSOP[-1])
        h5 = h5py.File(fname, 'w')
        h5.create_dataset('Nbs', data=np.array(Nbs))
        h5.create_dataset('mean', data=np.array(timings_hSOP))
        h5.create_dataset('stderr', data=np.array(stdevs_hSOP))
        h5.close()

#bath_size_scaling_identity(1, False)
#bath_size_scaling_identity(2, False)
#bath_size_scaling_identity(3, False)
#bath_size_scaling_identity(4, False)

bath_size_scaling_identity(1, True)
bath_size_scaling_identity(2, True)
bath_size_scaling_identity(3, True)
bath_size_scaling_identity(4, True)
