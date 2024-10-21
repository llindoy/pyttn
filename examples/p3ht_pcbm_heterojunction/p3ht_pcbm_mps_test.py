import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
import h5py

sys.path.append("../../")
from pyttn import *
from pyttn.utils import visualise_tree
from model_hamiltonian import *

fs = 41.341374575751

def run_mps_dynamics(ofname='out_mps_two_integ_128.h5'):
    tmax = 200*fs
    tmax1 = 15*fs
    dt = 0.25*fs
    nsteps = int(tmax/(dt))+1
    nsteps1 = int(tmax1/(dt))+1
    nsteps2 = nsteps-nsteps1

    Nmodes = 114
    mdims = [0 for i in range(Nmodes)]
    mdims[0] = 26
    mdims[1] = 128

    for i in range(2, Nmodes):
        mdims[i] = 30
    
    ldims = [0 for i in range(Nmodes)]
    ldims[0] = 26
    ldims[1] = 24
    for i in range(2, Nmodes):
        ldims[i] = 12

    ldims0 = [0 for i in range(Nmodes)]
    ldims0[0] = 26
    for i in range(1, Nmodes):
        ldims0[i] = 6

    #set up the sum of product operator Hamiltonian
    H, opdict = hamiltonian()

    chimax = 128
    chi0 = 2

    sysinf = system_modes(Nmodes)
    sysinf[0] = generic_mode(26)
    for i in range(1, Nmodes):
        sysinf[i] = boson_mode(mdims[i])


    topo = ntreeBuilder.mps_tree(mdims, chi0, ldims0)
    capacity = ntreeBuilder.mps_tree(mdims, chimax, ldims)

    A = ttn(topo, capacity, dtype=np.complex128)
    state = np.zeros(Nmodes, dtype=int)
    A.set_state(state)

    ops = []
    for i in range(13):
        ops.append(site_operator(sOP("|LE%d><LE%d|"%(i, i), 0), sysinf, opdict))
        ops.append(site_operator(sOP("|CS%d><CS%d|"%(i, i), 0), sysinf, opdict))

    mel = matrix_element(A)
    h = sop_operator(H, A, sysinf, opdict)

    sweepA = tdvp(A, h, krylov_dim = 24, expansion='subspace', subspace_neigs=6)
    sweepA.spawning_threshold = 1e-6
    sweepA.unoccupied_threshold = 1e-6
    sweepA.expmv_tol=1e-12
    sweepA.minimum_unoccupied=1
    sweepA.dt = dt
    sweepA.coefficient = -1.0j


    sweepB = tdvp(A, h, krylov_dim = 24)
    sweepB.expmv_tol=1e-12
    sweepB.dt = dt
    sweepB.coefficient = -1.0j

    res = np.zeros((nsteps+1, len(ops)), dtype=np.complex128)
    maxchi = np.zeros(nsteps+1)

    for i in range(len(ops)):
        res[0, i] = mel(ops[i], A)

    ts = np.logspace(-7, -1, 10)*fs
    tp = 0
    for i in range(len(ts)):
        sweepA.dt = ts[i]-tp
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        tp = ts[i]
    sweepA.dt = dt

    for j in range(len(ops)):
        res[1, j] = mel(ops[j], A)
    maxchi[1] = A.maximum_bond_dimension()

    i = 0
    for i in range(1, nsteps1):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        for j in range(len(ops)):
            res[i+1, j] = mel(ops[j], A)
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt/fs, mel(h, A), A.maximum_bond_dimension())
        if(i % 10 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nsteps+1)*dt/fs))
            for j in range(13):
                h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
                h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nsteps+1)*dt/fs))
        for j in range(13):
            h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
            h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
        h5.create_dataset('maxchi', data=maxchi)
        h5.close()

    sweepB.prepare_environment(A, h)
    for i in range(nsteps1, nsteps):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        for j in range(len(ops)):
            res[i+1, j] = mel(ops[j], A)
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt/fs, mel(h, A), A.maximum_bond_dimension())
        if(i % 10 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nsteps+1)*dt/fs))
            for j in range(13):
                h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
                h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nsteps+1)*dt/fs))
        for j in range(13):
            h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
            h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
        h5.create_dataset('maxchi', data=maxchi)
        h5.close()


run_mps_dynamics()
