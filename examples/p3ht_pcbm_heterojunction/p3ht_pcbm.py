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

def run_mps_dynamics(ofname='model_B/out_mps_32.h5'):
    tmax = 200*fs
    dt = 0.1*fs
    nsteps = int(tmax/(dt))+1

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

    chimax = 32
    chi0 = 16

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
    for i in range(1, nsteps):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        for j in range(len(ops)):
            res[i+1, j] = mel(ops[j], A)
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt, mel(h, A), A.maximum_bond_dimension())
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

#set this up to use mode combination to make convergence easier
def run_mctdh_dynamics(ofname='model_B/out_mlmctdh_32.h5'):
    tmax = 200*fs
    dt = 0.1*fs
    nsteps = int(tmax/(dt))+1

    Nmodes = 114
    mdims = [0 for i in range(Nmodes)]
    mdims[0] = 26
    mdims[1] = 128

    for i in range(2, Nmodes):
        mdims[i] = 30
    
    mdims = np.array(mdims, dtype=int)

    ldims = [0 for i in range(Nmodes)]
    ldims[0] = 26
    ldims[1] = 24
    for i in range(2, Nmodes):
        ldims[i] = 12

    ldims0 = [0 for i in range(Nmodes)]
    ldims0[0] = 26
    for i in range(1, Nmodes):
        ldims0[i] = 6

    ldims = np.array(ldims, dtype=int)
    ldims0 = np.array(ldims0, dtype=int)
    #set up the sum of product operator Hamiltonian
    H, opdict = hamiltonian()


    chimax = 32
    chimax2 = 24
    chimax3 = 24
    chi0 = 16

    mode_F = [2+i for i in range(8)]
    modeOT_lf = [] 
    modeOT_hf = [] 
    for n in range(13):
        for l in range(6):
            modeOT_lf.append(1+1+8+ot_mode_index(13, n, l))

        for l in range(6, 8):
            modeOT_hf.append(1+1+8+ot_mode_index(13, n, l))

    sysinf = system_modes(Nmodes)
    sysinf[0] = generic_mode(26)
    for i in range(1, Nmodes):
        sysinf[i] = boson_mode(mdims[i])



    topo = ntree("(1(%d(26))(%d(%d(128)))(%d(%d)(%d)))"%(ldims0[0], chi0, ldims0[1], chi0, chi0, chi0))
    ntreeBuilder.mlmctdh_subtree(topo()[1], mdims[mode_F[:]], 2, chi0, ldims0[mode_F[:]])
    ntreeBuilder.mlmctdh_subtree(topo()[2][0], mdims[modeOT_lf[:]], 2, chi0, ldims0[modeOT_lf[:]])
    ntreeBuilder.mlmctdh_subtree(topo()[2][1], mdims[modeOT_hf[:]], 2, chi0, ldims0[modeOT_hf[:]])


    capacity = ntree("(1(%d(26))(%d(%d(128)))(%d(%d)(%d)))"%(mdims[0], chimax, ldims[1], chimax,chimax, chimax))
    ntreeBuilder.mlmctdh_subtree(capacity()[1], mdims[mode_F[:]], 2, chimax2, ldims[mode_F[:]])
    ntreeBuilder.mlmctdh_subtree(capacity()[2][0], mdims[modeOT_lf[:]], 2, chimax, ldims[modeOT_lf[:]])
    ntreeBuilder.mlmctdh_subtree(capacity()[2][1], mdims[modeOT_hf[:]], 2, chimax3, ldims[modeOT_hf[:]])

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

    for i in range(1,nsteps):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        for j in range(len(ops)):
            res[i+1, j] = mel(ops[j], A)
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt, mel(h, A), A.maximum_bond_dimension())
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


run_mctdh_dynamics()
#run_mps_dynamics()
