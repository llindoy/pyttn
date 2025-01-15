import os
os.environ['OPENBLAS_NUM_THREADS']="1"

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
import h5py

sys.path.append("../../")
from pyttn import *
from pyttn.utils import visualise_tree
from hamiltonian import *

fs = 41.341374575751

def run_mps_dynamics(ofname='multiset_model_B/out_mps_1.h5'):
    tmax = 200*fs
    dt = 0.125*fs
    nsteps = int(tmax/(dt))+1

    Nmodes = 113
    mdims = [0 for i in range(Nmodes)]
    mdims[0] = 128

    for i in range(1, Nmodes):
        mdims[i] = 30
    
    ldims = [0 for i in range(Nmodes)]
    ldims[0] = 24
    for i in range(1, Nmodes):
        ldims[i] = 12

    #set up the sum of product operator Hamiltonian
    H = hamiltonian()

    chimax = 24

    sysinf = system_modes(Nmodes)
    for i in range(Nmodes):
        sysinf[i] = boson_mode(mdims[i])

    topo = ntreeBuilder.mps_tree(mdims, chimax, ldims)

    A = ms_ttn(topo, 26, dtype=np.complex128)
    As = ttn(topo, dtype=np.complex128)
    state = [[0 for i in range(Nmodes)] for j in range(26)]
    coeff = np.zeros(26, dtype=np.complex128)
    coeff[0] = 1
    A.set_state(coeff, state)

    mel = matrix_element(A)
    h = multiset_sop_operator(H, A, sysinf)

    sweepA = tdvp(A, h, krylov_dim = 12)
    sweepA.expmv_tol=1e-12
    sweepA.dt = dt
    sweepA.coefficient = -1.0j

    res = np.zeros((nsteps+1, 26), dtype=np.complex128)
    maxchi = np.zeros(nsteps+1)



    for i in range(26):
        As = ttn(A.slice(i))
        res[0, i] = mel(As)

    ts = np.logspace(-7, -1, 10)*fs
    tp = 0
    for i in range(len(ts)):
        sweepA.dt = ts[i]-tp
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        tp = ts[i]

    sweepA.dt = dt

    mchi = 0
    for j in range(26):
        As = ttn(A.slice(i))
        if(As.maximum_bond_dimension()> mchi):
            mchi = As.maximum_bond_dimension()
        res[1, j] = mel(As)


    maxchi[1] = mchi
    for i in range(1, nsteps):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        mchi = 0
        for j in range(26):
            As = ttn(A.slice(j))
            if(As.maximum_bond_dimension()> mchi):
                mchi = As.maximum_bond_dimension()
            res[i+1, j] = mel(As)
        maxchi[i+1] = mchi

        print((i+1)*dt, mel(A), maxchi)
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

def run_mctdh_dynamics(ofname='multiset_model_B/out_mlmctdh_4.h5'):
    tmax = 200*fs
    dt = 0.1*fs
    nsteps = int(tmax/(dt))+1

    Nmodes = 113
    mdims = [0 for i in range(Nmodes)]
    mdims[0] = 128

    for i in range(1, Nmodes):
        mdims[i] = 30
    
    ldims = [0 for i in range(Nmodes)]
    ldims[0] = 24
    for i in range(1, Nmodes):
        ldims[i] = 12

    mdims = np.array(mdims, dtype=int)
    ldims = np.array(ldims, dtype=int)
    #set up the sum of product operator Hamiltonian
    H = hamiltonian()

    chimax = 4

    sysinf = system_modes(Nmodes)
    for i in range(Nmodes):
        sysinf[i] = boson_mode(mdims[i])

    chimax = 4
    chimax2 = 4
    chimax3 = 4

    mode_F = [1+i for i in range(8)]
    modeOT_lf = [] 
    modeOT_hf = [] 
    for n in range(13):
        for l in range(6):
            modeOT_lf.append(1+8+ot_mode_index(13, n, l))

        for l in range(6, 8):
            modeOT_hf.append(1+8+ot_mode_index(13, n, l))

    sysinf = system_modes(Nmodes)
    for i in range(Nmodes):
        sysinf[i] = boson_mode(mdims[i])


    topo = ntree("(1(%d(%d(128)))(%d(%d)(%d)))"%(chimax, ldims[0], chimax,chimax, chimax))
    ntreeBuilder.mlmctdh_subtree(topo()[0], mdims[mode_F[:]], 2, chimax2, ldims[mode_F[:]])
    ntreeBuilder.mlmctdh_subtree(topo()[1][0], mdims[modeOT_lf[:]], 2, chimax, ldims[modeOT_lf[:]])
    ntreeBuilder.mlmctdh_subtree(topo()[1][1], mdims[modeOT_hf[:]], 2, chimax3, ldims[modeOT_hf[:]])

    A = ms_ttn(topo, 26, dtype=np.complex128)
    state = [[0 for i in range(Nmodes)] for j in range(26)]
    coeff = np.zeros(26, dtype=np.complex128)
    coeff[0] = 1
    A.set_state(coeff, state)

    mel = matrix_element(A)
    h = multiset_sop_operator(H, A, sysinf)

    sweepA = tdvp(A, h, krylov_dim = 12)
    sweepA.expmv_tol=1e-12
    sweepA.dt = dt
    sweepA.coefficient = -1.0j

    res = np.zeros((nsteps+1, 26), dtype=np.complex128)
    maxchi = np.zeros(nsteps+1)

    for i in range(26):
        As = ttn(A.slice(i))
        res[0, i] = mel(As)

    ts = np.logspace(-7, -1, 10)*fs
    tp = 0
    for i in range(len(ts)):
        sweepA.dt = ts[i]-tp
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        tp = ts[i]

    sweepA.dt = dt

    mchi = 0
    for j in range(26):
        As = ttn(A.slice(i))
        if(As.maximum_bond_dimension()> mchi):
            mchi = As.maximum_bond_dimension()
        res[1, j] = mel(As)
    maxchi[1] = mchi
    for i in range(1, nsteps):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        mchi = 0
        for j in range(26):
            As = ttn(A.slice(j))
            if(As.maximum_bond_dimension()> mchi):
                mchi = As.maximum_bond_dimension()
            res[i+1, j] = mel(As)
        maxchi[i+1] = mchi

        print((i+1)*dt, mel(A), maxchi)
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



#run_mctdh_dynamics()
run_mps_dynamics()
