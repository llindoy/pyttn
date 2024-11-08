import numpy as np
import time
import sys
import copy
import h5py

sys.path.append("../../")
from pyttn import *
from hamiltonian import *

fs = 41.341374575751

def run_mps_dynamics(ofname='model_A_mps_30.h5'):
    tmax = 200*fs
    dt = 0.25*fs
    nsteps = int(tmax/(dt))+1

    Nmodes = 101
    mdims = [0 for i in range(Nmodes)]
    mdims[0] = 2
    mdims[1] = 70

    for i in range(2, Nmodes):
        mdims[i] = 30
    
    ldims = [0 for i in range(Nmodes)]
    ldims[0] = 2
    ldims[1] = 30
    for i in range(1, Nmodes):
        ldims[i] = 12

    ldims0 = [0 for i in range(Nmodes)]
    ldims0[0] = 2
    for i in range(1, Nmodes):
        ldims0[i] = 4

    #set up the sum of product operator Hamiltonian
    H, opdict = hamiltonian()

    chimax = 100
    chi0 = 8

    sysinf = system_modes(Nmodes)
    sysinf[0] = generic_mode(2)
    for i in range(1, Nmodes):
        sysinf[i] = boson_mode(mdims[i])


    topo = ntreeBuilder.mps_tree(mdims, chi0, ldims0)
    capacity = ntreeBuilder.mps_tree(mdims, chimax, ldims)

    A = ttn(topo, capacity, dtype=np.complex128)
    state = np.zeros(Nmodes, dtype=int)
    A.set_state(state)

    ops = []
    ops.append(site_operator(sOP("|LE0><LE0|", 0), sysinf, opdict))
    ops.append(site_operator(sOP("|CS0><CS0|", 0), sysinf, opdict))

    mel = matrix_element(A)
    h = sop_operator(H, A, sysinf, opdict)

    sweepA = tdvp(A, h, krylov_dim = 12, expansion='subspace', subspace_neigs=6)
    sweepA.spawning_threshold = 5e-6
    sweepA.minimum_unoccupied=0
    sweepA.dt = dt
    sweepA.coefficient = -1.0j

    res = np.zeros((nsteps+1, len(ops)), dtype=np.complex128)
    maxchi = np.zeros(nsteps+1)

    for i in range(len(ops)):
        res[0, i] = mel(ops[i], A)


    for i in range(nsteps):
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
            for j in range(1):
                h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
                h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nsteps+1)*dt/fs))
        for j in range(1):
            h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
            h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
        h5.create_dataset('maxchi', data=maxchi)
        h5.close()



def run_mctdh_dynamics(ofname='model_A_mlmctdh_30_64.h5'):
    tmax = 200*fs
    dt = 0.25*fs
    nsteps = int(tmax/(dt))+1

    Nmodes = 101
    mdims = [0 for i in range(Nmodes)]
    mdims[0] = 2
    mdims[1] = 70

    for i in range(2, Nmodes):
        mdims[i] = 30
    
    ldims = [0 for i in range(Nmodes)]
    ldims[0] = 2
    ldims[1]= 30
    for i in range(1, Nmodes):
        ldims[i] = 12

    ldims0 = [0 for i in range(Nmodes)]
    ldims0[0] = 2
    for i in range(1, Nmodes):
        ldims0[i] = 4

    #set up the sum of product operator Hamiltonian
    H, opdict = hamiltonian()

    chimax = 64
    chi0 = 4

    sysinf = system_modes(Nmodes)
    sysinf[0] = generic_mode(2)
    for i in range(1, Nmodes):
        sysinf[i] = boson_mode(mdims[i])

    topo = ntree("(1(2)(%d(70))(%d))"%(ldims0[1], chi0))
    capacity = ntree("(1(2)(%d(70))(%d))"%(ldims[1], chimax))
    ntreeBuilder.mlmctdh_subtree(topo()[2], mdims[2:], 2, chi0, ldims0[2:])
    ntreeBuilder.mlmctdh_subtree(capacity()[2], mdims[2:], 2, chimax, ldims[2:])

    A = ttn(topo, capacity, dtype=np.complex128)
    state = np.zeros(Nmodes, dtype=int)
    A.set_state(state)

    ops = []
    ops.append(site_operator(sOP("|LE0><LE0|", 0), sysinf, opdict))
    ops.append(site_operator(sOP("|CS0><CS0|", 0), sysinf, opdict))

    mel = matrix_element(A)
    h = sop_operator(H, A, sysinf, opdict)

    sweepA = tdvp(A, h, krylov_dim = 12, expansion='subspace', subspace_neigs=6)
    sweepA.spawning_threshold = 5e-6
    sweepA.minimum_unoccupied=0
    sweepA.dt = dt
    sweepA.coefficient = -1.0j

    res = np.zeros((nsteps+1, len(ops)), dtype=np.complex128)
    maxchi = np.zeros(nsteps+1)

    for i in range(len(ops)):
        res[0, i] = mel(ops[i], A)


    for i in range(nsteps):
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
            for j in range(1):
                h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
                h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nsteps+1)*dt/fs))
        for j in range(1):
            h5.create_dataset("|LE%d><LE%d|"%(j, j), data=res[:, 2*j])
            h5.create_dataset("|CS%d><CS%d|"%(j, j), data=res[:, 2*j+1])
        h5.create_dataset('maxchi', data=maxchi)
        h5.close()

run_mctdh_dynamics()
