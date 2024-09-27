import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
import h5py

sys.path.append("../../")
from pyttn import *
from pyrazine_tree import *
from pyrazine_hamiltonian import *

fs = 41.341374575751

def pyrazine_test(N1, N2, N3, N4, N5, Nb, nstep, dt, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, ofname='pyrazine.h5'):

    N = 25
    #set up the vibrational basis set sizes
    m = [40, 32, 20, 12, 8, 4, 8, 24, 24, 8, 8, 24, 20, 4, 72, 80, 6, 20, 6, 6, 6, 32, 6, 4]

    #set up the system information object
    sysinf = system_modes(N)
    sysinf[0] = generic_mode(2)
    for i in range(N-1):
        sysinf[i+1] = boson_mode(m[i])

    #build topology and capacity trees
    topo = build_topology(4,4,4,4,4,4,m)
    capacity= build_topology(N1, N2, N3, N4, N5, Nb, m)

    #set up the sum of product operator Hamiltonian
    H, opdict = hamiltonian(m)  

    #setup the wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)
    state = np.zeros(N, dtype=int)
    state[0]=1
    A.set_state(state)

    B = ttn(topo, dtype=np.complex128)
    state = np.zeros(N, dtype=int)
    state[0]=1
    B.set_state(state)

    #setup the hierarchical SOP hamiltonian
    h = sop_operator(H, A, sysinf, opdict)

    mel = matrix_element(A)

    #csetup the evolution object
    sweepA = tdvp(A, h, krylov_dim = 12, expansion='subspace')
    sweepA.spawning_threshold = spawning_threshold
    sweepA.unoccupied_threshold=unoccupied_threshold
    sweepA.minimum_unoccupied=nunoccupied
    sweepA.dt = dt
    sweepA.coefficient = -1.0j

    res = np.zeros(nstep+1, dtype=np.complex128)
    maxchi = np.zeros(nstep+1)
    res[0] = mel(B, A)
    maxchi[0] = A.maximum_bond_dimension()

    print(0, np.real(res[0]), np.imag(res[0]), maxchi[0])
    for i in range(nstep):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        B = copy.deepcopy(A)
        B.conj()
        res[i+1] = mel(B, A)
        maxchi[i+1] = A.maximum_bond_dimension()

        if(i % 10 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt*2/fs))
            h5.create_dataset('Sz', data=res)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt*2/fs))
    h5.create_dataset('Sz', data=res)
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()

tmax = 150*fs
dt = 0.05*fs
nsteps = int(tmax/(2*dt))+1


from multiprocessing.pool import Pool
import os

def run(x):
    os.environ['OMP_NUM_THREADS']='1'
    if x == 12:
        pyrazine_test(12, 12, 12, 8, 8, 8, 60, nsteps, dt, ofname='pyrazine-12.h5')
    elif x == 16:
        pyrazine_test(16, 16, 16, 10, 10, 8, 60, nsteps, dt, ofname='pyrazine-16.h5')
    elif x == 24:
        pyrazine_test(24, 24, 24, 16, 16, 16, 60, nsteps, dt, ofname='pyrazine-24.h5')
    elif x == 32:
        pyrazine_test(32, 32, 32, 20, 20, 16, 60, nsteps, dt, ofname='pyrazine-32.h5')
    elif x == 48:
        pyrazine_test(48, 48, 48, 24, 24, 16, 60, nsteps, dt, ofname='pyrazine-48.h5')
    elif x == 64:
        pyrazine_test(64, 64, 64, 30, 30, 16, 60, nsteps, dt, ofname='pyrazine-64.h5')
    elif x == 80:
        pyrazine_test(80, 80, 80, 30, 30, 16, 60, nsteps, dt, ofname='pyrazine-80.h5')
    elif x == 128:
        pyrazine_test(128, 128, 128, 30, 30, 16, 60, nsteps, dt, ofname='pyrazine-128.h5')

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS']='1'
    pyrazine_test(24, 24, 24, 12, 12, 12, nsteps, dt, ofname='pyrazine-16.h5')
    #p = Pool(8)
    #inds = [12, 16, 24, 32, 48, 64, 80, 128]
    #p.map(run, inds)
