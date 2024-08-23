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

def pyrazine_test(N1, N2, N3, N4, N5, Nb, Nprim, nstep, dt, ofname='pyrazine.h5'):

    N = 24
    #set up the vibrational basis set sizes
    m = [Nprim for i in range(24)]

    #build topology and capacity trees
    topo= build_topology_from_string_multiset(N1, N2, N3, N4, N5, Nb, Nprim)
    topoB= build_topology_from_string_multiset(1,1,1,1,1,1, Nprim)

    #set up the sum of product operator Hamiltonian
    H, sysinf = multiset_hamiltonian(m)  
    print(topo)
    print("this one")
    #setup the wavefunction
    A = ms_ttn(topo, 2, dtype=np.complex128)
    coeff = np.zeros(2)
    coeff[1] = 1.0
    state = [[0 for i in range(N)] for j in range(2)]
    A.set_state(coeff, state)
    print("A built")

    B = ms_ttn(topoB, 2, dtype=np.complex128)
    B.set_state(coeff, state)
    print("B built")

    #setup the hierarchical SOP hamiltonian
    h = multiset_sop_operator(H, A, sysinf)
    print("hsop built")

    mel = matrix_element(A)

    #csetup the evolution object
    sweepA = tdvp(A, h, krylov_dim = 12)
    sweepA.dt = dt
    sweepA.coefficient = -1.0j
    print("sweep setup")

    res = np.zeros(nstep+1, dtype=np.complex128)
    maxchi = np.zeros(nstep+1)
    res[0] = mel(B, A)
    print("res computed")

    print(0, np.real(res[0]), np.imag(res[0]), maxchi[0])
    for i in range(nstep):
        t1 = time.time()
        sweepA.step(A, h)
        B = copy.deepcopy(A)
        B.conj()
        t2 = time.time()
        res[i+1] = mel(B, A)

        print((i+1)*(2*dt)/fs, np.real(res[i+1]), np.imag(res[i+1]), np.real(mel(A, A)), t2-t1)
        sys.stdout.flush()

        if(i % 100):
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


from multiprocessing.pool import Pool
import os

def run(x):
    tmax = 150*fs
    dt = 0.05*fs
    nsteps = int(tmax/(2*dt))+1
    os.environ['OMP_NUM_THREADS']='1'
    if x == 8:
        pyrazine_test(8, 8, 8, 5, 5, 8, 60, nsteps, dt, ofname='pyrazine-8.h5')
    elif x == 12:
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
    elif x == 128:
        pyrazine_test(128, 128, 128, 35, 25, 16, 60, nsteps, dt, ofname='pyrazine-128.h5')


if __name__ == '__main__':

    tmax = 600*fs
    dt = 0.05*fs
    nsteps = int(tmax/dt)+1
    #p = Pool(8)
    #inds = [8, 12, 16, 24, 32, 48, 64, 128]
    #p.map(run, inds)
    pyrazine_test(32, 32, 21, 12, 17, 14, 80, nsteps, dt, ofname='pyrazine_ms.h5')
