import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
import scipy.linalg as spla

sys.path.append("../../")
from pyttn import *

def hyperfines_exp(N, N0):
    return np.sqrt((1-np.exp(-2.0/(N0-1)))/(1-np.exp(-2*N/(N0-1))))*np.exp(-np.arange(N)/(N0-1))

def hyperfines(N):
    return np.sqrt(6.0*N/(2.0*N*N+3*N+1)) * (N-np.arange(N))/N

def hamiltonian(B, A):
    Nb = len(A)
    N = Nb+1
    #set up the Hamiltonian
    H = SOP(N)

    H += B*sOP("sz", 0)

    for i in range(Nb):
        H += A[i] * (sOP("sx", 0) * sOP("sx", i+1) )
        H += A[i] * (sOP("sy", 0) * sOP("sy", i+1) )
        H += A[i] * (sOP("sz", 0) * sOP("sz", i+1) )

    return H

#need to make sure this is correctly sampling the spin coherent states
def sample_spin_half_coherent_state():
    x = np.random.normal(size=3)
    r = np.sqrt(x[0]*x[0] + x[1]*x[1]+x[2]*x[2])
    v = x/r
    theta = np.arccos(v[2])
    phi = np.arctan2(v[0], v[1])
    mu = np.exp(1.0j*phi)*np.tan(theta/2.0)

    return np.array([np.sin(theta/2), np.cos(theta/2)*np.exp(-1.0j*phi)])
    

def csm_test(Nb, B, chi, dt, nstep = 1, degree = 2, yrange=[0, 0.25], nsamples=100, sampling_scheme = 'cs'):
    A = hyperfines(Nb)

    N = Nb+1

    H = hamiltonian(B, A)

    bath_dims = [2 for i in range(Nb)]

    #set up the system information
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = spin_mode(2)

    #construct the tree topology used for these calculations.  Here we are using a binary tree to partition the up and down spin sectors
    #with the impurity orbitals taking the first site on each subtree
    topo = ntree("(1(2(2))(2))")
    
    #then we are adding on trees for the bath degrees of freedom
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], bath_dims, degree, chi)
    else:
        raise RuntimeError("mps subtree currently doesn't work correctly.")
    ntreeBuilder.sanitise(topo)
    print(topo)

    A = ttn(topo, dtype=np.complex128)

    #op = site_operator([1/np.sqrt(2), 1.0j/np.sqrt(2), 1/np.sqrt(2), 1.0j/np.sqrt(2)], optype="diagonal_matrix", mode=0)
    sz_op = site_operator([0.5, -0.5], optype="diagonal_matrix", mode=0)

    h = sop_operator(H, A, sysinf)

    res = np.zeros((nsamples, nstep+1))
    mel = matrix_element(A)
    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    l1 = ax.plot(np.arange(nstep+1)*dt, np.mean(res, axis=0), 'k')[0]
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=yrange)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$<S_z(t) S_z(0)>$")

    for sample in range(nsamples):
        if(sampling_scheme == 'cs'):
            state = []
            state.append(np.array([1, 0], dtype=np.complex128))
            for i in range(Nb):
                state.append(sample_spin_half_coherent_state())
            A.set_product(state)
        else:
            dist = [0.5*np.ones(2) for i in range(Nb+1)]
            dist[0][0] = 1
            dist[0][1] = 0
            A.sample_product_state(dist)

        #set up the tdvp engine
        sweep = tdvp(A, h, krylov_dim = 8)
        sweep.dt = dt
        sweep.coefficient = 1.0j
        sweep.prepare_environment(A, h)

        #set up interactive plotting
        l2 = ax.plot(np.arange(nstep+1)*dt,res[sample, :], 'grey')[0]

        res[sample, 0] = np.real(mel(sz_op, A, A))
        #do the time evolution
        for i in range(nstep):
            t1 = time.time()
            sweep.step(A, h)
            t2 = time.time()
            res[sample, i+1] = np.real(mel(sz_op, A, A))
            print((i+1)*dt, res[sample, i+1], np.real(mel(A, A)), t2-t1)
            if(plt.fignum_exists(num)):
                plt.gcf().canvas.draw()
                l1.set_data(np.arange(nstep+1)*dt, np.mean(res[:(sample+1), :], axis=0))
                l2.set_data(np.arange(nstep+1)*dt, res[sample, :])
                plt.pause(0.01)

    plt.ioff()
    plt.show()

csm_test(99, 0.0, 16, 0.05, nstep = 1000, yrange =[-0.5, 0.5])

