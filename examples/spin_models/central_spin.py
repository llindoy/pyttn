import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *


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

def csm_test(Nb, B, chi, dt, nstep = 1, degree = 2, yrange=[0, 0.25]):
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

    state = [i%2 for i in range(Nb+1)]
    A.set_state(state)
    #A.set_identity_purification()

    #op = site_operator([1/np.sqrt(2), 1.0j/np.sqrt(2), 1/np.sqrt(2), 1.0j/np.sqrt(2)], optype="diagonal_matrix", mode=0)
    sz_op = site_operator([0.5, -0.5], optype="diagonal_matrix", mode=0)

    h = sop_operator(H, A, sysinf)

    mel = matrix_element(A)

    print(mel(h, A))

    #set up the tdvp engine
    sweep = tdvp(A, h, krylov_dim = 8)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)

    res = np.zeros(nstep+1)

    #set up interactive plotting
    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    l1 = ax.plot(np.arange(nstep+1)*dt, res[:])[0]
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=yrange)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$<S_z(t) S_z(0)>$")

    res[0] = np.real(mel(sz_op, A, A))
    #do the time evolution
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        res[i+1] = np.real(mel(sz_op, A, A))
        print((i+1)*dt, res[i+1], t2-t1)
        if(plt.fignum_exists(num)):
            plt.gcf().canvas.draw()
            l1.set_data(np.arange(nstep+1)*dt, res)
            plt.pause(0.1)

    plt.ioff()
    fig, ax = plt.subplots()
    num = fig.number
    l1 = ax.plot(np.arange(nstep+1)*dt, res[:])[0]
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=yrange)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$<S_z(t) S_z(0)>(t)$")
    plt.show()

csm_test(49, 0.0, 32, 0.05, nstep = 1000, yrange =[-0.5, 0.5])

