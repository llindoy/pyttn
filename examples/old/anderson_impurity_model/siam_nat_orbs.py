import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *

from siam_core import *

def extract_matrix_block(M, inds):
    N = len(inds)
    
    Mt = np.zeros((N, N), dtype=M.dtype)
    for ci, i in enumerate(inds):
        for cj, j in enumerate(inds):
            Mt[ci, cj] = M[i, j]

    return Mt

def set_matrix_block(Mt, inds, M):
    N = len(inds)
    
    for ci, i in enumerate(inds):
        for cj, j in enumerate(inds):
            M[i, j] = Mt[ci, cj]

    return M


def siam_test(Nb, Gamma,  W, ek, U, chi, dt, nstep = 1, degree = 2, yrange=[0, 2], ndmrg=4):
    V, e = siam_star(Nb, Gamma, W)

    print(U)
    N = Nb+1
    H = hamiltonian(ek, U, V, e, geom='star')

    bath_dims = [2 for i in range(Nb)]

    #set up the system information
    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = fermion_mode()

    H.jordan_wigner(sysinf)

    rho = compute_rdm(H, bath_dims, degree, chi, ndmrg)

    #now construct the bath 


    #construct the tree topology used for these calculations.  Here we are using a binary tree to partition the up and down spin sectors
    #with the impurity orbitals taking the first site on each subtree
    topo = build_tree(bath_dims, degree, chi)

    A = ttn(topo, dtype=np.complex128)
    A.random()

    h = sop_operator(H, A, sysinf, compress=True)

    mel = matrix_element(A)

    #prepare the ground state of the non-interacting Hamiltonian
    dmrg_sweep = dmrg(A, h0, krylov_dim = 4)
    dmrg_sweep.restarts = 1
    dmrg_sweep.prepare_environment(A, h0)

    for i in range(ndmrg):
        dmrg_sweep.step(A, h0)
        print(i, dmrg_sweep.E())


    #set up observables
    ops = []
    for i in range(2*N):
        ops.append(site_operator_complex(sOP("n", i), sysinf))


    #set up the tdvp engine
    sweep = tdvp(A, h, krylov_dim = 8)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)

    res = np.zeros((2*N, nstep+1))

    #set up interactive plotting
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    num = fig.number
    im = ax[0].imshow(res[:N, :] + res[-1:(N-1):-1, :], vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    l1 = ax[1].plot(np.arange(nstep+1)*dt, res[N-1, :]+res[N,:])[0]
    ax[1].set(xlim=[0, dt*nstep])
    ax[1].set(ylim=yrange)
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$n_u(t) + n_d(t)$")
    ax[1].set_ylabel(r"$n(t)$")

    for j in range(2*N):
        res[j, 0] = np.real(mel(ops[j], A, A))
    #do the time evolution
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        for j in range(2*N):
            res[j, i+1] = np.real(mel(ops[j], A, A))
        print((i+1)*dt, t2-t1)
        if(plt.fignum_exists(num)):
            plt.gcf().canvas.draw()
            im.set_data(res[:N, :] + res[-1:(N-1):-1, :])
            l1.set_data(np.arange(nstep+1)*dt, res[N-1, :]+res[N, :])
            plt.pause(0.1)

    plt.ioff()
    plt.show()

siam_test(32, 1.0, 10, -1.25*np.pi, 2.5*np.pi, 16, 0.01, nstep = 1000, ndmrg = 20)

