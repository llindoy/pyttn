import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *

from siam_core import *

def setup_interactive_plots(nstep, dt, N, res, yrange):
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
    return fig, ax, num, im, l1

def update_interactive_plots(num, im, l1, nstep, dt, N, res):
    if(plt.fignum_exists(num)):
        plt.gcf().canvas.draw()
        im.set_data(res[:N, :] + res[-1:(N-1):-1, :])
        l1.set_data(np.arange(nstep+1)*dt, res[N-1, :]+res[N, :])
        plt.pause(0.1)

def siam_test(Nb, Gamma,  W, ek, U, chi, dt, geom='star', nstep = 1, degree = 2, yrange=[0, 2], ndmrg=4):
    V = None
    e = None
    if(geom == 'star'):
        V, e = siam_star(Nb, Gamma, W)
    elif geom == 'chain':
        V, e = siam_chain(Nb, Gamma, W)
    else:
        raise RuntimeError("Index not found.")

    N = Nb+1
    #set up the system information
    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = fermion_mode()

    H0 = None
    if(geom == 'star'):
        H0 = hamiltonian(10, 0, 0.0*V, e, geom=geom)
    else:
        V0 = copy.deepcopy(V)
        V0[0] = 0
        H0 = hamiltonian(10.0, 0, V0, e, geom=geom)
    H0.jordan_wigner(sysinf)

    H = hamiltonian(ek, U, V, e, geom=geom)
    H.jordan_wigner(sysinf)

    bath_dims = [2 for i in range(Nb)]

    topo = build_tree(bath_dims, degree, chi)

    A = ttn(topo, dtype=np.complex128)
    A.random()

    h0 = sop_operator(H0, A, sysinf, compress=True)
    h = sop_operator(H, A, sysinf, compress=True)

    mel = matrix_element(A)

    #set up observables
    ops = []
    for i in range(2*N):
        ops.append(site_operator_complex(sOP("n", i), sysinf))
    
    #prepare the ground state of the non-interacting Hamiltonian
    dmrg_sweep = dmrg(A, h0, krylov_dim = 4)
    dmrg_sweep.restarts = 1
    dmrg_sweep.prepare_environment(A, h0)

    for i in range(ndmrg):
        dmrg_sweep.step(A, h0)
        print(i, dmrg_sweep.E())

    #set up the tdvp engine
    sweep = tdvp(A, h, krylov_dim = 8)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)

    res = np.zeros((2*N, nstep+1))

    #set up interactive plotting
    fig, ax, num, im, l1 = setup_interactive_plots(nstep, dt, N, res, yrange)

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
        update_interactive_plots(num, im, l1, nstep, dt, N, res)

    plt.ioff()
    plt.show()

siam_test(16, 1.0, 10, -1.25*np.pi, 2.5*np.pi, 16, 0.01, geom='star', nstep = 1000, ndmrg = 20)

