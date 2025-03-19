import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *
import pyttn

from siam_core import *

def setup_interactive_plots(nstep, dt, N, res, yrange):
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    num = fig.number
    im = ax[0].imshow(res, vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    l1 = ax[1].plot(np.arange(nstep+1)*dt, res[N-1, :]-res[N, :])[0]
    ax[1].set(xlim=[0, dt*nstep])
    ax[1].set(ylim=yrange)
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$n_u(t) + n_d(t)$")
    ax[1].set_ylabel(r"$n(t)$")
    return fig, ax, num, im, l1

def update_interactive_plots(num, im, l1, nstep, dt, N, res):
    if(plt.fignum_exists(num)):
        plt.gcf().canvas.draw()
        #im.set_data(res)
        l1.set_data(np.arange(nstep+1)*dt, res[N-1, :]-res[N, :])
        plt.pause(0.001)

def siam_test(Nb, Gamma, W, ek, U, chi, dt, nstep = 1, degree = 2, yrange=[0, 1]):
    V = None
    e = None

    poly = legendre_polynomial(Nb)
    poly.scale(W) 
    poly.compute_nodes_and_weights()
    e = np.array(poly.nodes())
    w = np.array(poly.weights())

    V = np.sqrt(np.pi*Gamma/W)*np.sqrt(w)

    t = np.linspace(0, 100, 1000)
    res = np.zeros(t.shape, dtype=np.complex128)
    for i in range(Nb):
        res += V[i]*V[i]*np.exp(1.0j*e[i]*t)

    N = Nb+1

    onsite_energy = np.zeros(2*(Nb+1))
    onsite_energy[N-1] = 100
    onsite_energy[N] = -100
    onsite_energy[N+1:] = e
    onsite_energy[:(N-1)] = e[::-1]

    #set up the system information
    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = fermion_mode()


    H = hamiltonian(ek, U, V, e)
    H.jordan_wigner(sysinf)

    bath_dims = [2 for i in range(Nb)]

    topo = build_tree(bath_dims, degree, chi)

    state = np.zeros(2*N, dtype=int)
    for i in range(2*N):
        if(onsite_energy[i] > 0):
            state[i] = 1
    A = ttn(topo, dtype=np.complex128)
    A.set_state(state)

    h = sop_operator(H, A, sysinf, compress=True)

    mel = matrix_element(A)

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
    #fig, ax, num, im, l1 = setup_interactive_plots(nstep, dt, N, res, yrange)

    for j in range(2*N):
        res[j, 0] = np.real(mel(ops[j], A, A))
    #do the time evolution
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        for j in range(2*N):
            res[j, i+1] = np.real(mel(ops[j], A, A))
        print((i+1)*dt, res[N-1, i], res[N, i], t2-t1)
        sys.stdout.flush()
        #update_interactive_plots(num, im, l1, nstep, dt, N, res)

    plt.ioff()
    plt.show()

W = 1
Gamma = 0.5
siam_test(64, Gamma, W, -0.1875*W, 15*W, 24, 0.01, nstep = 10000)

