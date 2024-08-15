import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *
from chain_map import chain_map

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
        plt.pause(0.01)


#setup the star Hamiltonian for the spin boson model
def setup_star_hamiltonian(ek, U, V, e):
    Nb = len(V)
    N = Nb+1
    #set up the Hamiltonian
    H = SOP(2*N)

    #add on the on-site interaction term
    H += U * fermion_operator("n", nU(0, N)) * fermion_operator("n", nD(0, N))

    for s in ['u', 'd']:
        H += ek * fermion_operator("n", ind(0, N, s))

        for i in range(Nb):
            H += V[i]*(fermion_operator("cdag", ind(0, N, s))*fermion_operator("c", ind(i+1, N, s)))
            H += V[i]*(fermion_operator("cdag", ind(i+1, N, s))*fermion_operator("c", ind(0, N, s)))
            H += e[i]*fermion_operator("n", ind(i+1, N, s))
    return H




#setup the chain hamiltonian for the spin boson model - this is the tedopa method
def setup_chain_hamiltonian(ek, U, _V, _e, include_coupling = True):
    Nb = len(_V)
    N = Nb+1
    #set up the Hamiltonian
    H = SOP(2*N)

    V, e = chain_map(_V, _e)
    V = np.array(V)

    if not include_coupling:
        V[0] = 0.0
    #add on the on-site interaction term
    H += U * fermion_operator("n", nU(0, N)) * fermion_operator("n", nD(0, N))

    for s in ['u', 'd']:
        H += ek * fermion_operator("n", ind(0, N, s))
        for i in range(Nb):
            H += V[i]*(fermion_operator("cdag", ind(i, N, s))*fermion_operator("c", ind(i+1, N, s)))
            H += V[i]*(fermion_operator("cdag", ind(i+1, N, s))*fermion_operator("c", ind(i, N, s)))
            H += e[i]*fermion_operator("n", ind(i+1, N, s))
    return H


#setup the chain hamiltonian for the spin boson model - that is this implements the method described in Nuomin, Beratan, Zhang, Phys. Rev. A 105, 032406
#TO DO: Implement this correctly for fermionic baths.
def setup_ipchain_hamiltonian(ek, _U, _V, _e):
    Nb = len(_V)
    N = Nb+1
    #set up the Hamiltonian
    H = SOP(2*N)

    t, e, P = chain_map(_V, _e, return_unitary = True)
    t0 = t[0]

    l = _e

    #add on the on-site interaction term
    H += _U * fermion_operator("n", nU(0, N)) * fermion_operator("n", nD(0, N))

    class func_class:
        def __init__(self, i, t0, e0, U0, conj = False):
            self.i = i
            self.conj=conj
            self.t0 = t0
            self.e = copy.deepcopy(e0)
            self.U = copy.deepcopy(U0)

        def __call__(self, ti):
            val = self.t0*np.conj(self.U[:, 0])@(np.exp(-1.0j*ti*self.e)*self.U[:, self.i])

            if(self.conj):
                val = np.conj(val)

            return val

    for s in ['u', 'd']:
        H += ek * fermion_operator("n", ind(0, N, s))
        for i in range(Nb):
            H += coeff(func_class(i, t0, l, P, conj=True))*(fermion_operator("cdag", ind(i, N, s))*fermion_operator("c", ind(i+1, N, s)))
            H += coeff(func_class(i, t0, l, P, conj=False ))*(fermion_operator("cdag", ind(i+1, N, s))*fermion_operator("c", ind(i, N, s)))

    return H

def siam_test(Nb, Gamma,  W, ek, U, chi, dt, geom='star', nstep = 1, degree = 2, yrange=[0, 2], ndmrg=4):
    V = None
    e = None
    V, e = siam_star(Nb, Gamma, W)

    N = Nb+1
    #set up the system information
    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = fermion_mode()

    H0 = None
    if(geom == 'star'):
        H0 = setup_star_hamiltonian(10, 0, 0.0*V, e)
    else:
        H0 = setup_chain_hamiltonian(10.0, 0, V, e, include_coupling = False)

    H0.jordan_wigner(sysinf)

    H = None
    if(geom == 'star'):
        H = setup_star_hamiltonian(ek, U, V, e)
    elif (geom=='chain'):
        H = setup_chain_hamiltonian(ek, U, V, e)
    else:
        H = setup_ipchain_hamiltonian(ek, U, V, e)

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

siam_test(16, 1.0, 10, -1.25*np.pi, 2.5*np.pi, 16, 0.01, geom='ipchain', nstep = 1000, ndmrg = 20)

