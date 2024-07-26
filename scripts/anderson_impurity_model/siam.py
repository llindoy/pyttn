import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *


def siam_star(N, Gamma, W):
    ek = W*np.cos(np.pi*np.arange(N)/(N+1.0))
    Vk = np.sqrt(Gamma*W/(N+1.0))*np.sin(np.pi*np.arange(N)/(N+1.0))
    return Vk, ek

def siam_chain(N, Gamma, W):
    em = np.zeros(N)
    t = np.ones(N)*W/2
    t[0] = np.sqrt(Gamma*W/2.0)
    return t, em

def nU(i, N):
    return N-(i+1)
def nD(i, N):
    return N+i

def ind(i, N, s):
    if s == 'u':
        return nU(i, N)
    else:
        return nD(i, N)


def hamiltonian(ek, U, V, e, geom='star'):
    Nb = len(V)
    N = Nb+1
    #set up the Hamiltonian
    H = SOP(2*N)

    #add on the on-site interaction term
    H += U * fermion_operator("n", nU(0, N)) * fermion_operator("n", nD(0, N))

    for s in ['u', 'd']:
        H += ek * fermion_operator("n", ind(0, N, s))

        if(geom == 'star'):
            for i in range(Nb):
                H += V[i]*(fermion_operator("cdag", ind(0, N, s))*fermion_operator("c", ind(i+1, N, s)))
                H += V[i]*(fermion_operator("cdag", ind(i+1, N, s))*fermion_operator("c", ind(0, N, s)))
                H += e[i]*fermion_operator("n", ind(i+1, N, s))
        else:
            for i in range(Nb):
                H += V[i]*(fermion_operator("cdag", ind(i, N, s))*fermion_operator("c", ind(i+1, N, s)))
                H += V[i]*(fermion_operator("cdag", ind(i+1, N, s))*fermion_operator("c", ind(i, N, s)))
                H += e[i]*fermion_operator("n", ind(i+1, N, s))
    return H

def siam_test(Nb, Gamma,  W, ek, U, chi, dt, geom='star', nstep = 1, degree = 2, yrange=[0, 1]):
    V = None
    e = None
    if(geom == 'star'):
        V, e = siam_star(Nb, Gamma, W)
    elif geom == 'chain':
        V, e = siam_chain(Nb, Gamma, W)
    else:
        raise RuntimeError("Index not found.")

    N = Nb+1

    H0 = None
    if(geom == 'star'):
        H0 = hamiltonian(10, 0, V*0.0, e, geom=geom)
    else:
        V0 = copy.deepcopy(V)
        V0[0] = 0
        H0 = hamiltonian(10, 0, V0, e, geom=geom)
    H0.jordan_wigner()

    H = hamiltonian(ek, U, V, e, geom=geom)
    H.jordan_wigner()

    print(H)

    bath_dims = [2 for i in range(Nb)]

    #set up the system information
    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = fermion_mode()


    #construct the tree topology used for these calculations.  Here we are using a binary tree to partition the up and down spin sectors
    #with the impurity orbitals taking the first site on each subtree
    topo = ntree("(1(%d(2(2)))(%d(2(2))))" % (chi, chi))
    
    #then we are adding on trees for the bath degrees of freedom
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[0], bath_dims, degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1], bath_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[0], sbg.mode_dims, degree, chi)
        ntreeBuilder.mps_subtree(topo()[1], sbg.mode_dims, degree, chi)
    ntreeBuilder.sanitise(topo)
    print(topo)

    A = ttn(topo, dtype=np.complex128)
    A.random()

    h0 = sop_operator(H0, A, sysinf)
    h = sop_operator(H, A, sysinf)

    mel = matrix_element(A)

    #set up observables
    ops = []
    for i in range(2*N):
        ops.append(site_operator_complex(sOP("n", i), sysinf))
    
    #prepare the ground state of the non-interacting Hamiltonian
    dmrg_sweep = dmrg(A, h0, krylov_dim = 16)
    dmrg_sweep.restarts = 1
    dmrg_sweep.prepare_environment(A, h0)

    for i in range(5):
        dmrg_sweep.step(A, h0)
        print(i, np.real(dmrg_sweep.E()))

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
    im = ax[0].imshow(res, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    l1 = ax[1].plot(np.arange(nstep+1)*dt, res[N-1, :])[0]
    l2 = ax[1].plot(np.arange(nstep+1)*dt, res[N, :])[0]
    ax[1].set(xlim=[0, dt*nstep])
    ax[1].set(ylim=yrange)
    ax[1].set_xlabel(r"$t$")
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
            im.set_data(res)
            l1.set_data(np.arange(nstep+1)*dt, res[N-1, :])
            l2.set_data(np.arange(nstep+1)*dt, res[N, :])
            plt.pause(0.1)

    plt.ioff()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    num = fig.number
    im = ax[0].imshow(res, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    l1 = ax[1].plot(np.arange(nstep+1)*dt, res[N-1, :])[0]
    l2 = ax[1].plot(np.arange(nstep+1)*dt, res[N, :])[0]
    ax[1].set(xlim=[0, dt*nstep])
    ax[1].set(ylim=yrange)
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$n(t)$")
    plt.show()

siam_test(128, 1.0, 10, -1.25*np.pi, 2.5*np.pi, 32, 0.05, geom='star', nstep = 300)

