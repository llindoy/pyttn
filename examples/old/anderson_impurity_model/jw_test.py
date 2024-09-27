import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *

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

def siam_star(N, Gamma, W):
    ek = W*np.cos(np.pi*(np.arange(N)+1)/(N+1))
    Vk = np.sqrt(Gamma*W/(N+1.0))*np.sin(np.pi*(np.arange(N)+1)/(N+1))
    return Vk, ek

def nU(i, N):
    return N-(i+1)
def nD(i, N):
    return N+i

def ind(i, N, s):
    if s == 'u':
        return nU(i, N)
    else:
        return nD(i, N)


def hamiltonian(ek, U, V, e):
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


def compute_rdm(H, bath_dims, degree, chi, ndmrg):
    Nb = len(bath_dims)
    N = Nb+1

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

    h = sop_operator(H, A, sysinf)

    mel = matrix_element(A)

    #prepare the ground state of the non-interacting Hamiltonian
    dmrg_sweep = dmrg(A, h, krylov_dim = 4)
    dmrg_sweep.restarts = 1
    dmrg_sweep.prepare_environment(A, h)

    for i in range(ndmrg):
        dmrg_sweep.step(A, h)
        print(i, dmrg_sweep.E())

    #now compute the 1rdm from the 
    rho = np.zeros((2*N, 2*N), dtype=np.complex128)
    for i in range(2*N):
        for j in range(2*N):
            if(i == j):
                op = site_operator_complex(sOP("n", i), sysinf)
                rho[i, i] = mel(op, A, A)
            else:

                adaga = SOP(2*N)
                adaga += 1.0*fermion_operator("cdag", i)*fermion_operator("c", j)
                print(adaga)
                adaga.jordan_wigner()
                print(adaga)
    exit()
def siam_test(Nb, Gamma,  W, ek, U, chi, dt, nstep = 1, degree = 2, yrange=[0, 2], ndmrg=4):
    V = None
    e = None
    V, e = siam_star(Nb, Gamma, W)

    N = Nb+1
    H = hamiltonian(ek, U, V, e)
    H.jordan_wigner()

    bath_dims = [2 for i in range(Nb)]

    #set up the system information
    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = fermion_mode()

    compute_rdm(H, bath_dims, degree, chi, ndmrg)
    return

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

siam_test(4, 1.0, 10, -1.25*np.pi, 0.0, 16, 0.01, nstep = 1000, ndmrg = 20)

