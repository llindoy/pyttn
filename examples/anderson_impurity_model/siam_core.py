import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *

def siam_star(N, Gamma, W):
    ek = W*np.cos(np.pi*(np.arange(N)+1)/(N+1))
    Vk = np.sqrt(Gamma*W/(N+1.0))*np.sin(np.pi*(np.arange(N)+1)/(N+1))
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

def build_tree(bath_dims, degree, chi):
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
    return topo

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

def compute_rdm(H, bath_dims, degree, chi, ndmrg):
    Nb = len(bath_dims)
    N = Nb+1

    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = fermion_mode()

    topo = build_tree(bath_dims, degree, chi)

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
            adaga = SOP(2*N)
            adaga += fermion_operator("cdag", i)*fermion_operator("c", j)
            adaga.jordan_wigner(sysinf)
            ada = sop_operator(adaga, A, sysinf)
            rho[i, j] = mel(ada, A, A)
    
    return rho
