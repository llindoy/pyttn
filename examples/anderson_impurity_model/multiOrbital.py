import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *

from siam_core import siam_star

def build_tree(bath_dims, chiU, chii, chib, degree=2):
    topo = ntree("(1(%d(%d(2(2)))(%d(2(2)))(%d(2(2))))(%d(%d(2(2)))(%d(2(2)))(%d(2(2)))))" % (chiU, chii, chii, chii, chiU, chii, chii, chii))
    if(degree > 1):
        #spin up orbitals
        ntreeBuilder.mlmctdh_subtree(topo()[0][0], bath_dims, degree, chib)
        ntreeBuilder.mlmctdh_subtree(topo()[0][1], bath_dims, degree, chib)
        ntreeBuilder.mlmctdh_subtree(topo()[0][2], bath_dims, degree, chib)

        #spin down orbitals
        ntreeBuilder.mlmctdh_subtree(topo()[1][0], bath_dims, degree, chib)
        ntreeBuilder.mlmctdh_subtree(topo()[1][1], bath_dims, degree, chib)
        ntreeBuilder.mlmctdh_subtree(topo()[1][2], bath_dims, degree, chib)

    else:
        raise RuntimeError("MPS subtree currently not working.")
    ntreeBuilder.sanitise(topo)
    return topo

def inds(i, s, imp, N):
    return i + imp * N + 3*N*s

def hamiltonian(U, J, eps, Vs, es, density_density = False):
    Nb = len(Vs[0])
    N = Nb+1
    Nt = 6*N

    H = SOP(Nt)

    #first add on the local impurity terms
    for s in range(2):
        for m in range(3):
            H += fermion_operator("n", inds(0, s, m, N));
    
    #add on the density densit contributions
    for m in range(3):
        H += U*fermion_operator("n", inds(0, 0, m, N))*fermion_operator("n", inds(0, 1, m, N))
        for s in range(2):
            sp = (s+1)%2
            for mp in range(m+1, 3):
                H += (U-2*J)*fermion_operator("n", inds(0, s, m, N))*fermion_operator("n", inds(0, sp, mp, N))
                H += (U-3*J)*fermion_operator("n", inds(0, s, m, N))*fermion_operator("n", inds(0, s, mp, N))

    #optional add on the spin flip terms
    if not density_density:
        for m in range(3):
            for mp in range(m+1,3):
                H += J*fermion_operator("cdag", inds(0, 0, m, N))*fermion_operator("c", inds(0, 1, m, N))*fermion_operator("c", inds(0, 0, mp, N))*fermion_operator("cdag", inds(0, 1, mp, N))
                H += J*fermion_operator("c", inds(0, 1, mp, N))*fermion_operator("cdag", inds(0, 0, mp, N))*fermion_operator("cdag", inds(0, 1, m, N))*fermion_operator("c", inds(0, 0, m, N))
                H -= J*fermion_operator("cdag", inds(0, 0, m, N))*fermion_operator("cdag", inds(0, 1, m, N))*fermion_operator("c", inds(0, 0, mp, N))*fermion_operator("c", inds(0, 1, mp, N))
                H -= J*fermion_operator("c", inds(0, 1, mp, N))*fermion_operator("cdag", inds(0, 0, mp, N))*fermion_operator("c", inds(0, 1, m, N))*fermion_operator("c", inds(0, 0, m, N))

    #add on the bath terms
    for s in range(2):
        for m in range(3):
            for l in range(Nb):
                H+=es[m][l]*fermion_operator("n", inds(l+1, s, m, N))
                H+=Vs[m][l]*fermion_operator("cdag", inds(0, s, m, N))*fermion_operator("c", inds(l+1, s, m, N))
                H+=Vs[m][l]*fermion_operator("cdag", inds(l+1, s, m, N))*fermion_operator("c", inds(0, s, m, N))

    return H

def multiorbital_test(U, J, eps, Gamma, W, Nb, chiU, chii, chib, dt, nstep = 1, degree = 2, yrange=[0, 2], ndmrg=4, density_density=False):
    V, e = siam_star(Nb, Gamma, W)
    Vs = [V, V, V]
    es = [e, e, e]
    N = Nb+1
    Nt = 6*N

    sysinf = system_modes(Nt)
    for i in range(Nt):
        sysinf[i] = fermion_mode()

    H =hamiltonian(U, J, eps, Vs, es, density_density=density_density)
    H.jordan_wigner(sysinf)

    bath_dims = [2 for i in range(Nb)]

    topo = build_tree(bath_dims, chiU, chii, chib, degree=degree)
    print(topo)

    A = ttn(topo, dtype=np.complex128)
    A.random()

    h = sop_operator(H, A, sysinf, compress=True)
    mel = matrix_element(A)

    dmrg_sweep = dmrg(A, h, krylov_dim = 4)
    dmrg_sweep.restarts = 1
    dmrg_sweep.prepare_environment(A, h)

    for i in range(ndmrg):
        dmrg_sweep.step(A, h)
        print(i, dmrg_sweep.E())

    h.Eshift = dmrg_sweep.E()

    jw = site_operator(np.array([[1, 0], [0, -1]], dtype=np.complex128), mode=0, optype="matrix")
    cdag = site_operator(np.array([[0, 1], [0, 0]], dtype=np.complex128), mode=0, optype="matrix")
    c = site_operator(np.array([[0, 0], [1, 0]], dtype=np.complex128), mode=0, optype="matrix")

    B = copy.deepcopy(A)
    A.apply_one_body_operator(cdag)

    res = mel(c, B, A)
    print(0, np.real(res), np.imag(res))
    tdvpA = tdvp(A, h, krylov_dim =8)
    tdvpA.dt = dt
    tdvpA.prepare_environment(A, h)
    tdvpB = tdvp(B, h, krylov_dim =8)
    tdvpB.dt = dt
    tdvpB.prepare_environment(A, h)
    for i in range(nstep):
        tdvpA.step(A, h)
        tdvpB.step(B, h)
        res = mel(c, A, B)
        print((i+1)*dt, np.real(res), np.imag(res))

U = 2.0
J = U/6
eps = 0
D=1
Gamma = 2/D
Nb = 19
multiorbital_test(U, J, eps, Gamma, D, Nb, 8, 8, 8, 0.001, ndmrg=5, nstep=100)
