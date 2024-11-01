import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import h5py
import copy
from itertools import product

sys.path.append("../")
from pyttn import *
from pyttn.utils import visualise_tree

def inds_1(i,s,N) :
    if s == -1 :
        return N-i-1
    else :
        return N+i

def inds(i,s,N):
    zi = i%3 
    Ni = N//3

    iv = zi*Ni+(i-zi)//3
    if s == -1 :
        return iv
    else :
        return N+iv


def inds_2(i,s,N):
    zi = i%3 
    Ni = N//3

    iv = (zi*Ni+(i-zi)//3)*2
    if s == -1 :
        return iv
    else:
        return iv+1

def Hed(U, eimp):
    """
    contruct H from parameters read in param_3
    """
    norb = eimp.shape[0]
    N = norb
    nimp = U.shape[0]
    print(N)

    h = SOP(2*norb)

    spins = [-1,1]
    for i, j,k,l in product(range(nimp), range(nimp),range(nimp), range(nimp)):
        u = U[i,j,k,l]
        for s1,s2 in product(spins,spins) :
            h += u/2.0*fOP("cdag", inds(i,s1,N))*fOP("cdag", inds(j,s2,N))*fOP("c", inds(l,s2,N))*fOP("c", inds(k,s1, N))

    for i, j in product(range(norb), range(norb)):
        for s1 in spins :
            h += eimp[i,j]*fOP("cdag", inds(i, s1, N))*fOP("c", inds(j, s1, N))
    return h


def build_tree(dim, chiS, chiB, degreeS=1, degreeB=1):
    Nspin = 6
    topo = None

    lchi = [chiB for i in range(Nspin)]
    if degreeS == 1:
        topo = ntreeBuilder.mps_tree(lchi, chiS)
    else:
        topo = ntreeBuilder.mlmctdh_tree(lchi, degreeS, chiS, include_local_basis_transformation=False)

    #now we iterate over
    leaf_indices = topo.leaf_indices()
    print(leaf_indices)

    for li in leaf_indices:
        topo.at(li).insert(2)

        if degreeB == 1:
            ntreeBuilder.mps_subtree(topo.at(li), dim, chiB)
        else:
            ntreeBuilder.mlmctdh_subtree(topo.at(li), dim, degreeB, chiB)

    return topo

def multiorbital_test(chiU, chib, degree = 1, yrange=[0, 2], ndmrg=4, density_density=False):
    f = h5py.File('aim_42.h5')
    e = np.array(f["e"])
    U = np.array(f["U"])
    f.close()

    H = Hed(U, e)
    N = e.shape[0]//3
    Nt = 6*N
    sysinf = system_modes(Nt)
    for i in range(Nt):
        sysinf[i] = fermion_mode()

    H.prune_zeros()
    H.jordan_wigner(sysinf)

    bath_dims = [2 for i in range(N-1)]

    topo = ntreeBuilder.mps_tree([2 for i in range(Nt)], chiU)
    capacity = ntreeBuilder.mps_tree([2 for i in range(Nt)], chiU)
    #topo = build_tree(bath_dims, chiU, chib, degreeS=degree)
    #capacity = build_tree(bath_dims, chiU, chib, degreeS=degree)
    #visualise_tree(capacity, add_labels=False)
    #plt.show()
    #exit()
    A = ttn(topo, capacity, dtype=np.complex128)
    A.random()

    h = sop_operator(H, A, sysinf)
    mel = matrix_element(A)

    dmrg_sweep = dmrg(A, h, krylov_dim = 4)#, subspace_neigs=6)

    print(mel(h, A))
    dmrg_sweep.prepare_environment(A, h)
    #dmrg_sweep.spawning_threshold = 1e-4
    #dmrg_sweep.unoccupied_threshold=1e-4
    #dmrg_sweep.minimum_unoccupied=1
    dmrg_sweep.eigensolver_tol=1e-6
    dmrg_sweep.eigensolver_reltol=1e-6
    dmrg_sweep.restarts=1
    #dmrg_sweep.maximum_bond_dimension = 12
    for i in range(5):
        dmrg_sweep.step(A, h)
        print(i, dmrg_sweep.E(), A.maximum_bond_dimension())

    #dmrg_sweep.maximum_bond_dimension = max(chiU, chib)
    for i in range(ndmrg):
        dmrg_sweep.step(A, h)
        print(i, dmrg_sweep.E(), A.maximum_bond_dimension())


multiorbital_test(384, 32, ndmrg=30)
