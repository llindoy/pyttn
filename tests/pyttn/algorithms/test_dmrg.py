import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import sys
import copy
import h5py

from pyttn import system_modes, tls_mode, SOP, sOP, ntreeBuilder
from pyttn import ttn, sop_operator, matrix_element, dmrg

import pytest

@pytest.mark.parametrize("N, expected_result, adaptive, use_mps", [   
        (16, -1.2510242438, False, True), (20, -1.255389856, False, True), (32, -1.2620097863, False, True),
        (16, -1.2510242438, True, True), (20, -1.255389856, True, True), (32, -1.2620097863, True, True),
        (16, -1.2510242438, False, False), (20, -1.255389856, False, False),
        (16, -1.2510242438, True, False), (20, -1.255389856, True, False)])
def test_dmrg_mps_tfim(N, expected_result, adaptive, use_mps):
    """Tests the DMRG algorithm on the transverse field Ising model at its critical point.

    Here we optionally allow for the use of an MPS wavefunction or balanced binary tree
    representation wavefunction
    """
    J = 1.0
    h = 1.0
    chi = 16
    chi0 = 4
    nsteps = 10

    #set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    #set up the system Hamiltonian
    H = SOP(N)

    #add on the onsite transversal fields
    for i in range(N):
        H += -1.0*h*sOP("sx", i)

    #now add on the zz interactions
    for i in range(N-1):
        H += -1.0*h*sOP("sz", i)*sOP("sz", i+1)

    #setup the system topology
    A = None
    dims = [2 for i in range(N)]
    if adaptive:
        if use_mps:
            topo = ntreeBuilder.mps_tree(dims, chi0)
            capacity = ntreeBuilder.mps_tree(dims, chi)
            A = ttn(topo, capacity, dtype=np.complex128)
        else:
            topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi0)
            capacity = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
            A = ttn(topo, capacity, dtype=np.complex128)
    else:
        if use_mps:
            topo = ntreeBuilder.mps_tree(dims, chi)
            A = ttn(topo, dtype=np.complex128)
        else:
            topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
            A = ttn(topo, dtype=np.complex128)

    #now set up the wavefunction
    A.random()

    #set up the matrix element object
    mel = matrix_element(A, nbuffers=1)

    #set up the sop operator
    h = sop_operator(H, A, sysinf)

    sweep = None
    if adaptive:
        sweep = dmrg(A, h, krylov_dim = 12, expansion='subspace', subspace_krylov_dim=12, subspace_neigs=6)
        sweep.spawning_threshold = 1e-6
        sweep.minimum_unoccupied=0
    else:
        sweep = dmrg(A, h, krylov_dim = 12)

    for i in range(nsteps):
        sweep(A, h)
        res = np.real(mel(h, A))/N
        print(i, res)

    res = np.real(mel(h, A))/N
    
    assert pytest.approx(res, 1e-6) == expected_result



@pytest.mark.parametrize("N, expected_result, second_chain_interactions, adaptive", [   
        (16, -1.2510242438, False, False), (20, -1.255389856, False, False),
        (16, -1.2510242438, False, True), (20, -1.255389856, False, True),
        (16, -1.2510242438, True, False), (20, -1.255389856, True, False),
        (16, -1.2510242438, True, True), (20, -1.255389856, True, True)])
def test_dmrg_mps_tfim_strided(N, expected_result, second_chain_interactions, adaptive):
    """Tests the DMRG algorithm on the transverse field Ising model at its critical point.  
    Here we additionally double the number of spins in the system but do not include interactions 
    between even and odd indexed spins.  Additionally if the variable second_chain_interactions is
    False then this does not include any Hamiltonian terms on the odd indexed spins
    """
    J = 1.0
    h = 1.0
    chi = 32
    chi0 = 8
    nsteps = 15

    #set up the system object
    sysinf = system_modes(2*N)
    for i in range(2*N):
        sysinf[i] = tls_mode()

    #set up the system Hamiltonian
    H = SOP(2*N)

    #add on the onsite transversal fields
    for i in range(N):
        H += -1.0*h*sOP("sx", 2*i)

    #now add on the zz interactions
    for i in range(N-1):
        H += -1.0*h*sOP("sz", 2*i)*sOP("sz", 2*(i+1))

    if second_chain_interactions:
        #add on the onsite transversal fields
        for i in range(N):
            H += -1.0*h*sOP("sx", 2*i+1)

        #now add on the zz interactions
        for i in range(N-1):
            H += -1.0*h*sOP("sz", 2*i+1)*sOP("sz", 2*(i+1)+1)

    #setup the system topology
    A = None
    if adaptive:
        topo = ntreeBuilder.mps_tree([2 for i in range(2*N)], chi0)
        capacity = ntreeBuilder.mps_tree([2 for i in range(2*N)], chi)
        A = ttn(topo, capacity, dtype=np.complex128)
    else:
        topo = ntreeBuilder.mps_tree([2 for i in range(2*N)], chi)
        A = ttn(topo, dtype=np.complex128)

    #now set up the wavefunction
    A.random()

    #set up the matrix element object
    mel = matrix_element(A, nbuffers=1)

    #set up the sop operator
    h = sop_operator(H, A, sysinf)

    sweep = None
    if adaptive:
        sweep = dmrg(A, h, krylov_dim = 12, expansion='subspace', subspace_krylov_dim=12, subspace_neigs=6)
        sweep.spawning_threshold = 1e-6
        sweep.minimum_unoccupied=0
    else:
        sweep = dmrg(A, h, krylov_dim = 12)

    for i in range(nsteps):
        sweep(A, h)
        res = np.real(mel(h, A))/N
        print(i, res)

    res = np.real(mel(h, A))/N
    
    if second_chain_interactions:
        assert pytest.approx(res, 1e-6) == 2*expected_result
    else:
        assert pytest.approx(res, 1e-6) == expected_result


@pytest.mark.parametrize("N, expected_result, adaptive", [   
        (16, -1.2510242438, False), (20, -1.255389856, False),
        (16, -1.2510242438, True), (20, -1.255389856, True)])
def test_dmrg_mps_tfim_fork(N, expected_result, adaptive):
    """Tests the DMRG algorithm on the transverse field Ising model at its critical point.  
    Here we use a fork MPS data structure for handling the wavefunction but do not include
    any Hamiltonian terms on the fork.
    """
    J = 1.0
    h = 1.0
    chi = 16
    chiFork = 8
    chi0 = 4
    nsteps = 10
    Nfork = 3

    #set up the system object
    sysinf = system_modes((1+Nfork)*N)
    for i in range((1+Nfork)*N):
        sysinf[i] = tls_mode()

    #set up the system Hamiltonian
    H = SOP((1+Nfork)*N)

    #add on the onsite transversal fields
    for i in range(N):
        H += -1.0*h*sOP("sx", 4*i)

    #add on transverse fields along the fork degrees of freedom
    for i in range(N):
        H += -1.0*sOP("sz", 4*i+1)
        H += -1.0*sOP("sz", 4*i+2)
        H += -1.0*sOP("sz", 4*i+3)

    #now add on the zz interactions
    for i in range(N-1):
        H += -1.0*h*sOP("sz", 4*i)*sOP("sz", 4*(i+1))

    def build_tree(chi, chif):
        topo = ntreeBuilder.mps_tree([chif for i in range(N)], chi)
        inds = topo.leaf_indices()
        for ind in inds:
            topo.at(ind).insert(2)
            ntreeBuilder.mps_subtree(topo.at(ind), [2 for i in range(Nfork)],chif) 
        return topo
    #setup the system topology
    A = None
    if adaptive:
        topo = build_tree(chi0, chi0)
        capacity = build_tree(chi, chiFork)
        A = ttn(topo, capacity, dtype=np.complex128)
    else:
        topo = build_tree(chi, chiFork)
        A = ttn(topo, dtype=np.complex128)

    #now set up the wavefunction
    A.random()

    #set up the matrix element object
    mel = matrix_element(A, nbuffers=1)

    #set up the sop operator
    h = sop_operator(H, A, sysinf)

    sweep = None
    if adaptive:
        sweep = dmrg(A, h, krylov_dim = 12, expansion='subspace', subspace_krylov_dim=12, subspace_neigs=6)
        sweep.spawning_threshold = 1e-6
        sweep.minimum_unoccupied=0
    else:
        sweep = dmrg(A, h, krylov_dim = 12)

    for i in range(nsteps):
        sweep(A, h)
        res = np.real(mel(h, A))/N
        print(i, res)

    res = np.real(mel(h, A))/N
    
    assert pytest.approx(res, 1e-6) == (expected_result-3)

