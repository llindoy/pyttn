import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest
import pickle
import copy

from pyttn import (
    SOP,
    dmrg,
    matrix_element,
    ntreeBuilder,
    sOP,
    sop_operator,
    system_modes,
    tls_mode,
    ttn,
)


@pytest.mark.parametrize(
    "N, adaptive, use_mps",
    [
        (16, False, True),
        (20, False, True),
        (32, False, True),
        (16, True, True),
        (20, True, True),
        (32, True, True),
        (16, False, False),
        (20, False, False),
        (16, True, False),
        (20, True, False),
    ],
)
def test_dmrg_mps_tfim(N, adaptive, use_mps):
    """Tests the DMRG algorithm on the transverse field Ising model at its critical point.

    Here we optionally allow for the use of an MPS wavefunction or balanced binary tree
    representation wavefunction
    """
    J = 1.0
    h = 1.0
    chi = 16
    chi0 = 4
    nsteps = 10

    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    # set up the system Hamiltonian
    H = SOP(N)

    # add on the onsite transversal fields
    for i in range(N):
        H += -1.0 * h * sOP("sx", i)

    # now add on the zz interactions
    for i in range(N - 1):
        H += -1.0 * J * sOP("sz", i) * sOP("sz", i + 1)

    # setup the system topology
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

    # now set up the wavefunction
    A.random()

    # set up the matrix element object
    mel = matrix_element(A, nbuffers=1)

    # set up the sop operator
    h = sop_operator(H, A, sysinf)

    sweep = None
    if adaptive:
        sweep = dmrg(
            A,
            h,
            krylov_dim=12,
            expansion="subspace",
            subspace_krylov_dim=12,
            subspace_neigs=6,
        )
        sweep.spawning_threshold = 1e-6
        sweep.minimum_unoccupied = 0
    else:
        sweep = dmrg(A, h, krylov_dim=12)

    sweep(A, h)
    res = np.real(mel(h, A)) / N

    B = copy.deepcopy(A)
    pickled = pickle.dumps(sweep)
    sweep2 = pickle.loads(pickled)


    sweep(A, h)
    sweep2(B, h)

    res = np.real(mel(h, A)) / N
    resB = np.real(mel(h, B)) / N

    assert pytest.approx(res, 1e-8) == resB