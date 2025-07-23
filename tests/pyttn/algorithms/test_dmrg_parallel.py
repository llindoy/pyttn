import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

from pyttn import (
    SOP,
    dmrg,
    matrix_element,
    ms_SOP,
    ms_sop_operator,
    ms_ttn,
    ntreeBuilder,
    sOP,
    sop_operator,
    system_modes,
    tls_mode,
    ttn,
)


@pytest.mark.parametrize(
    "nthreads",
    [
        1, 2, 4, 8,
    ],
)
def test_dmrg_mps_tfim_parallel(nthreads):
    """Tests the DMRG algorithm on the transverse field Ising model at its critical point.

    Here we optionally allow for the use of an MPS wavefunction or balanced binary tree
    representation wavefunction
    """
    N=16
    expected_result = -1.2510242438
    adaptive = False
    use_mps = True
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
    elif use_mps:
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
            num_threads=nthreads,
        )
        sweep.spawning_threshold = 1e-6
        sweep.minimum_unoccupied = 0
    else:
        sweep = dmrg(A, h, krylov_dim=12)

    for i in range(nsteps):
        sweep(A, h)
        res = np.real(mel(h, A)) / N
        print(i, res)

    res = np.real(mel(h, A)) / N

    assert pytest.approx(res, 1e-6) == expected_result




@pytest.mark.parametrize(
    "nthreads, nset_threads",
    [
        (1, 1),
        #(1, 2),
        #(1, 4),
        #(2, 1),
        #(2, 2),
        #(2, 4),
        #(4, 1),
        #(4, 2),
        #(4, 4),
    ],
)
def test_dmrg_ms_mps_tfim_parallel(nthreads, nset_threads):
    expected_result = -1.2510242438

    nsteps = 10
    N = 16
    chi = 16
    J = 1.0
    h = 1.0

    H = ms_SOP(2, N - 1)

    # add on the Hamiltonian terms including the first two sites
    H[0, 1] += -1.0 * h
    H[1, 0] += -1.0 * h

    H[0, 0] += -1.0 * J * sOP("sz", 0)
    H[1, 1] += 1.0 * J * sOP("sz", 0)

    # add on the Hamiltonian terms acting on the remainder of the chain
    # add on the onsite transversal fields
    for i in range(N - 1):
        H[0, 0] += -1.0 * h * sOP("sx", i)
        H[1, 1] += -1.0 * h * sOP("sx", i)

    # now add on the zz interactions
    for i in range(N - 2):
        H[0, 0] += -1.0 * J * sOP("sz", i) * sOP("sz", i + 1)
        H[1, 1] += -1.0 * J * sOP("sz", i) * sOP("sz", i + 1)

    # set up the system object
    sysinf = system_modes(N-1)
    for i in range(N-1):
        sysinf[i] = tls_mode()

    # setup the system topology
    A = None
    dims = [2 for i in range(N - 1)]
    topo = ntreeBuilder.mps_tree(dims, chi)
    A = ms_ttn(2, topo, dtype=np.complex128)

    # now set up the wavefunction
    A.random()

    # set up the sop operator
    h = ms_sop_operator(H, A, sysinf)
    sweep = dmrg(A, h, krylov_dim=12, num_threads=nthreads, set_var_num_threads=nset_threads)

    for _ in range(nsteps):
        sweep(A, h)

    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(h, A)) / N

    assert pytest.approx(res, 1e-6) == expected_result
