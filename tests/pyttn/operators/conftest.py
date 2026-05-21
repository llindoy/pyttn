import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

from pyttn import (
    SOP,
    dmrg,
    Op,
    ntreeBuilder,
    product_operator,
    site_operator,
    multiset_SOP,
    operator_dictionary,
    sOP,
    sop_operator,
    system_modes,
    tls_mode,
    ttn,
)


def tfim_hamiltonian():
    J = 1.0
    h = 1.0
    N = 16

    H = SOP(N)

    # add on the onsite transversal fields
    for i in range(N):
        H += -1.0 * h * sOP("sx", i)

    # now add on the zz interactions
    for i in range(N - 1):
        H += -1.0 * J * sOP("sz", i) * sOP("sz", i + 1)

    return H


def tfim_hamiltonian_ms():
    J = 1.0
    h = 1.0
    N = 16

    H = multiset_SOP(2, N - 1)

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

    return H

def tfim_hamiltonian_opdict():
    J = 1.0
    h = 1.0
    N = 16

    H = SOP(N)

    # add on the onsite transversal fields
    for i in range(N):
        H += -1.0 * h * sOP("SpinX", i)

    # now add on the zz interactions
    for i in range(N - 1):
        H += -1.0 * J * sOP("SpinZ", i) * sOP("SpinZ", i + 1)

    ops = operator_dictionary(N)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    for i in range(N):
        ops.insert(i, "SpinZ", site_operator(sz, optype="matrix", mode=i))
        ops.insert(i, "SpinX", site_operator(sx, optype="matrix", mode=i))
    return H, ops


def tfim_hamiltonian_ms_opdict():
    J = 1.0
    h = 1.0
    N = 16

    H = multiset_SOP(2, N - 1)

    # add on the Hamiltonian terms including the first two sites
    H[0, 1] += -1.0 * h
    H[1, 0] += -1.0 * h

    H[0, 0] += -1.0 * J * sOP("SpinZ", 0)
    H[1, 1] += 1.0 * J * sOP("SpinZ", 0)

    # add on the Hamiltonian terms acting on the remainder of the chain
    # add on the onsite transversal fields
    for i in range(N - 1):
        H[0, 0] += -1.0 * h * sOP("SpinX", i)
        H[1, 1] += -1.0 * h * sOP("SpinX", i)

    # now add on the zz interactions
    for i in range(N - 2):
        H[0, 0] += -1.0 * J * sOP("SpinZ", i) * sOP("SpinZ", i + 1)
        H[1, 1] += -1.0 * J * sOP("SpinZ", i) * sOP("SpinZ", i + 1)

    ops = operator_dictionary(N-1)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    for i in range(N-1):
        ops.insert(i, "SpinZ", site_operator(sz, optype="matrix", mode=i))
        ops.insert(i, "SpinX", site_operator(sx, optype="matrix", mode=i))

    return H, ops


def tfim_dmrg(A):
    nsteps = 10
    N = A.nmodes()

    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    # set up the system Hamiltonian
    H = tfim_hamiltonian()

    # now set up the wavefunction
    A.random()

    # set up the sop operator
    h = sop_operator(H, A, sysinf)

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

    for _ in range(nsteps):
        sweep(A, h)

    return A


@pytest.fixture
def ttn_rand():
    """Prepares a vacuum state MPS with fixed bond dimension 6 that is also the same as its capacity"""
    N = 16
    chi = 6

    dims = [2 for i in range(N)]
    topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
    A = ttn(topo, dtype=np.complex128)
    A.random()
    return A


@pytest.fixture
def ttn_1():
    """Prepares a vacuum state MPS with fixed bond dimension 6 that is also the same as its capacity"""
    N = 16
    chi = 6

    dims = [2 for i in range(N)]
    topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
    A = ttn(topo, dtype=np.complex128)
    A.set_state([0 for i in range(N)])
    return A


@pytest.fixture
def ttn_2():
    """ """
    N = 16
    chi = 16
    chi0 = 8

    dims = [2 for i in range(N)]
    topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi0)
    capacity = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0])
    A.orthogonalise()
    return A


@pytest.fixture
def ttn_3():
    """ """
    N = 16
    chi = 16
    chi0 = 8

    dims = [2 for i in range(N)]
    topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi0)
    capacity = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0])
    A *= 2.0
    return A


@pytest.fixture
def ttn_4():
    """Prepares the ground state of the TFIM with N=16 sites at the critical point"""
    chi = 16
    chi0 = 4
    N = 16

    # setup the system topology
    A = None
    dims = [2 for i in range(N)]
    topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi0)
    capacity = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
    A = ttn(topo, capacity, dtype=np.complex128)

    return tfim_dmrg(A)


"""
DEFINE THE OPERATORS USED FOR TESTS AS FIXTURES
"""


@pytest.fixture
def Sz0():
    N = 16
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    return site_operator(sOP("sz", 0), sysinf)


@pytest.fixture
def Sz6():
    N = 16
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    return site_operator(sOP("sz", 6), sysinf)


@pytest.fixture
def Sx0():
    N = 16
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    return site_operator(sOP("sx", 0), sysinf)


@pytest.fixture
def Sx6():
    N = 16
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    return site_operator(sOP("sx", 6), sysinf)


@pytest.fixture
def Sz_prod():
    N = 16
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    return product_operator(sOP("sz", 0) * sOP("sz", 6), sysinf)


@pytest.fixture
def Sx_prod():
    N = 16
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    return product_operator(sOP("sx", 0) * sOP("sx", 6), sysinf)


@pytest.fixture
def Sztot():
    N = 16
    op = SOP(N)
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    for i in range(N):
        op += sOP("sz", i)

    return op, sysinf


@pytest.fixture
def Stot():
    N = 16
    op = SOP(N)
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    for i in range(N):
        for j in range(N):
            op += sOP("sz", i) * sOP("sz", j)

    return op, sysinf


@pytest.fixture
def S0tot():
    N = 16
    op = SOP(N)
    # set up the system object
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    op += sOP("sz", 0)

    return op, sysinf

@pytest.fixture
def H():
    J = 1.0
    h = 1.0
    N = 16
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()
    op = SOP(N)

    # add on the onsite transversal fields
    for i in range(N):
        op += -1.0 * h * sOP("sx", i)

    # now add on the zz interactions
    for i in range(N - 1):
        op += -1.0 * J * sOP("sz", i) * sOP("sz", i + 1)

    return op, sysinf

#######

#Op object fixtures

@pytest.fixture
def Sz0_op():
    return Op(np.array([[1.0,0],[0, -1.0]]), [0], [2])


@pytest.fixture
def Sz6_op():
    return Op(np.array([[1.0,0],[0, -1.0]]), [6], [2])


@pytest.fixture
def Sx0_op():
    return Op(np.array([[0,1.0],[1.0, 0]]), [0], [2])


@pytest.fixture
def Sz01_op():
    sz = np.array([[1,0.0],[0.0, -1.0]])
    return Op(np.kron(sz, sz), [0, 1], [2, 2])

#######