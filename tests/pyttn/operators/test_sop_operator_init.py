import os
import pytest 

os.environ["OMP_NUM_THREADS"] = "1"

import copy
import numpy as np

from pyttn import (
    SOP,
    ntreeBuilder,
    site_operator,
    multiset_SOP,
    operator_dictionary,
    sOP,
    sop_operator,
    multiset_sop_operator,
    matrix_element,
    system_modes,
    tls_mode,
    ttn,
    ms_ttn
)


def tfim_hamiltonian(N):
    J = 1.0
    h = 1.0

    H = SOP(N)

    # add on the onsite transversal fields
    for i in range(N):
        H += -1.0 * h * sOP("sx", i)

    # now add on the zz interactions
    for i in range(N - 1):
        H += -1.0 * J * sOP("sz", i) * sOP("sz", i + 1)

    return H

def tfim_hamiltonian_ms(N):
    J = 1.0
    h = 1.0

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

def tfim_hamiltonian_opdict(N):
    J = 1.0
    h = 1.0

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


def tfim_hamiltonian_ms_opdict(N):
    J = 1.0
    h = 1.0

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
        ops.insert(i, "SpinZ", site_operator(sz, optype="matrix", mode=i+1))
        ops.insert(i, "SpinX", site_operator(sx, optype="matrix", mode=i+1))

    return H, ops

@pytest.mark.parametrize("N",[4, 8, 16, 32, 11, 15])
def test_sop_operator_initialisation_dictionary(N):
    chi = 6
    dims = [2 for i in range(N)]
    topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
    A = ttn(topo, dtype=np.complex128)
    A.random()
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    H1 = tfim_hamiltonian(N)
    H2, opdict = tfim_hamiltonian_opdict(N)

    h1 = sop_operator(H1, A, sysinf)
    h2 = sop_operator(H2, A, sysinf, opdict)

    mel = matrix_element(A, nbuffers=1)

    e1 = mel(h1, A)
    e2 = mel(h2, A)
    assert e1 == pytest.approx(e2, abs=1e-8)

@pytest.mark.parametrize("N",[4, 8, 16, 32, 11, 15])
def test_ms_sop_operator_initialisation_dictionary(N):
    chi = 6
    dims = [2 for i in range(N-1)]
    topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
    A = ms_ttn(2, topo, dtype=np.complex128)
    A.random()
    sysinf = system_modes(N-1)
    for i in range(N-1):
        sysinf[i] = tls_mode()

    H1 = tfim_hamiltonian_ms(N)
    H2, opdict = tfim_hamiltonian_ms_opdict(N)

    print(H2, opdict)

    h1 = multiset_sop_operator(H1, A, sysinf)
    h2 = multiset_sop_operator(H2, A, sysinf, opdict)

    mel = matrix_element(A, nbuffers=1)

    e1 = mel(h1, A)
    e2 = mel(h2, A)
    assert e1 == pytest.approx(e2, abs=1e-8)