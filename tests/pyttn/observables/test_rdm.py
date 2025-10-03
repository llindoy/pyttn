import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from pyttn import rdm, ttn, ntreeBuilder, matrix_element, Op

import pytest
import random

@pytest.mark.parametrize(
    "N, adaptive, use_mps, ind",
    [
        (16, False, True, 0),
        (20, False, True, 0),
        (32, False, True, 0),
        (16, True, True, 0),
        (20, True, True, 0),
        (32, True, True, 0),
        (16, False, False, 0),
        (20, False, False, 0),
        (16, True, False, 0),
        (20, True, False, 0),
        (16, False, True, 5),
        (20, False, True, 5),
        (32, False, True, 5),
        (16, True, True, 5),
        (20, True, True, 5),
        (32, True, True, 5),
        (16, False, False, 5),
        (20, False, False, 5),
        (16, True, False, 5),
        (20, True, False, 5),
        (16, False, True, 11),
        (20, False, True, 11),
        (32, False, True, 11),
        (16, True, True, 11),
        (20, True, True, 11),
        (32, True, True, 11),
        (16, False, False, 11),
        (20, False, False, 11),
        (16, True, False, 11),
        (20, True, False, 11),
    ],
)
def test_rdm_1_body(N, adaptive, use_mps, ind):
    chi0 = 6
    chi = 16
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
            topo = ntreeBuilder.mps_tree(dims, chi0)
            A = ttn(topo, dtype=np.complex128)
        else:
            topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi0)
            A = ttn(topo, dtype=np.complex128)

    A.set_state([0 for i in range(N)], random_primitive=True)

    rdm_eval = rdm(A)

    B = np.array(rdm_eval(A, ind))
    Bp = np.zeros((2,2))
    Bp[0,0] = 1
    for i in range(2):
        for j in range(2):
            assert pytest.approx(B[i,j], 1e-8) == Bp[i, j]


@pytest.mark.parametrize(
    "N",
    [
        16,
        20,
        32,
    ],
)
def test_rdm_1_body_2(N):
    chi0 = 6
    chi = 16
    A = None

    for test in range(10):
        ind = random.randint(0, N-1)
        dims = [5 for i in range(N)]
        topo = ntreeBuilder.mlmctdh_tree(dims, 2, chi0)
        capacity = ntreeBuilder.mlmctdh_tree(dims, 2, chi)
        A = ttn(topo, capacity, dtype=np.complex128)

        state = [random.randint(0, 4) for i in range(N)]
        A.set_state(state, random_primitive=True)

        rdm_eval = rdm(A)

        B = np.array(rdm_eval(A, ind))
        Bp = np.zeros((5,5))
        Bp[state[ind],state[ind]] = 1
        for i in range(5):
            for j in range(5):
                assert pytest.approx(B[i,j], 1e-8) == Bp[i, j]


@pytest.mark.parametrize(
    "N, order",
    [
        (3,1),
        (16,1),
        (20,1),
        (32,1),
        (3,2),       
        (16,2),
        (20,2),
        (32,2),
        (3,3),
        (16,3),
        (20,3),
        (32,3),
    ],
)
def test_rdm_2_body_product_state(N, order):
    chi0 = 6
    chi = 16
    A = None

    for test in range(10):
        ind = random.randint(0, N-1)
        ind2 = ind
        while ind2 == ind:
            ind2 = random.randint(0, N-1)



        dims = [5 for i in range(N)]
        if order == 1:
            topo = ntreeBuilder.mps_tree(dims, chi0)
            capacity = ntreeBuilder.mps_tree(dims, chi)
            A = ttn(topo, capacity, dtype=np.complex128)
        else:
            topo = ntreeBuilder.mlmctdh_tree(dims, order, chi0)
            capacity = ntreeBuilder.mlmctdh_tree(dims, order, chi)
            A = ttn(topo, capacity, dtype=np.complex128)

        state = [random.randint(0, 4) for i in range(N)]
        A.set_state(state, random_primitive=True)

        rdm_eval = rdm(A)

        B = np.array(rdm_eval(A, ind, ind2))
        Bp = np.zeros((5*5,5*5))

        #now ensure that ind1 is smaller than ind2. This is required for the comparison as 
        #the rdm will be returned with the smaller of the two indices as the slowest moving index.
        if ind2 < ind:
            temp = ind2
            ind2 = ind
            ind = temp

        Bp[state[ind]*5+state[ind2],state[ind]*5+state[ind2]] = 1
        for i in range(Bp.shape[0]):
            for j in range(Bp.shape[1]):
                assert pytest.approx(B[i,j], 1e-8) == Bp[i, j]


@pytest.mark.parametrize(
    "N, order",
    [
        (3,1),
        (16,1),
        (20,1),
        (32,1),
        (3,2),       
        (16,2),
        (20,2),
        (32,2),
        (3,3),
        (16,3),
        (20,3),
        (32,3),
    ],
)
def test_rdm_1_body_GHz(N, order):
    chi0 = 1
    chi = 16

    #set up the GHz state
    dims = [2 for i in range(N)]
    topo = None
    capacity = None
    if order == 1:
        topo = ntreeBuilder.mps_tree(dims, chi0)
        capacity = ntreeBuilder.mps_tree(dims, chi)
    else:
        topo = ntreeBuilder.mlmctdh_tree(dims, order, chi0)
        capacity = ntreeBuilder.mlmctdh_tree(dims, order, chi)
    A = ttn(topo, capacity, dtype=np.complex128)

    A.set_state([0 for i in range(N)])
    cnot_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
    Hmat = 1.0/np.sqrt(2.0)*np.array([[1, 1], [1, -1]], dtype=np.complex128)

    A@=Op(Hmat, [0], [2])
    for i in range(N-1):
        A@=Op(cnot_mat, [i, i+1], [2, 2])

    rho_ref = np.array([[0.5, 0], [0, 0.5]])
    rdm_eval = rdm(A)

    for i in range(N):
        rhoi = np.array(rdm_eval(A, i))
        assert np.allclose(rhoi, rho_ref, 1e-8)

@pytest.mark.parametrize(
    "N, order",
    [
        (3,1),
        (5,1),
        (8,1),
        (3,2),       
        (5,2),
        (8,2),
        (3,3),
        (5,3),
        (8,3),
    ],
)
def test_rdm_2_body_GHz(N, order):
    chi0 = 1
    chi = 16

    #set up the GHz state
    dims = [2 for i in range(N)]
    topo = None
    capacity = None
    if order == 1:
        topo = ntreeBuilder.mps_tree(dims, chi0)
        capacity = ntreeBuilder.mps_tree(dims, chi)
    else:
        topo = ntreeBuilder.mlmctdh_tree(dims, order, chi0)
        capacity = ntreeBuilder.mlmctdh_tree(dims, order, chi)
    A = ttn(topo, capacity, dtype=np.complex128)

    A.set_state([0 for i in range(N)])
    cnot_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
    Hmat = 1.0/np.sqrt(2.0)*np.array([[1, 1], [1, -1]], dtype=np.complex128)

    A@=Op(Hmat, [0], [2])
    for i in range(N-1):
        A@=Op(cnot_mat, [i, i+1], [2, 2])

    rho_ref = np.array([[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]])
    rdm_eval = rdm(A)

    for i in range(N):
        for j in range(i+1, N):
            rhoij = np.array(rdm_eval(A, i, j))
            assert np.allclose(rhoij, rho_ref, 1e-8)

