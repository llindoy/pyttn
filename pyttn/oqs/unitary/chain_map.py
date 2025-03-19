# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
import copy


def remove_zeros(M, tol=1e-14):
    M[np.abs(M) < tol] = 0.0
    return M


def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1


def householder_tridiagonalise(M):
    d = M.shape[0]
    U = np.identity(d)
    A = copy.deepcopy(M)
    for i in range(d - 1):
        alpha = -sgn(A[i + 1, i]) * np.linalg.norm(A[i + 1 :, i])
        r = np.sqrt(0.5 * (alpha**2 - A[i + 1, i] * alpha))
        v = np.zeros((d, 1))
        v[i + 1] = (A[i + 1, i] - alpha) / (2 * r)
        for j in range(i + 2, d):
            v[j] = A[j, i] / (2 * r)
        w = np.identity(d) - 2 * v @ v.T
        U = U @ w
        A = w @ A @ w
    return A, U


def gram_schmidt(U):
    nv = U.shape[0]
    V = np.zeros(U.shape, dtype=U.dtype)

    for i in range(0, nv):
        V[:, i] = U[:, i] / np.linalg.norm(U[:, i])
        for k in range(i + 1, nv):
            U[:, k] = U[:, k] - (np.conj(V[:, i]) @ U[:, k]) * V[:, i]

    return V


# transform this into a basis of a collective mode and a set of random modes
def extract_collective(M, tol=1e-14):
    U = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(M.shape[0]), size=M.shape)
    U[:, 0] = 0
    U[0, :] = 0
    U[0, 0] = 1

    # set the first column to map this problem to a collective mode
    U[1:, 1] = M[1:, 0] / np.linalg.norm(M[1:, 0])

    Ui = gram_schmidt(U[1:, 1:])
    U[1:, 1:] = Ui

    t = M @ U
    Ms = np.conj(U).T @ t
    return remove_zeros(Ms, tol=tol), U


# function for mapping a matrix M into the chain form.  Here we assume that the matrix is already in a star format and note that the approach will fail if it is not
def chain_map(g, w, tol=1e-14, return_unitary=False):
    Nb = g.shape[0]
    N = Nb + 1

    M = np.zeros((N, N))

    w2 = np.zeros(N)
    w2[1:] = w
    M[1:, 0] = g
    M[0, 1:] = g
    np.fill_diagonal(M, w2)

    M, U = extract_collective(M, tol=tol)

    # now tridiagonalise the sub-block of M
    U2 = np.zeros(U.shape, U.dtype)
    U2[0, 0] = 1.0
    M[1:, 1:], U2[1:, 1:] = householder_tridiagonalise(M[1:, 1:])

    e = np.diagonal(M)[1:]
    t = np.diagonal(M, offset=1)
    if return_unitary:
        return t, e, (U @ U2)[1:, 1:]
    else:
        return t, e
