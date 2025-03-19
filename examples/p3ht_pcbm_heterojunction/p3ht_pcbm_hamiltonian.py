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
from pyttn import SOP, sOP

# convert from eV to hartree
eV = 0.0367493049512081
meV = eV / 1000

Nfragments = 13
Nf = 8
Not = 8

eps_le = 100 * meV
eps_cs = (
    np.array(
        [0.0, 33.6, 47.4, 56.0, 61.8, 65.7, 68.4, 70.0, 70.9, 71.2, 71.1, 70.5, 69.5]
    )
    * meV
)

J = 100 * meV
t = -120 * meV
L = -200 * meV

wr = 10 * meV
wf = np.array([33.264, 55.892, 79.933, 93.952, 141.11, 177.853, 184.269, 200.025]) * meV
wot = (
    np.array([401.283, 397.773, 182.714, 178.531, 134.550, 111.848, 42.621, 18.316])
    * meV
)

gr_cs1cs1 = 30 / np.sqrt(2.0) * meV
gr_lecs1 = -10 / np.sqrt(2.0) * meV

gf = np.array([9.86, -23.732, -13.629, 28.026, -17.511, -40.280, 65.701, 45.246]) * meV
g_otcs = (
    np.array([7.017, -0.077, -67.849, 57.668, -40.145, 11.68, -10.784, -12.309]) * meV
)
g_otle = (
    np.array([4.035, 2.921, -129.712, 46.885, -32.908, 36.591, -20.211, -7.77]) * meV
)


def ot_mode_index(Nf, n, li):
    if li < 6:
        return n * 6 + li
    else:
        return Nf * 6 + 2 * n + li - 6


def build_operator_dictionary(N, Nfragments):
    from pyttn import site_operator, operator_dictionary

    # set up the system operator dictionary
    opdict = operator_dictionary(N)

    # add on the onsite terms
    for i in range(Nfragments):
        v = np.zeros((2 * Nfragments, 2 * Nfragments), dtype=np.complex128)
        v[2 * i, 2 * i] = 1.0
        op = site_operator(v, optype="matrix", mode=0)
        opdict.insert(0, "|LE%d><LE%d|" % (i, i), op)

        v2 = np.zeros((2 * Nfragments, 2 * Nfragments), dtype=np.complex128)
        v2[2 * i + 1, 2 * i + 1] = 1.0
        op2 = site_operator(v2, optype="matrix", mode=0)
        opdict.insert(0, "|CS%d><CS%d|" % (i, i), op2)

    # add on the hopping terms
    for i in range(Nfragments - 1):
        v = np.zeros((2 * Nfragments, 2 * Nfragments), dtype=np.complex128)
        v[2 * i, 2 * (i + 1)] = 1.0
        v[2 * (i + 1), 2 * i] = 1.0
        op = site_operator(v, optype="matrix", mode=0)
        opdict.insert(0, "LE_hop%d" % (i), op)

        v3 = np.zeros((2 * Nfragments, 2 * Nfragments), dtype=np.complex128)
        v3[2 * i + 1, 2 * (i + 1) + 1] = 1.0
        v3[2 * (i + 1) + 1, 2 * i + 1] = 1.0
        op3 = site_operator(v3, optype="matrix", mode=0)
        opdict.insert(0, "CS_hop%d" % (i), op3)

    v = np.zeros((2 * Nfragments, 2 * Nfragments), dtype=np.complex128)
    v[0, 1] = 1.0
    v[1, 0] = 1.0
    op = site_operator(v, optype="matrix", mode=0)
    opdict.insert(0, "LE_CS_hop", op)

    return opdict


def hamiltonian():

    N = Nfragments * Not + Nf + 1 + 1
    # construct the operator dictionary
    opdict = build_operator_dictionary(N, Nfragments)

    # now build the Hamiltonian
    H = SOP(N)

    # add on the vibrational mode terms
    H += wr * sOP("n", 1)
    for i in range(Nf):
        H += wf[i] * sOP("n", 1 + 1 + i)

    for n in range(Nfragments):
        for li in range(Not):
            H += wot[li] * sOP("n", 1 + 1 + Nf + ot_mode_index(Nfragments, n, li))

    # add on the purely electronic terms
    # the onsite energies
    for i in range(Nfragments):
        H += eps_le * sOP("|LE%d><LE%d|" % (i, i), 0)
        H += eps_cs[i] * sOP("|CS%d><CS%d|" % (i, i), 0)

    # now add on the hopping terms
    for i in range(Nfragments - 1):
        H += J * sOP("LE_hop%d" % (i), 0)
        H += t * sOP("CS_hop%d" % (i), 0)
    H += L * sOP("LE_CS_hop", 0)

    # now add on the vibronic couplings
    # first we do the vibronic couplings to the R mode
    H += gr_cs1cs1 * sOP("|CS0><CS0|", 0) * (sOP("adag", 1) + sOP("a", 1))
    H += gr_lecs1 * sOP("LE_CS_hop", 0) * (sOP("adag", 1) + sOP("a", 1))

    # now we add on the vibronic couplings to the fullerene super-particle modes
    for n in range(Nfragments):
        for li in range(Nf):
            H += ( gf[li] * sOP("|CS%d><CS%d|" % (n, n), 0) * (sOP("adag", 1 + 1 + li) + sOP("a", 1 + 1 + li)))

    # now we add on the oligothiophene modes
    for n in range(Nfragments):
        for li in range(Not):
            H += (
                g_otcs[li]
                * sOP("|CS%d><CS%d|" % (n, n), 0)
                * (
                    sOP("adag", 1 + 1 + Nf + ot_mode_index(Nfragments, n, li))
                    + sOP("a", 1 + 1 + Nf + ot_mode_index(Nfragments, n, li))
                )
            )
            H += (
                g_otle[li]
                * sOP("|LE%d><LE%d|" % (n, n), 0)
                * (
                    sOP("adag", 1 + 1 + Nf + ot_mode_index(Nfragments, n, li))
                    + sOP("a", 1 + 1 + Nf + ot_mode_index(Nfragments, n, li))
                )
            )

    return H, opdict


def multiset_hamiltonian():
    from pyttn import multiset_SOP, sOP

    N = Nfragments * Not + Nf + 1

    # now build the Hamiltonian
    H = multiset_SOP(26, N)

    # add on the vibrational mode terms
    for msind in range(26):
        H[msind, msind] += wr * sOP("n", 0)
        for i in range(Nf):
            H[msind, msind] += wf[i] * sOP("n", 1 + i)

        for n in range(Nfragments):
            for li in range(Not):
                H[msind, msind] += wot[li] * sOP(
                    "n", 1 + Nf + ot_mode_index(Nfragments, n, li)
                )

    # add on the purely electronic terms
    # the onsite energies
    for i in range(Nfragments):
        H[2 * i, 2 * i] += eps_le
        H[2 * i + 1, 2 * i + 1] += eps_cs[i]

    # now add on the hopping terms
    for i in range(Nfragments - 1):
        H[2 * i, 2 * (i + 1)] += J
        H[2 * (i + 1), 2 * i] += J

        H[2 * i + 1, 2 * (i + 1) + 1] += t
        H[2 * (i + 1) + 1, 2 * i + 1] += t

    H[0, 1] += L
    H[1, 0] += L

    # now add on the vibronic couplings
    # first we do the vibronic couplings to the R mode
    H[1, 1] += gr_cs1cs1 * sOP("adag", 0)
    H[1, 1] += gr_cs1cs1 * sOP("a", 0)
    H[0, 1] += gr_lecs1 * sOP("adag", 0)
    H[0, 1] += gr_lecs1 * sOP("a", 0)
    H[1, 0] += gr_lecs1 * sOP("adag", 0)
    H[1, 0] += gr_lecs1 * sOP("a", 0)

    # now we add on the vibronic couplings to the fullerene super-particle modes
    for n in range(Nfragments):
        for li in range(Nf):
            H[2 * n + 1, 2 * n + 1] += gf[li] * sOP("adag", 1 + li)
            H[2 * n + 1, 2 * n + 1] += gf[li] * sOP("a", 1 + li)

    # now we add on the oligothiophene modes
    for n in range(Nfragments):
        for li in range(Not):
            H[2 * n + 1, 2 * n + 1] += g_otcs[li] * sOP(
                "adag", 1 + Nf + ot_mode_index(Nfragments, n, li)
            )
            H[2 * n + 1, 2 * n + 1] += g_otcs[li] * sOP(
                "a", 1 + Nf + ot_mode_index(Nfragments, n, li)
            )
            H[2 * n, 2 * n] += g_otle[li] * sOP(
                "adag", 1 + Nf + ot_mode_index(Nfragments, n, li)
            )
            H[2 * n, 2 * n] += g_otle[li] * sOP(
                "a", 1 + Nf + ot_mode_index(Nfragments, n, li)
            )

    return H
