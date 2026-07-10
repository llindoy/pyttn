# This files is part of the pyTTN package.
#(C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
from itertools import product

import h5py
import numpy as np

import pyttn
from pyttn import boson_mode, nlevel_mode, ntree, operator_dictionary, site_operator, sop_operator, ttn
from pyttn import NodeIncrementSetter, set_topology_properties
from pyttn import ntreeBuilder as ntB
from pyttn.simulation import Observable, TDVPSimulation
from pyttn.ttns.sop import OperatorBuilder


def vc_hamiltonian(ls, ws, alps, bs):
    """Build the vibronic-coupling Hamiltonian using the labelled OperatorBuilder interface.

    :param ls: electronic-state onsite energies, shape (N, N)
    :param ws: vibrational mode frequencies, shape (M,)
    :param alps: linear vibronic couplings, shape (N, N, M)
    :param bs: quadratic vibronic couplings, shape (N, N, M, M)
    :return: labelled (symbolic) Hamiltonian
    :rtype: lSOP
    """
    N, M = alps.shape[0], ws.shape[0]
    b = OperatorBuilder()

    # NOTE: the |i><j| electronic-state projectors are custom matrix operators, not
    # built-in sOP labels. As in pyrazine_hamiltonian.py, the labelled interface has no
    # end-to-end path for registering custom-matrix operators under a site label, so we
    # keep these as raw sOP calls via b.op (placeholder index still tracked/relabelled
    # normally) and register the matching raw operator_dictionary entries once a
    # physical site_map is known (see electronic_operator_dictionary below).
    H = None
    for i, j in product(range(N), range(N)):
        if np.abs(ls[i, j]) > 1e-12:
            term = ls[i, j] * b.op("|%d><%d|" % (i, j), "el")
            H = term if H is None else H + term

    for k in range(M):
        if np.abs(ws[k]) > 1e-12:
            term = ws[k] * b.op("n", "v%d" % k)
            H = term if H is None else H + term

    for i, j, k in product(range(N), range(N), range(M)):
        if np.abs(alps[i, j, k]) > 1e-12:
            term = alps[i, j, k] * b.op("|%d><%d|" % (i, j), "el") * b.op("q", "v%d" % k)
            H = term if H is None else H + term

    for i, j, k, kp in product(range(N), range(N), range(M), range(M)):
        if np.abs(bs[i, j, k, kp]) > 1e-12:
            term = bs[i, j, k, kp] * b.op("|%d><%d|" % (i, j), "el") * b.op("q", "v%d" % k) * b.op("q", "v%d" % kp)
            H = term if H is None else H + term

    return b.wrap(H)


def electronic_operator_dictionary(N, site_map, nmodes):
    """Build the raw operator_dictionary containing the custom |i><j| electronic-mode
    projectors, keyed to the physical mode index of the "el" label (see the note in
    vc_hamiltonian above for why this stays on the old integer-indexed API)."""
    el = site_map["el"]
    opdict = operator_dictionary(nmodes)
    for i, j in product(range(N), range(N)):
        v = np.zeros((N, N), dtype=np.complex128)
        v[i, j] = 1.0
        opdict.insert(el, "|%d><%d|" % (i, j), site_operator(v, optype="matrix", mode=0))
    return opdict


def vibronic_dynamics(ls, ws, alps, bs):
    N, M = alps.shape[0], ws.shape[0]
    d = 16  # vibrational mode Hilbert space dimension
    dims = [d for _ in range(M)]
    Nspf = 16  # number of single particle functions
    degree = 2  # degree of tree (2=binary)

    H = vc_hamiltonian(ls, ws, alps, bs)
    site_map = {"el": 0, **{"v%d" % k: k + 1 for k in range(M)}}
    Hsop = H.compile(site_map, M + 1)
    opdict = electronic_operator_dictionary(N, site_map, M + 1)

    # set up tree containing root and electron dof, then attach a binary tree handling the vibrational modes
    topo, capacity = ntree("(1(%d(%d)))" % (N, N)), ntree("(1(%d(%d)))" % (N, N))
    ntB.mlmctdh_subtree(topo(), dims, degree, 4)
    ntB.mlmctdh_subtree(capacity(), dims, degree, Nspf)
    set_topology_properties(capacity, NodeIncrementSetter(4, maxchi=Nspf), [N] + dims, chi_local_transform=[N] + dims)

    sysinf = pyttn.system_modes(M + 1)
    sysinf[0] = nlevel_mode(N)
    for k in range(M):
        sysinf[k + 1] = boson_mode(d)
    psi = ttn(topo, capacity)
    psi.set_state([0] + [0 for _ in range(M)])

    Hop = sop_operator(Hsop, psi, sysinf, opdict)

    # one observable per |i><j| electronic-state projector, matching the original N*N ops list
    ops = [Observable("|%d><%d|" % (i, j), op=site_operator(pyttn.sOP("|%d><%d|" % (i, j), 0), sysinf, opdict)) for i in range(N) for j in range(N)]

    dt, nsteps = 0.1, 1000
    sim = TDVPSimulation(psi, Hop, dt=dt, nstep=nsteps, coefficient=-1.0j, observables=ops, expansion="subspace", integrator_kwargs={"krylov_dim": 8, "subspace_neigs": 4, "subspace_krylov_dim": 8}, output_file="res.h5", output_stride=10)
    sim.integrator.spawning_threshold, sim.integrator.unoccupied_threshold, sim.integrator.minimum_unoccupied = 1e-5, 1e-5, 1
    results = sim.run()

    # the original script writes a single 2D "res" dataset of shape (N*N, nsteps+1); reshape the per-observable columns from ResultsBuffer to match that layout
    res = np.stack([results.data[obs.label] for obs in ops])
    h5 = h5py.File("res.h5", "w")
    h5.create_dataset("t", data=results.t)
    h5.create_dataset("res", data=res)
    h5.close()


if __name__ == "__main__":
    # run vibronic coupling dynamics for the anthracene/C60 complex model presented
    # in J. Chem. Phys. 142, 084706 (2015)
    with open(os.path.join(os.path.dirname(__file__), "params.pkl"), "rb") as filehandler:
        omegas, couplings = pickle.load(filehandler)
    omegas = np.array(omegas)
    lambdas, alphas = couplings[0], couplings[1]
    N, M = len(lambdas), len(omegas)
    betas = np.zeros((N, N, M, M))

    vibronic_dynamics(lambdas, omegas, alphas, betas)
