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
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np
from numba import jit

import pyttn
from pyttn import oqs, utils
from pyttn.ttns.sop import OperatorBuilder, SystemInfo


def sbm_dynamics_timing(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta=None, nstep=1, Nw=10.0, degree=2, compress=True, adaptive=False, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):
    """Set up the spin boson model (star geometry bath) with the labelled OQS interface and time ``nstep`` TDVP steps, returning the mean and standard deviation of the per-step wall-clock time. Mode combination (``nbmax``/``nhilbmax``) is not yet supported by :class:`~pyttn.oqs.MethodBuilder` and has been dropped relative to the original script."""

    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    sysinfo = SystemInfo()
    sysinfo["spin"] = pyttn.tls_mode()
    b = OperatorBuilder()
    model = oqs.OQSModel(system_info=sysinfo, system_generator=b.wrap(eps * b.op("sz", "spin") + delta * b.op("sx", "spin")))

    bath = oqs.BosonicBath(J, beta=beta)
    coupling = OperatorBuilder()
    chi0 = min(4, chi) if adaptive else chi
    params = {"decomposition": oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc), "truncation": utils.DepthTruncation(nbose), "degree": degree, "chi0": chi0, "chi": chi, "geom": "star"}
    model.add_bath(bath, coupling.wrap(coupling.op("sz", "spin")), tag="phonon", params=params)

    result = oqs.MethodBuilder(model).build("unitary", min_chi=chi0, max_chi=chi)
    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    A.set_state([0 for _ in range(result.system_modes.nmodes())])
    h = pyttn.sop_operator(result.generator, A, result.system_modes, compress=compress)

    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim=12)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim=12, subspace_neigs=6, expansion="subspace")
        sweep.spawning_threshold, sweep.unoccupied_threshold, sweep.minimum_unoccupied = spawning_threshold, unoccupied_threshold, nunoccupied
    sweep.dt = dt
    sweep.coefficient = -1.0j

    timings = np.zeros(nstep)
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        timings[i] = t2 - t1

    stdev = 0
    if nstep > 1:
        stdev = np.std(timings)
    return np.mean(timings), stdev
