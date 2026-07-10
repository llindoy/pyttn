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

import h5py
import numpy as np
from numba import jit

import pyttn
from pyttn import oqs, utils
from pyttn.simulation import Observable, TDVPSimulation
from pyttn.ttns.sop import OperatorBuilder, SystemInfo


def xychain_dynamics(Ns, alpha, wc, eta, chi, dt, method="unitary", Nb=40, Nw=4.0, geom="star", nbose=30, nbose_min=4, L=30, K=6, Lmin=6, Ecut=10.0, beta=None, nstep=1, ofname=None, degree=None, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):
    """Dynamics of a dissipative XY spin chain, each spin coupled to its own independent bosonic bath, for any of the four supported methods. As in :mod:`sbm`, the only method-dependent piece is which bath decomposition to register per site; one :class:`~pyttn.oqs.OQSModel` with Ns independently tagged baths and one :class:`~pyttn.oqs.MethodBuilder` call proposes the entire joint system+bath tree topology and (for heom/pseudomode) the Tr[rho] trace state, regardless of method."""

    @jit(nopython=True)
    def J(w):
        return 2 * np.pi * alpha * w * np.exp(-np.abs(w / wc) ** 2)

    labels = [f"spin{i}" for i in range(Ns)]
    sysinfo = SystemInfo()
    for label in labels:
        sysinfo[label] = pyttn.tls_mode()

    b = OperatorBuilder()
    Hsys = b.op("sz", labels[0])
    for i in range(1, Ns):
        Hsys = Hsys + b.op("sz", labels[i])
    for i in range(Ns - 1):
        Hsys = Hsys + (1 - eta) * b.op("sx", labels[i]) * b.op("sx", labels[i + 1]) + (1 + eta) * b.op("sy", labels[i]) * b.op("sy", labels[i + 1])
    model = oqs.OQSModel(system_info=sysinfo, system_generator=b.wrap(Hsys))

    unitary_like = method in ("unitary", "tedopa")
    chi0 = (min(4, chi) if unitary_like else min(16, chi)) if adaptive else chi
    krylov_dim = 12 if unitary_like else 16
    for i in range(Ns):
        bath = oqs.BosonicBath(J, beta=beta) if unitary_like else oqs.BosonicBath(J, beta=beta, wmax=wc * Ecut)
        coupling = OperatorBuilder()
        if unitary_like:
            params = {"decomposition": oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc), "truncation": utils.EnergyTruncation(Ecut * wc, Lmax=nbose, Lmin=nbose_min), "degree": degree if degree is not None else 1, "chi0": chi0, "chi": chi, "geom": geom}
        else:
            params = {"decomposition": oqs.ESPRITDecomposition(K=K, tmax=nstep * dt, Nt=nstep), "truncation": utils.EnergyTruncation(Ecut * wc, Lmax=L, Lmin=Lmin), "degree": degree if degree is not None else 1, "chi0": chi0, "chi": chi}
        model.add_bath(bath, coupling.wrap(coupling.op("sz", labels[i])), tag=f"bath{i}", params=params)

    result = oqs.MethodBuilder(model).build(method, min_chi=chi0, max_chi=chi)
    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    state = [0 for _ in range(result.system_modes.nmodes())]
    state[result.topology.leaf_order().index(labels[(Ns - 1) // 2])] = 1 if unitary_like else 3
    A.set_state(state)
    h = pyttn.sop_operator(result.generator, A, result.system_modes)
    obs_builder = OperatorBuilder()
    szops = [obs_builder.wrap(obs_builder.op("sz", label)) for label in labels]

    expansion, integrator_kwargs = ("subspace", {"krylov_dim": krylov_dim, "subspace_neigs": 6}) if adaptive else ("onesite", {"krylov_dim": krylov_dim})
    observables = [Observable(f"Sz{i}", op=szops[i]) for i in range(Ns)] + ([Observable("norm")] if result.trace_state is not None else [])
    ofname = ofname if ofname is not None else f"xychain_{method}.h5"
    sim = TDVPSimulation(A, h, dt=dt, nstep=nstep, coefficient=-1.0j, observables=observables, reference_states=result.trace_state, system_modes=result.system_modes, site_map=result.site_map, expansion=expansion, integrator_kwargs=integrator_kwargs, output_file=ofname, output_stride=1)
    if adaptive:
        sim.integrator.spawning_threshold, sim.integrator.unoccupied_threshold, sim.integrator.minimum_unoccupied = spawning_threshold, unoccupied_threshold, nunoccupied

    t1 = time.time()
    sim.run()
    t2 = time.time()
    with h5py.File(ofname, "a") as h5:
        h5.create_dataset("time", data=np.array([t2 - t1]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamics of a dissipative XY spin chain, for any of the unitary/tedopa/heom/pseudomode methods")

    parser.add_argument("--method", type=str, default="unitary", choices=["unitary", "tedopa", "heom", "pseudomode"])
    parser.add_argument("--Ns", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=0.32)
    parser.add_argument("--wc", type=float, default=4)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--chi", type=int, default=36)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=5)
    parser.add_argument("--fname", type=str, default=None)
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-5)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)
    parser.add_argument("--ecut", type=float, default=10)

    # unitary/tedopa (chain-mode discretisation) parameters
    parser.add_argument("--Nb", type=int, default=40)
    parser.add_argument("--Nw", type=float, default=4.0)
    parser.add_argument("--geom", type=str, default="star")
    parser.add_argument("--nbose", type=int, default=30)
    parser.add_argument("--nbose_min", type=int, default=6)

    # heom/pseudomode (pole/exponential-fit decomposition) parameters
    parser.add_argument("--L", type=int, default=30)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--Lmin", type=int, default=6)

    args = parser.parse_args()
    nstep = int(args.tmax / args.dt) + 1

    xychain_dynamics(args.Ns, args.alpha, args.wc, args.eta, args.chi, args.dt, method=args.method, Nb=args.Nb, Nw=args.Nw, geom=args.geom, nbose=args.nbose, nbose_min=args.nbose_min, L=args.L, K=args.K, Lmin=args.Lmin, Ecut=args.ecut, beta=args.beta, nstep=nstep, ofname=args.fname, degree=args.degree, adaptive=args.subspace, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold=args.unoccupied_threshold)
