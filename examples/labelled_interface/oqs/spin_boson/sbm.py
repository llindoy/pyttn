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

import numpy as np
from numba import jit

import pyttn
from pyttn import oqs, utils
from pyttn.simulation import Observable, TDVPSimulation
from pyttn.ttns.sop import OperatorBuilder, SystemInfo


def sbm_dynamics(alpha, wc, s, eps, delta, chi, dt, method="unitary", Nb=8, Nw=10.0, geom="star", nbose=30, L=30, K=6, Lmin=6, beta=None, nstep=1, ofname=None, degree=None, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):
    """Dynamics of the spin boson model, built with the labelled OQS interface, for any of the four supported methods. The only method-dependent piece is which bath decomposition to register (a chain-mode discretisation for unitary/tedopa, a pole/exponential-fit decomposition for heom/pseudomode) - :class:`~pyttn.oqs.MethodBuilder` derives the Liouville-space doubling and the Tr[rho] trace state automatically wherever the method needs them, so the rest (topology, generator, observables, run loop) is shared code."""

    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    sysinfo = SystemInfo()
    sysinfo["spin"] = pyttn.tls_mode()
    b = OperatorBuilder()
    model = oqs.OQSModel(system_info=sysinfo, system_generator=b.wrap(eps / 2 * b.op("sz", "spin") + delta / 2 * b.op("sx", "spin")))

    bath = oqs.BosonicBath(J, beta=beta)
    coupling = OperatorBuilder()
    if method in ("unitary", "tedopa"):
        chi0 = min(4, chi) if adaptive else chi
        params = {"decomposition": oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc), "truncation": utils.EnergyTruncation(10 * wc, Lmax=nbose, Lmin=4), "degree": degree if degree is not None else 1, "chi0": chi0, "chi": chi, "geom": geom}
        krylov_dim = 16
    else:
        chi0 = 16 if adaptive else chi
        params = {"decomposition": oqs.ESPRITDecomposition(K=K, tmax=nstep * dt, Nt=nstep), "truncation": utils.EnergyTruncation(10 * wc, Lmax=L, Lmin=Lmin), "degree": degree if degree is not None else 2, "chi0": chi0, "chi": chi}
        krylov_dim = 12
    model.add_bath(bath, coupling.wrap(coupling.op("sz", "spin")), tag="phonon", params=params)

    result = oqs.MethodBuilder(model).build(method, min_chi=chi0, max_chi=chi)
    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    print(result.topology.tree)
    print(result.capacity.tree)

    A.set_state([0 for _ in range(result.system_modes.nmodes())])
    h = pyttn.sop_operator(result.generator, A, result.system_modes)
    sz = coupling.wrap(coupling.op("sz", "spin"))

    expansion, integrator_kwargs = ("subspace", {"krylov_dim": krylov_dim, "subspace_neigs": 2}) if adaptive else ("onesite", {"krylov_dim": krylov_dim})
    observables = [Observable("Sz", op=sz)] + ([Observable("norm")] if result.trace_state is not None else [])  # "norm" = Tr[rho](t), only meaningful in Liouville space
    ofname = ofname if ofname is not None else f"sbm_{method}.h5"
    sim = TDVPSimulation(A, h, dt=dt, nstep=nstep, coefficient=-1.0j, observables=observables, reference_states=result.trace_state, system_modes=result.system_modes, site_map=result.site_map, expansion=expansion, integrator_kwargs=integrator_kwargs, output_file=ofname, output_stride=10)
    if adaptive:
        sim.integrator.spawning_threshold, sim.integrator.unoccupied_threshold, sim.integrator.minimum_unoccupied = spawning_threshold, unoccupied_threshold, nunoccupied
    sim.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamics of the spin boson model, for any of the unitary/tedopa/heom/pseudomode methods")

    parser.add_argument("alpha", type=float)
    parser.add_argument("--method", type=str, default="unitary", choices=["unitary", "tedopa", "heom", "pseudomode"])
    parser.add_argument("--wc", type=float, default=5)
    parser.add_argument("--s", type=float, default=1)
    parser.add_argument("--delta", type=float, default=1)
    parser.add_argument("--eps", type=float, default=0)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--chi", type=int, default=32)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=10)
    parser.add_argument("--fname", type=str, default=None)
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-5)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    # unitary/tedopa (chain-mode discretisation) parameters
    parser.add_argument("--Nb", type=int, default=64)
    parser.add_argument("--Nw", type=float, default=10.0)
    parser.add_argument("--geom", type=str, default="star")
    parser.add_argument("--nbose", type=int, default=30)

    # heom/pseudomode (pole/exponential-fit decomposition) parameters
    parser.add_argument("--L", type=int, default=30)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--Lmin", type=int, default=6)

    args = parser.parse_args()
    nstep = int(args.tmax / args.dt)

    sbm_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.dt, method=args.method, Nb=args.Nb, Nw=args.Nw, geom=args.geom, nbose=args.nbose, L=args.L, K=args.K, Lmin=args.Lmin, beta=args.beta, nstep=nstep, ofname=args.fname, degree=args.degree, adaptive=args.subspace, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold=args.unoccupied_threshold)
