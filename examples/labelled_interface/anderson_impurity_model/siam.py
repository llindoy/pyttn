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
from pyttn import oqs
from pyttn.simulation import Observable, TDVPSimulation
from pyttn.ttns.sop import OperatorBuilder, SystemInfo


def siam_dynamics(Nb, Gamma, W, epsd, deps, U, chi, dt, chi0=8, beta=None, nstep=1, geom="star", ofname="siam.h5", degree=1, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, init_state="up"):
    """Dynamics of the single impurity Anderson model, built with the labelled OQS interface. Each spin channel's lead is declared once as a single fermionic bath, split internally into occupied/empty orbitals (`channels="filled_empty"`) and attached to the tree as two branches under one node - :class:`~pyttn.oqs.MethodBuilder` proposes the system topology automatically and Jordan-Wigner transforms the assembled Hamiltonian (and, via `result.jordan_wigner(...)`, the observables) using an explicit user-specified label ordering."""

    @jit(nopython=True)
    def V(w):
        return np.where(np.abs(w / W) < 1, Gamma * np.sqrt( W**2 - w ** 2) / (W), 0)

    sysinfo = SystemInfo({"c_d": pyttn.fermion_mode(), "c_u": pyttn.fermion_mode()})
    b = OperatorBuilder()
    model = oqs.OQSModel(system_info=sysinfo, system_generator=b.wrap(epsd * b.fop("n", "c_d") + epsd * b.fop("n", "c_u") + U * b.fop("n", "c_d") * b.fop("n", "c_u")))

    # both leads are physically the same continuum bath, coupled independently to
    # each spin channel; each is split into an occupied ("filled") and an
    # unoccupied ("empty") channel, discretised with bounds appropriate to each.
    bath = oqs.FermionicBath(V, beta=beta)

    decomp_filled = oqs.OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W, sigma="+"))
    decomp_empty = oqs.OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W, sigma="-"))
    lead_params = {"decomposition": (decomp_filled, decomp_empty), "channels": "filled_empty", "Ef": 0.0, "attachment": "branch", "degree": degree, "chi0": chi0, "chi": chi, "geom": geom}

    coupling_d, coupling_u = OperatorBuilder(), OperatorBuilder()
    model.add_bath(bath, [coupling_d.wrap(coupling_d.fop("cdag", "c_d")), coupling_d.wrap(coupling_d.fop("c", "c_d"))], tag="lead_d", params=lead_params)
    model.add_bath(bath, [coupling_u.wrap(coupling_u.fop("cdag", "c_u")), coupling_u.wrap(coupling_u.fop("c", "c_u"))], tag="lead_u", params=lead_params)

    # Jordan-Wigner ordering: reversed within the down-spin channel, normal within
    # the up-spin channel - the same "meet in the middle" trick the original
    # hand-built script used to keep JW strings short, expressed via labels rather
    # than hand-computed indices, and independent of whatever tree MethodBuilder proposes.
    jw_ordering = list(reversed(["c_d"] + [f"lead_d_{i}" for i in range(2 * Nb)])) + ["c_u"] + [f"lead_u_{i}" for i in range(2 * Nb)]

    result = oqs.MethodBuilder(model).build("unitary", min_chi=chi0, max_chi=chi, jordan_wigner_ordering=jw_ordering)
    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    print(result.capacity.tree)
    # initial state: Fermi sea in both leads (occupied/"filled" orbitals populated,
    # empty orbitals not) plus one extra electron on the requested impurity spin
    state = [0 for _ in range(len(result.site_map))]
    for tag in ("lead_d", "lead_u"):
        for i in range(Nb):
            state[result.site_map[f"{tag}_{i}"]] = 1
    #state[result.site_map["c_u" if init_state == "up" else "c_d"]] = 1
    A.set_state(state)

    h = pyttn.sop_operator(result.generator, A, result.system_modes)
    obs_builder = OperatorBuilder()
    n_u = result.jordan_wigner(obs_builder.wrap(obs_builder.fop("n", "c_u")))
    n_d = result.jordan_wigner(obs_builder.wrap(obs_builder.fop("n", "c_d")))
    n_ud = result.jordan_wigner(obs_builder.wrap(obs_builder.fop("n", "c_u") * obs_builder.fop("n", "c_d")))

    expansion, integrator_kwargs = ("subspace", {"krylov_dim": 12, "subspace_krylov_dim": 10, "subspace_neigs": 2}) if adaptive else ("onesite", {"krylov_dim": 12})
    observables = [Observable("n_u", op=n_u), Observable("n_d", op=n_d), Observable("n_u n_d", op=n_ud)]
    sim = TDVPSimulation(A, h, dt=dt, nstep=nstep, coefficient=-1.0j, observables=observables, system_modes=result.system_modes, site_map=result.site_map, expansion=expansion, integrator_kwargs=integrator_kwargs, output_file=ofname, output_stride=10)
    if adaptive:
        sim.integrator.spawning_threshold, sim.integrator.unoccupied_threshold, sim.integrator.minimum_unoccupied = spawning_threshold, unoccupied_threshold, nunoccupied
    if geom == "ipchain":
        sim.integrator.use_time_dependent_hamiltonian = True
    sim.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamics of the single impurity Anderson model.")

    parser.add_argument("--Gamma", type=float, default=1)
    parser.add_argument("--W", type=float, default=10)
    parser.add_argument("--epsd", type=float, default=-1.25*np.pi)
    parser.add_argument("--deps", type=float, default=0.0)
    parser.add_argument("--U", type=float, default=2.5*np.pi)
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--geom", type=str, default="chain")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--chi", type=int, default=128)
    parser.add_argument("--chi0", type=int, default=16)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=100)
    parser.add_argument("--fname", type=str, default="siam.h5")
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-6)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)
    parser.add_argument("--initial_state", type=str, default="up")

    args = parser.parse_args()
    nstep = int(args.tmax / args.dt) + 1

    siam_dynamics(args.N, args.Gamma, args.W, args.epsd, args.deps, args.U, args.chi, args.dt, chi0=args.chi0, beta=args.beta, nstep=nstep, geom=args.geom, ofname=args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold=args.unoccupied_threshold, adaptive=args.subspace, degree=args.degree, init_state=args.initial_state)
