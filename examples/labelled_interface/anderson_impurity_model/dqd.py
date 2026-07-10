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
from pyttn.ttns.sop import SystemInfo


def siam_dynamics(Nb, Gamma, W, eps1, eps2, U, chi, dt, chi0=8, beta=None, nstep=1, geom="star", ofname="siam.h5", degree=1, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, init_state="up"):
    """Dynamics of the single impurity Anderson model, built with the labelled OQS interface. Each spin channel's lead is declared once as a single fermionic bath, split internally into occupied/empty orbitals (`channels="filled_empty"`) and attached to the tree as two branches under one node - :class:`~pyttn.oqs.MethodBuilder` proposes the system topology automatically and Jordan-Wigner transforms the assembled Hamiltonian (and, via `result.jordan_wigner(...)`, the observables) using an explicit user-specified label ordering."""

    @jit(nopython=True)
    def V(w):
        return np.where(np.abs(w / W) < 1, Gamma * np.sqrt( W**2 - w ** 2) / (W), 0)

    #define the system operators
    @pyttn.operator
    def H():
        return (
            eps1 * pyttn.fop("n", "c_d1") + eps1 * pyttn.fop("n", "c_u1") 
            + eps2 * pyttn.fop("n", "c_d2") + eps2 * pyttn.fop("n", "c_u2")
            + U * (pyttn.fop("n", "c_d1") + pyttn.fop("n", "c_u1"))* (pyttn.fop("n", "c_d2") + pyttn.fop("n", "c_u2"))
        )
    
    @pyttn.operator
    def cdag(site):
        return pyttn.fop("cdag", "c_"+site)

    @pyttn.operator
    def c(site):
        return pyttn.fop("c", "c_"+site)
    
    @pyttn.operator
    def n(site):
        return pyttn.fop("n", "c_"+site)

    sysinfo = SystemInfo({"c_d1": pyttn.fermion_mode(), "c_u1": pyttn.fermion_mode(), "c_d2": pyttn.fermion_mode(), "c_u2": pyttn.fermion_mode()})
    model = oqs.OQSModel(system_info=sysinfo, system_generator=H())
    # set up fermionic bath object
    bath = oqs.FermionicBath(V, beta=beta)

    #and decomposition objects for the filled and empty contributions to the bath respectively
    decomp_filled = oqs.OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W, sigma="+"))
    decomp_empty = oqs.OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W, sigma="-"))
    lead_params = {"decomposition": (decomp_filled, decomp_empty), "channels": "filled_empty", "Ef": 0.0, "attachment": "branch", "degree": degree, "chi0": chi0, "chi": chi, "geom": geom}

    #set up the coupling operators and add them to the bath
    for label in ["d1", "u1", "d2", "u2"]:
        model.add_bath(bath, [cdag(label), c(label)], tag="lead_"+label, params=lead_params)

    # Jordan-Wigner ordering: reversed within the down-spin channel, normal within the up-spin channel 
    jw_ordering_1 = list(reversed(["c_d1"] + [f"lead_d1_{i}" for i in range(2 * Nb)])) + ["c_u1"] + [f"lead_u1_{i}" for i in range(2 * Nb)]
    jw_ordering_2 = list(reversed(["c_d2"] + [f"lead_d2_{i}" for i in range(2 * Nb)])) + ["c_u2"] + [f"lead_u2_{i}" for i in range(2 * Nb)]

    result = oqs.MethodBuilder(model).build("unitary", min_chi=chi0, max_chi=chi, jordan_wigner_ordering=jw_ordering_1+jw_ordering_2)
    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    print(result.topology.tree)
    # initial state: Fermi sea in both leads (occupied/"filled" orbitals populated, empty orbitals not)
    state = [0 for _ in range(len(result.site_map))]
    for tag in ("lead_d1", "lead_u1", "lead_d2", "lead_u2"):
        for i in range(Nb):
            state[result.site_map[f"{tag}_{i}"]] = 1
    A.set_state(state)

    h = pyttn.sop_operator(result.generator, A, result.system_modes)
    n_u1 = result.jordan_wigner(n("u1"))
    n_d1 = result.jordan_wigner(n("d1"))
    n_ud1 = result.jordan_wigner(n("u1")*n("d1"))
    n_u2 = result.jordan_wigner(n("u2"))
    n_d2 = result.jordan_wigner(n("d2"))
    n_ud2 = result.jordan_wigner(n("u2")*n("d2"))

    expansion, integrator_kwargs = ("subspace", {"krylov_dim": 12, "subspace_krylov_dim": 10, "subspace_neigs": 2}) if adaptive else ("onesite", {"krylov_dim": 12})
    observables1 = [Observable("n_u1", op=n_u1), Observable("n_d1", op=n_d1), Observable("n_u1 n_d1", op=n_ud1)]
    observables2 = [Observable("n_u2", op=n_u2), Observable("n_d2", op=n_d2), Observable("n_u2 n_d2", op=n_ud2)]

    sim = TDVPSimulation(A, h, dt=dt, nstep=nstep, coefficient=-1.0j, observables=observables1+observables2, system_modes=result.system_modes, site_map=result.site_map, expansion=expansion, integrator_kwargs=integrator_kwargs, output_file=ofname, output_stride=10)
    if adaptive:
        sim.integrator.spawning_threshold, sim.integrator.unoccupied_threshold, sim.integrator.minimum_unoccupied = spawning_threshold, unoccupied_threshold, nunoccupied
    sim.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamics of the single impurity Anderson model.")

    parser.add_argument("--Gamma", type=float, default=1)
    parser.add_argument("--W", type=float, default=10)
    parser.add_argument("--eps1", type=float, default=-1.25*np.pi)
    parser.add_argument("--eps2", type=float, default=-1.25*np.pi)
    parser.add_argument("--U", type=float, default=2.5*np.pi)
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--geom", type=str, default="chain")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--chi", type=int, default=128)
    parser.add_argument("--chi0", type=int, default=16)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=5)
    parser.add_argument("--fname", type=str, default="dqd.h5")
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-6)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)
    parser.add_argument("--initial_state", type=str, default="up")

    args = parser.parse_args()
    nstep = int(args.tmax / args.dt) + 1

    siam_dynamics(args.N, args.Gamma, args.W, args.eps1, args.eps2, args.U, args.chi, args.dt, chi0=args.chi0, beta=args.beta, nstep=nstep, geom=args.geom, ofname=args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold=args.unoccupied_threshold, adaptive=args.subspace, degree=args.degree, init_state=args.initial_state)
