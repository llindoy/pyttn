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

import copy
import os
import sys
import time
os.environ["OMP_NUM_THREADS"] = "1"

import h5py
import numpy as np
from pyrazine_hamiltonian import electronic_operator_dictionary, hamiltonian, modes

from pyttn import boson_mode, generic_mode, matrix_element, sop_operator, system_modes, tdvp, ttn, CorrelationMeasures, generate_hierarchical_clustering_tree, convert_nx_to_tree, ntreeBuilder

fs = 41.341374575751

def run_initial_step(A, h, sweep, dt, nstep=10):
    """Logarithmic timestep ramp for the first step (matches TDVPSimulation's initial_ramp, hand-rolled here since the a(t) measurement below does not fit the Simulation abstraction)."""
    tp = 0
    ts = np.logspace(np.log10(dt * 1e-5), np.log10(dt), nstep)
    for i in range(nstep):
        dti = ts[i] - tp
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    return A, h, sweep


def output_results(ofname, timepoints, res, maxchi, runtime, cmat):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=timepoints)
    h5.create_dataset("a(t)", data=res)
    h5.create_dataset("maxchi", data=maxchi)
    h5.create_dataset("runtime", data=runtime * np.ones(1))
    h5.create_dataset("mutualinf", data=cmat)
    h5.close()


def mutual_information_tree(_A, _h, sweep, nstep, dt, nstrides=50):
    """Run a short trial MPS dynamics simulation, tracking pairwise mutual information, then
    build a hierarchical-clustering tree topology from the accumulated correlation matrix.
    This bespoke topology-discovery machinery is unrelated to the labelled-operator interface."""
    A = copy.deepcopy(_A)
    h = copy.deepcopy(_h)
    run_initial_step(A, h, sweep, dt)

    corr_mat = np.zeros((len(A), len(A)))

    def update_corr_mat(A, corr_mat):
        corr = CorrelationMeasures()
        for xi in range(len(A)):
            for yi in range(xi + 1, len(A)):
                mij = corr.mutualInformation(A, xi, yi)
                print("mutual information:", xi, yi, mij)
                if mij > corr_mat[xi, yi]:
                    corr_mat[xi, yi] = mij
        for xi in range(len(A)):
            for yi in range(xi + 1, len(A)):
                corr_mat[yi, xi] = corr_mat[xi, yi]

    for i in range(1, nstep):
        sweep.step(A, h)
        print("mps dyn:", i, "of", nstep, end='                                   \r', flush=True)
        if i % nstrides == 0:
            update_corr_mat(A, corr_mat)

    update_corr_mat(A, corr_mat)

    spanning_tree, spanning_root_ind = generate_hierarchical_clustering_tree(corr_mat)
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)
    ntreeBuilder.sanitise(tree)

    return corr_mat, tree, leaf_ordering

def initialise_state(topo, capacity, Nc):
    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_seed(0)
    state = np.zeros(Nc + 1, dtype=int)
    state[0] = 1
    A.set_state(state)
    return A

def initialise_propagation(A, h, adaptive, spawning_threshold, unoccupied_threshold, nunoccupied, dt):
    if adaptive:
        sweep = tdvp(A, h, krylov_dim=12, expansion="subspace", subspace_krylov_dim=12, subspace_neigs=6)
        sweep.spawning_threshold, sweep.unoccupied_threshold, sweep.minimum_unoccupied = spawning_threshold, unoccupied_threshold, nunoccupied
    else:
        sweep = tdvp(A, h, krylov_dim=12)
    sweep.dt = dt
    sweep.coefficient = -1.0j
    return sweep

def pyrazine_dynamics(maxchi, tmax, dt, adaptive=True, spawning_threshold=1e-6, unoccupied_threshold=1e-4, nunoccupied=0, ofname="pyrazine.h5", output_skip=1):
    # Here we half the total integration time as we are computing a(t) = <\psi(t/2)^*|\psi(t/2)>
    nsteps = int(tmax / (2 * dt)) + 1

    # The dimension of each of the 24 vibrational bosonic modes
    m = [40, 32, 20, 12, 8, 4, 8, 24, 24, 8, 8, 24, 20, 4, 72, 80, 6, 20, 6, 6, 6, 32, 6, 4]
    Nc = len(m)

    # flat mode ordering used for the trial MPS run: electronic mode + 24 vibrational modes
    sysinf = system_modes(Nc + 1)
    sysinf[0] = generic_mode(2)
    for i in range(len(m)):
        sysinf[i + 1] = [boson_mode(m[i])]

    # build the labelled Hamiltonian once, compiled against the flat "el" + 24 vibrational-mode primitive ordering
    H = hamiltonian()
    site_map = {label: i for i, label in enumerate(modes())}
    Hsop = H.compile(site_map, len(site_map))
    opdict = electronic_operator_dictionary(site_map, len(site_map))

    _chi = 8
    topo = ntreeBuilder.mps_tree(sysinf.mode_dimensions(), _chi, _chi)
    capacity = ntreeBuilder.mps_tree(sysinf.mode_dimensions(), maxchi, maxchi)

    A = initialise_state(topo, capacity, Nc)
    h = sop_operator(Hsop, A, sysinf, opdict)
    sweep = initialise_propagation(A, h, adaptive, spawning_threshold, unoccupied_threshold, nunoccupied, dt)

    B = initialise_state(topo, topo, Nc)
    h = sop_operator(Hsop, A, sysinf, opdict)
    mel = matrix_element(A)
    sweep = initialise_propagation(A, h, adaptive, spawning_threshold, unoccupied_threshold, nunoccupied, dt)

    res = np.zeros(nsteps + 1, dtype=np.complex128)
    maxchi_arr = np.zeros(nsteps + 1)
    res[0] = mel(B, A)
    maxchi_arr[0] = A.maximum_bond_dimension()

    t1 = time.time()

    A, h, sweep = run_initial_step(A, h, sweep, dt)

    B = copy.deepcopy(A)
    B.conj()
    res[1] = mel(B, A)
    maxchi_arr[1] = A.maximum_bond_dimension()

    sweep.dt = dt
    timepoints = np.arange(nsteps + 1) * dt * 2 / fs

    for i in range(1, nsteps):
        print(i, nsteps)
        sys.stdout.flush()
        sweep.step(A, h)

        B = copy.deepcopy(A)
        B.conj()
        t2 = time.time()

        res[i + 1] = mel(B, A)
        maxchi_arr[i + 1] = A.maximum_bond_dimension()

        if i % output_skip == 0:
            output_results(ofname, timepoints, res, maxchi_arr, (t2 - t1), np.zeros((1, 1)))

    output_results(ofname, timepoints, res, maxchi_arr, (t2 - t1), np.zeros((1, 1)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamics of the zero temperature spin boson model with")

    parser.add_argument("chimax", type=int)
    parser.add_argument("--fname", type=str, default=None)
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-7)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)
    parser.add_argument("--dt", type=float, default=0.125)
    parser.add_argument("--tmax", type=float, default=150)
    parser.add_argument("--output_skip", type=int, default=1)

    args = parser.parse_args()
    fname = args.fname if args.fname is not None else "results/pyrazine_mps_" + str(args.chimax) + ".h5"
    pyrazine_dynamics(args.chimax, args.tmax * fs, args.dt * fs, ofname=fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold=args.unoccupied_threshold, adaptive=args.subspace, output_skip=args.output_skip)
