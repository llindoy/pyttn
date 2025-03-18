import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
import sys
import h5py
import argparse

from pyttn import ntree, ntreeBuilder
from pyttn import system_modes, generic_mode, boson_mode
from pyttn import ttn, sop_operator, matrix_element, tdvp, site_operator, sOP

fs = 41.341374575751


def build_ttn_tree(mdims, chi1, chi2, ldims, degree=2, use_multiset=False):
    from p3ht_pcbm_hamiltonian import ot_mode_index

    class chi_step:
        def __init__(self, chimax, chimin, N, degree=degree):
            self.chimin = chimin
            if N % degree == 0:
                self.Nl = int(int(np.log(N) / np.log(degree)) + 1)
            else:
                self.Nl = int(int(np.log(N) / np.log(degree)) + 2)

            self.nx = int((chimax - chimin) // self.Nl)

        def __call__(self, li):
            ret = int((self.Nl - li) * self.nx + self.chimin)
            return ret

    # set up the mode indexing for splitting the trees as needed
    sys_skip = 1
    if use_multiset:
        sys_skip = 0

    mode_F = [sys_skip + 1 + i for i in range(8)]
    modeOT_lf = []
    modeOT_hf = []

    for n in range(13):
        for li in range(6):
            modeOT_lf.append(sys_skip + 1 + 8 + ot_mode_index(13, n, li))

        for li in range(6, 8):
            modeOT_hf.append(sys_skip + 1 + 8 + ot_mode_index(13, n, li))

    chim1 = min(chi1, 8)
    chim2 = min(chi2, 8)
    if use_multiset:
        topo = ntree("(1(%d(%d(128)))(%d(%d)(%d)))" % (chi1, ldims[0], chi1, chi1, chi1))

        ntreeBuilder.mlmctdh_subtree(
            topo()[0],
            mdims[mode_F[:]],
            2,
            chi_step(chi2, chim2, len(mode_F)),
            ldims[mode_F[:]],
        )

        ntreeBuilder.mlmctdh_subtree(
            topo()[1][0],
            mdims[modeOT_lf[:]],
            2,
            chi_step(chi1, chim1, len(modeOT_lf)),
            ldims[modeOT_lf[:]],
        )

        ntreeBuilder.mlmctdh_subtree(
            topo()[1][1],
            mdims[modeOT_hf[:]],
            2,
            chi_step(chi2, chim2, len(modeOT_hf)),
            ldims[modeOT_hf[:]],
        )
    else:
        topo = ntree(
            "(1(%d(26))(%d(%d(128)))(%d(%d)(%d)))"
            % (mdims[0], chi1, ldims[1], chi1, chi1, chi1)
        )
        ntreeBuilder.mlmctdh_subtree(
            topo()[1],
            mdims[mode_F[:]],
            2,
            chi_step(chi2, chim2, len(mode_F)),
            ldims[mode_F[:]],
        )
        ntreeBuilder.mlmctdh_subtree(
            topo()[2][0],
            mdims[modeOT_lf[:]],
            2,
            chi_step(chi1, chim1, len(modeOT_lf)),
            ldims[modeOT_lf[:]],
        )
        ntreeBuilder.mlmctdh_subtree(
            topo()[2][1],
            mdims[modeOT_hf[:]],
            2,
            chi_step(chi2, chim2, len(modeOT_hf)),
            ldims[modeOT_hf[:]],
        )

    ntreeBuilder.sanitise(topo)
    return topo


def setup_local_dimensions():
    """Sets up the set of arrays storing the local Hilbert space dimension of each mode, the local dimension to use for"""
    Nmodes = 114
    mdims = [0 for i in range(Nmodes)]
    mdims[0] = 26
    mdims[1] = 128

    for i in range(2, Nmodes):
        mdims[i] = 30

    ldims = [0 for i in range(Nmodes)]
    ldims[0] = 26
    ldims[1] = 24
    for i in range(2, Nmodes):
        ldims[i] = 12

    ldims0 = [0 for i in range(Nmodes)]
    ldims0[0] = 26
    for i in range(1, Nmodes):
        ldims0[i] = 6

    mdims = np.array(mdims, dtype=int)
    ldims = np.array(ldims, dtype=int)
    ldims0 = np.array(ldims0, dtype=int)

    return mdims, ldims, ldims0


def run_initial_step(A, h, sweep, dt, nstep=10):
    tp = 0
    ts = np.logspace(np.log10(dt * 1e-5), np.log10(dt), nstep)
    for i in range(nstep):
        dti = ts[i] - tp
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    return A, h, sweep


def output_results(ofname, timepoints, res, maxchi, runtime):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=timepoints)
    for j in range(13):
        h5.create_dataset("|LE%d><LE%d|" % (j, j), data=res[:, 2 * j])
        h5.create_dataset("|CS%d><CS%d|" % (j, j), data=res[:, 2 * j + 1])
    h5.create_dataset("maxchi", data=maxchi)
    h5.create_dataset("runtime", data=runtime * np.ones(1))
    h5.close()


def p3ht_pcbm_single_set(topo, capacity, mode_dims, tmax=200, dt=0.25, adaptive=True,
                         spawning_threshold=1e-6, unoccupied_threshold=1e-4, nunoccupied=0,
                         ofname="p3ht_pcbm.h5", output_skip=1):
    from p3ht_pcbm_hamiltonian import hamiltonian

    """Function for performing the dynamics of the single set p3ht_pcbm model
    """
    nsteps = int(tmax / (dt)) + 1
    Nmodes = len(mode_dims)

    # set up the system information
    sysinf = system_modes(Nmodes)
    sysinf[0] = generic_mode(26)
    for i in range(1, Nmodes):
        sysinf[i] = boson_mode(mode_dims[i])

    # get the system Hamiltonian as a string and the required operator dictionary
    H, opdict = hamiltonian()

    # set up the wavefunction for simulating the dynamics
    A = ttn(topo, capacity, dtype=np.complex128)
    state = np.zeros(Nmodes, dtype=int)
    A.set_state(state)

    # set up all of the observable objects for the system
    ops = []
    for i in range(13):
        ops.append(site_operator(sOP("|LE%d><LE%d|" % (i, i), 0), sysinf, opdict))
        ops.append(site_operator(sOP("|CS%d><CS%d|" % (i, i), 0), sysinf, opdict))

    # setup the matrix element calculation object
    mel = matrix_element(A, nbuffers=1)

    # set up the sum of product operator object for evolution
    h = sop_operator(H, A, sysinf, opdict)

    # setup the evolution object
    sweep = None
    if adaptive:
        sweep = tdvp(
            A,
            h,
            krylov_dim=12,
            expansion="subspace",
            subspace_krylov_dim=12,
            subspace_neigs=6,
        )
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold = unoccupied_threshold
        sweep.minimum_unoccupied = nunoccupied
    else:
        sweep = tdvp(A, h, krylov_dim=12)
    sweep.expmv_tol = 1e-10
    sweep.dt = dt
    sweep.coefficient = -1.0j

    # setup buffers for storing the results
    res = np.zeros((nsteps + 1, len(ops)), dtype=np.complex128)
    maxchi = np.zeros(nsteps + 1)

    for i in range(len(ops)):
        res[0, i] = mel(ops[i], A)
    maxchi[0] = A.maximum_bond_dimension()

    t1 = time.time()

    # perform a set of steps with a logarithmic timestep discretisation
    A, h, sweep = run_initial_step(A, h, sweep, dt)

    for j in range(len(ops)):
        res[1, j] = mel(ops[j], A)
    maxchi[1] = A.maximum_bond_dimension()

    sweep.dt = dt

    timepoints = np.arange(nsteps + 1) * dt / fs

    # perform the remaining dynamics steps
    for i in range(1, nsteps):
        print(i, nsteps)
        sys.stdout.flush()
        sweep.step(A, h)

        t2 = time.time()

        for j in range(len(ops)):
            res[i + 1, j] = mel(ops[j], A)
        maxchi[i + 1] = A.maximum_bond_dimension()

        if i % output_skip == 0:
            output_results(ofname, timepoints, res, maxchi, (t2 - t1))

    t2 = time.time()
    output_results(ofname, timepoints, res, maxchi, (t2 - t1))


def p3ht_pcbm_multiset(
    topo, mode_dims, tmax=200, dt=0.25, ofname="p3ht_pcbm.h5", output_skip=1
):
    from pyttn import ms_ttn, ms_sop_operator
    from p3ht_pcbm_hamiltonian import multiset_hamiltonian

    """Function for performing the dynamics of the multiset p3ht_pcbm model
    """
    nsteps = int(tmax / (dt)) + 1
    Nmodes = len(mode_dims)

    # set up the system information
    sysinf = system_modes(Nmodes)
    for i in range(Nmodes):
        sysinf[i] = boson_mode(mode_dims[i])

    # get the system Hamiltonian as a string and the required operator dictionary
    H = multiset_hamiltonian()

    # set up the wavefunction for simulating the dynamics
    A = ms_ttn(26, topo, dtype=np.complex128)
    state = [[0 for i in range(Nmodes)] for j in range(26)]
    coeff = np.zeros(26, dtype=np.float64)
    coeff[0] = 1
    A.set_state(coeff, state)

    # setup the matrix element calculation object
    mel = matrix_element(A)

    # set up the sum of product operator object for evolution
    h = ms_sop_operator(H, A, sysinf)

    # setup the evolution object
    sweep = tdvp(A, h, krylov_dim=12)
    sweep.expmv_tol = 1e-10
    sweep.dt = dt
    sweep.coefficient = -1.0j

    # setup buffers for storing the results
    res = np.zeros((nsteps + 1, 26), dtype=np.complex128)
    maxchi = np.zeros((nsteps + 1))

    for j in range(26):
        res[0, j] = mel(A.slice(j))

    t1 = time.time()

    # perform a set of steps with a logarithmic timestep discretisation
    A, h, sweep = run_initial_step(A, h, sweep, dt)

    for j in range(26):
        res[1, j] = mel(A.slice(j))

    for j in range(26):
        res[1, j] = mel(A.slice(j))
    sweep.dt = dt

    timepoints = np.arange(nsteps + 1) * dt / fs

    # perform the remaining dynamics steps
    for i in range(1, nsteps):
        print(i, nsteps)
        sys.stdout.flush()
        sweep.step(A, h)

        t2 = time.time()

        for j in range(26):
            res[i + 1, j] = mel(A.slice(j))

        if i % output_skip == 0:
            output_results(ofname, timepoints, res, maxchi, (t2 - t1))

    t2 = time.time()
    output_results(ofname, timepoints, res, maxchi, (t2 - t1))


def p3ht_pcbm_dynamics(
    ansatz,
    chi1,
    chi2=None,
    tmax=200,
    dt=0.25,
    use_multiset=False,
    adaptive=True,
    spawning_threshold=1e-6,
    unoccupied_threshold=1e-4,
    nunoccupied=0,
    ofname="p3ht_pcbm.h5",
    output_skip=1,
):
    # if we are not performing multiset simulations
    if not use_multiset:
        # set up the tree topologies for representing the wavefunction
        mode_dims, local_dims, local_dims_initial = setup_local_dimensions()
        if ansatz == "mps":
            chi0 = min(16, chi1)

            topo = ntreeBuilder.mps_tree(mode_dims, chi0, local_dims_initial)
            capacity = ntreeBuilder.mps_tree(mode_dims, chi1, local_dims)

        elif ansatz == "ttn":
            chi0 = min(chi1, 16)
            chi0 = min(chi0, chi2)

            topo = build_ttn_tree(mode_dims, chi0, chi0, local_dims_initial)
            capacity = build_ttn_tree(mode_dims, chi1, chi2, local_dims)
        else:
            raise RuntimeError(
                'Ansatz argument not recognized.  Valid options are "mps" or"ttn"'
            )
        p3ht_pcbm_single_set(
            topo,
            capacity,
            mode_dims,
            tmax=tmax,
            dt=dt,
            adaptive=adaptive,
            spawning_threshold=spawning_threshold,
            unoccupied_threshold=unoccupied_threshold,
            nunoccupied=nunoccupied,
            ofname=ofname,
            output_skip=output_skip,
        )
    else:
        # set up the tree topologies for representing the wavefunction.  In the multiset form we discard the first mode dim
        mode_dims, local_dims, local_dims_initial = setup_local_dimensions()
        mode_dims = mode_dims[1:]
        local_dims = local_dims[1:]
        local_dims_initial = local_dims_initial[1:]

        if ansatz == "mps":
            chi0 = min(16, chi1)

            topo = ntreeBuilder.mps_tree(mode_dims, chi0, local_dims_initial)

        elif ansatz == "ttn":
            chi0 = min(chi1, 16)
            chi0 = min(chi0, chi2)

            topo = build_ttn_tree(
                mode_dims, chi0, chi0, local_dims_initial, use_multiset=True
            )
        else:
            raise RuntimeError(
                'Ansatz argument not recognized.  Valid options are "mps" or"ttn"'
            )
        p3ht_pcbm_multiset(
            topo, mode_dims, tmax=tmax, dt=dt, ofname=ofname, output_skip=output_skip
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyttn test")

    parser.add_argument("ansatz", type=str, default="mps")

    parser.add_argument("chi", type=int)
    parser.add_argument("--chi2", type=int, default=None)
    parser.add_argument("--use_multiset", action="store_true")

    # subspace expansion parameters
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-6)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    parser.add_argument("--fname", type=str, default="p3ht_pcbm.h5")

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.125)
    parser.add_argument("--tmax", type=float, default=200)

    parser.add_argument("--output_skip", type=int, default=1)
    args = parser.parse_args()

    p3ht_pcbm_dynamics(
        args.ansatz,
        args.chi,
        chi2=args.chi2,
        tmax=args.tmax * fs,
        dt=args.dt * fs,
        use_multiset=args.use_multiset,
        adaptive=args.adaptive,
        nunoccupied=args.nunoccupied,
        spawning_threshold=args.spawning_threshold,
        unoccupied_threshold=args.unoccupied_threshold,
        output_skip=args.output_skip,
        ofname=args.fname,
    )
