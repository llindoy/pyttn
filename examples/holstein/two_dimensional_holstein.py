import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
import h5py
import argparse

from pyttn import ntreeBuilder
from pyttn import system_modes, boson_mode
from pyttn import matrix_element, tdvp, sOP
from pyttn import ms_ttn, ms_sop_operator, multiset_SOP

import matplotlib.pyplot as plt


def gen_lattice(N):
    res = np.zeros((N, N, 3), dtype=int)
    for i in range(N):
        for j in range(N):
            res[i, j, :] = np.array([i, j, i * N + j], dtype=int)
    return res


def partition_points_x(x):
    if x.shape[0] % 2 == 0:
        skip = x.shape[0] // 2
    else:
        skip = (x.shape[0] + 1) // 2
    return x[:skip, :], x[skip:, :]


def partition_points_y(x):
    if x.shape[1] % 2 == 1:
        skip = x.shape[1] // 2
    else:
        skip = (x.shape[1] + 1) // 2
    return x[:, :skip], x[:, skip:]


def partition_and_get_path(x, path=None):
    if path is None:
        path = []

    lists = []

    # partition the two sets of points
    if x.shape[0] > 1:
        xs = [*partition_points_x(x)]
    elif x.shape[1] > 1:
        xs = [*partition_points_y(x)]
    else:
        return [[path, x[0, 0, 2]]]

    for i in range(2):
        curr_path = path + [i]
        partition = False

        if xs[i].shape[1] > 1:
            x2s = [*partition_points_y(xs[i])]
            partition = True
        elif xs[i].shape[0] > 1:
            x2s = [*partition_points_x(xs[i])]
            partition = True
        else:
            lists.append([curr_path, xs[i][0, 0, 2]])

        if partition:
            for j in range(2):
                fpath = curr_path + [j]
                lists = lists + partition_and_get_path(x2s[j], fpath)

    return lists


def plot_points(x):
    for xi in x:
        plt.scatter(xi[:, :, 0], xi[:, :, 1], 2 + xi[:, :, 2] / 10)


def tree_index_to_site_index(lists):
    return [int(li[1]) for li in lists]


def invert_indexing(inds):
    res = [0 for i in inds]
    for i in range(len(inds)):
        res[inds[i]] = i
    return res


def expand_nodes(inds, N):
    res = []
    for i in inds:
        for j in range(N):
            res.append(i * N + j)
    return res


def run_initial_step(A, h, sweep, dt, nstep=10):
    tp = 0
    ts = np.logspace(np.log10(dt * 1e-5), np.log10(dt), nstep)
    for i in range(nstep):
        dti = ts[i] - tp
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    return A, h, sweep


def output_results(ofname, timepoints, res, runtime):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=timepoints)

    N2 = res.shape[1]
    N = int(np.sqrt(N2))
    for i in range(N):
        for j in range(N):
            h5.create_dataset("|%d,%d><%d,%d|" % (i, j, i, j), data=res[:, i * N + j])
    h5.create_dataset("runtime", data=runtime * np.ones(1))
    h5.close()


def holstein_dynamics(g, w0, J, N, chi1, chi2, beta=None, tmax=200, dt=0.25, nbose=10, ofname="holstein_1d.h5", output_skip=1):
    # set up the time evolution parameters
    nsteps = int(tmax / (dt)) + 1

    # set up the system information
    sysinf = system_modes(N * N)
    for i in range(N * N):
        if beta is None:
            sysinf[i] = boson_mode(nbose)
        else:
            sysinf[i] = [boson_mode(nbose), boson_mode(nbose)]

    # set up the Hamiltonian
    H = multiset_SOP(N * N, N * N)

    # add on the electronic coupling terms
    for i in range(N):
        for j in range(N):
            i0 = i * N + j
            ip = ((i + 1) % N) * N + j
            jp = i * N + ((j + 1) % N)

            # add on the x hopping terms
            H[i0, ip] += J
            H[ip, i0] += J

            # add on the y hopping terms
            H[i0, jp] += J
            H[jp, i0] += J

    # add on the purely bosonic terms
    for i in range(N * N):
        for j in range(N * N):
            if beta is None:
                H[i, i] += w0 * sOP("n", j)
            else:
                H[i, i] += w0 * sOP("n", 2 * j)
                H[i, i] += -w0 * sOP("n", 2 * j + 1)

    # now add on the system bath coupling terms
    for i in range(N * N):
        if beta is None:
            H[i, i] += g * (sOP("adag", i) + sOP("a", i))
        else:
            gp = g * (0.5 * (1 + 1.0 / np.tanh(beta * w0 / 2)))
            gm = g * (0.5 * (1 + 1.0 / np.tanh(-beta * w0 / 2)))
            H[i, i] += gp * (sOP("adag", 2 * i) + sOP("a", 2 * i))
            H[i, i] += gm * (sOP("adag", 2 * i + 1) + sOP("a", 2 * i + 1))

    # build the ML-MCTDH tree for the vibrational modes
    class chi_step:
        def __init__(self, chimax, chimin, N, degree=2):
            self.chimin = chimin
            if N % degree == 0:
                self.Nl = int(int(np.log(N) / np.log(degree)))
            else:
                self.Nl = int(int(np.log(N) / np.log(degree)) + 1)

            self.nx = int((chimax - chimin) // (self.Nl - 1))

        def __call__(self, li):
            ret = int((self.Nl - li) * self.nx + self.chimin)
            return ret

    topo = ntreeBuilder.mlmctdh_tree(sysinf.mode_dimensions(), 2, chi_step(chi1, chi2, N * N))

    # generate the mapping from square lattice to tree indices
    x = gen_lattice(N)
    lists = partition_and_get_path(x)
    conv = tree_index_to_site_index(lists)
    site_to_tree = invert_indexing(conv)
    sysinf.mode_indices = site_to_tree

    # set up the wavefunction for simulating the dynamics
    A = ms_ttn(N * N, topo, dtype=np.complex128)
    state = [[0 for i in range(N * N)] for j in range(N * N)]
    coeff = np.zeros(N * N, dtype=np.float64)
    coeff[(N // 2) * (N + 1)] = 1
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
    res = np.zeros((nsteps + 1, N * N), dtype=np.complex128)

    for j in range(N * N):
        res[0, j] = mel(A.slice(j))

    t1 = time.time()

    # perform a set of steps with a logarithmic timestep discretisation
    A, h, sweep = run_initial_step(A, h, sweep, dt)

    for j in range(N * N):
        res[1, j] = mel(A.slice(j))

    for j in range(N * N):
        res[1, j] = mel(A.slice(j))
    sweep.dt = dt

    timepoints = np.arange(nsteps + 1) * dt

    # perform the remaining dynamics steps
    for i in range(1, nsteps):
        print(i, nsteps, flush=True)
        sweep.step(A, h)

        t2 = time.time()

        for j in range(N * N):
            res[i + 1, j] = mel(A.slice(j))

        if i % output_skip == 0:
            output_results(ofname, timepoints, res, (t2 - t1))

    t2 = time.time()
    output_results(ofname, timepoints, res, (t2 - t1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyttn test")

    parser.add_argument("g", type=float)
    parser.add_argument("w0", type=float)
    parser.add_argument("J", type=float)
    parser.add_argument("N", type=int)

    parser.add_argument("chi", type=int)
    parser.add_argument("--chi2", type=int, default=None)
    parser.add_argument("--nbose", type=int, default=8)
    parser.add_argument("--beta", type=float, default=None)

    parser.add_argument("--fname", type=str, default="holstein_2d.h5")

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=3)

    parser.add_argument("--output_skip", type=int, default=10)
    args = parser.parse_args()

    chi2 = args.chi2
    if chi2 is None:
        chi2 = args.chi

    holstein_dynamics(
        args.g,
        args.w0,
        args.J,
        args.N,
        args.chi,
        nbose=args.nbose,
        beta=args.beta,
        chi2=chi2,
        tmax=args.tmax,
        dt=args.dt,
        output_skip=args.output_skip,
        ofname=args.fname,
    )
