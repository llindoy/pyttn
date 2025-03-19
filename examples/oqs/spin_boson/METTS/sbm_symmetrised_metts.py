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
import time
import h5py
import copy
import argparse

import pyttn
from pyttn import oqs
from pyttn.oqs.bath_fitting import softmspace

from numba import jit

def evolve_imaginary_time_two(A, B, h, mel, sweep, sweepB, betasteps):
    beta_p = 0
    rho0 = 1.0
    for i in range(betasteps.shape[0]):
        sweep.dt = betasteps[i] - beta_p
        beta_p = betasteps[i]
        sweep.step(A, h)
        rho = A.normalise()
        rhoB = B.normalise()
        rho0 *= rhoB / rho0
        print("imstep:",i, A.maximum_bond_dimension(), end='                  \r', flush=True)
    return rho0


def evolve_imaginary_time(A, h, mel, sweep, betasteps):
    beta_p = 0
    rho0 = 1.0
    for i in range(betasteps.shape[0]):
        sweep.dt = betasteps[i] - beta_p
        beta_p = betasteps[i]
        sweep.step(A, h)
        rho = A.normalise()
        rho0 *= rho
        print("imstep:",i, A.maximum_bond_dimension(), end='                  \r', flush=True)

# need to fix this
def sigma(w, ij, eps, kappa):
    if kappa * np.abs(w * ij) < eps:
        return kappa * w * ij
    else:
        return eps


def measurement_basis(w, eps, kappa, n):
    res = np.exp(1.0j * np.random.uniform(0, 2 * np.pi, size=(n, n)))
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i, j] *= np.exp(
                    -((w * (i - j) / sigma(w, (i - j), eps, kappa)) ** 2)
                )
    Q, P = np.linalg.qr(res, mode="complete")
    return Q


def Ct(t, w, g):
    T, W = np.meshgrid(t, w)
    g2 = g**2
    fourier = np.exp(-1.0j * W * T)
    return g2 @ fourier


def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta=5, nbeta=100, nsamples=256, Ncut=50, nstep=1, 
                 Nw=9, ofname="sbm.h5", degree=2, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):

    t = np.arange(nstep + 1) * dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(np.abs(w)/wc))

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=None)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc) )

    #set up the total Hamiltonian
    N = Nb+1
    H = pyttn.SOP(N)
    Hb = pyttn.SOP(N)

    #and add on the system parts
    H += eps/2*pyttn.sOP("sz", 0)
    H += delta/2*pyttn.sOP("sx", 0)

    Hb += eps/2*pyttn.sOP("sz", 0)
    Hb += delta/2*pyttn.sOP("sx", 0)

    #now add on the bath bits getting additionally getting a frequency parameter that can be used in energy based
    #truncation schemes
    H = oqs.unitary.add_bosonic_bath_hamiltonian(H, pyttn.sOP("sz", 0), g, w, geom="star")
    Hb= oqs.unitary.add_bosonic_bath_hamiltonian(Hb, pyttn.sOP("sz", 0), g, w, geom="star")

    #mode_dims = [nbose for i in range(Nb)]
    mode_dims = [min(max(4, int(wc*Ncut/w[i])), nbose) for i in range(Nb)]

    #setup the system information object
    sysinf = pyttn.system_modes(N)
    sysinf[0] = pyttn.tls_mode()
    for i in range(Nb):
        sysinf[i+1] = pyttn.boson_mode(mode_dims[i])

    #construct the topology and capacity trees used for constructing 
    chi0 = 4

    #and add the node that forms the root of the bath.  
    topo = pyttn.ntree("(1(2(2))(2))")
    if(degree > 1):
        pyttn.ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi0)
    else:
        pyttn.ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi0, min(chi0, nbose))
    pyttn.ntreeBuilder.sanitise(topo)

    capacity = pyttn.ntree("(1(2(2))(2))")
    if(degree > 1):
        pyttn.ntreeBuilder.mlmctdh_subtree(capacity()[1], mode_dims, degree, chi)
    else:
        pyttn.ntreeBuilder.mps_subtree(capacity()[1], mode_dims, chi, min(chi, nbose))
    pyttn.ntreeBuilder.sanitise(capacity)

    A = pyttn.ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for x in range(N)])

    #build the mixing basis operators for each bosonic mode
    Uproj=[]
    #append sigma_x basis for the spin degree of freedom
    Uproj.append(np.array([[0, 1], [1, 0]], dtype=np.complex128))

    eps = 2
    kappa = 1

    for i in range(Nb):
        Uproj.append(measurement_basis(w[i], eps, kappa, mode_dims[i]))

    h = pyttn.sop_operator(H, A, sysinf, identity_opt=True, compress=True)
    hb = pyttn.sop_operator(Hb, A, sysinf, identity_opt=True, compress=True)

    mel = pyttn.matrix_element(A)

    op = pyttn.site_operator(pyttn.sOP("sz", 0), sysinf)

    sweep = None
    sweepB = None

    B = copy.deepcopy(A)

    sweep_therm = pyttn.tdvp(A, hb, krylov_dim=12, expansion="subspace")
    sweep_therm.spawning_threshold = spawning_threshold
    sweep_therm.unoccupied_threshold = unoccupied_threshold
    sweep_therm.minimum_unoccupied = nunoccupied

    sweep_thermB = pyttn.tdvp(B, hb, krylov_dim=12, expansion="subspace")
    sweep_thermB.spawning_threshold = spawning_threshold
    sweep_thermB.unoccupied_threshold = unoccupied_threshold
    sweep_thermB.minimum_unoccupied = nunoccupied

    sweep = pyttn.tdvp(A, h, krylov_dim=12, expansion="subspace")
    sweep.spawning_threshold = spawning_threshold
    sweep.unoccupied_threshold = unoccupied_threshold
    sweep.minimum_unoccupied = nunoccupied

    sweepB = pyttn.tdvp(B, h, krylov_dim=12, expansion="subspace")
    sweepB.spawning_threshold = spawning_threshold
    sweepB.unoccupied_threshold = unoccupied_threshold
    sweepB.minimum_unoccupied = nunoccupied

    res = np.zeros((nsamples, nstep + 1), dtype=np.complex128)

    beta_steps = softmspace(1e-6, beta / 2.0, nbeta)
    nwarmup = 4
    for i in range(nwarmup):
        print("warmup step:", i, flush=True)
        sweep_therm.dt = dt
        sweep_therm.coefficient = -1.0

        sweep_therm.t = 0
        print(A.collapse_basis(Uproj, nchi=2), mode_dims, flush=True)
        A.normalise()
        sweep_therm.prepare_environment(A, hb)
        evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)
        sweep_therm.t = 0

        print(A.collapse(nchi=2), mode_dims, flush=True)
        A.normalise()
        sweep_therm.prepare_environment(A, hb)
        evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)

    for sample in range(nsamples):
        print("samples step:", sample, flush=True)
        sweep_therm.t = 0

        sweep_therm.coefficient = -1.0
        for iter in range(1):
            sweep.t = 0
            print(A.collapse_basis(Uproj, nchi=2), mode_dims)
            A.normalise()
            sweep_therm.prepare_environment(A, hb)
            evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)

            sweep.t = 0
            print(A.collapse(nchi=2), mode_dims)
            A.normalise()
            sweep_therm.prepare_environment(A, hb)
            evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)

        sweep.t = 0
        print(A.collapse_basis(Uproj, nchi=2), mode_dims)
        A.normalise()
        sweep_therm.prepare_environment(A, hb)
        evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)

        sweep.t = 0
        print(A.collapse(nchi=2), mode_dims)

        B = copy.deepcopy(A)
        B.apply_one_body_operator(op)

        sweep_therm.prepare_environment(A, hb)
        sweep_thermB.prepare_environment(B, hb)
        coeff = evolve_imaginary_time_two(
            A, B, hb, mel, sweep_therm, sweep_thermB, beta_steps
        )

        C = copy.deepcopy(A)

        sweep.dt = dt
        sweep.coefficient = -1.0j
        sweepB.dt = dt
        sweepB.coefficient = -1.0j

        sweep.t = 0
        sweepB.t = 0

        sweep.prepare_environment(A, h)
        sweepB.prepare_environment(B, h)

        res[sample, 0] = np.real(mel(op, A, B)) * coeff
        for i in range(nstep):
            t1 = time.time()
            sweep.step(A, h)
            sweepB.step(B, h)
            t2 = time.time()
            res[sample, i + 1] = np.real(mel(op, B, A)) * coeff
            if i % 100 == 0:
                print((i+1)*dt, res[sample, i+1], t2-t1, A.maximum_bond_dimension(), end = '              \r', flush=True)

        h5 = h5py.File(ofname, "w")
        h5.create_dataset("t", data=(np.arange(nstep + 1) * dt))
        h5.create_dataset("Sz", data=res)
        h5.close()

        A = copy.deepcopy(C)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamics of the zero temperature spin boson model with"
    )

    # exponential bath cutoff parameters
    parser.add_argument("alpha", type=float)
    parser.add_argument("--wc", type=float, default=5)
    parser.add_argument("--s", type=float, default=1)

    # number of bath modes
    parser.add_argument("--N", type=int, default=128)

    # system hamiltonian parameters
    parser.add_argument("--eps", type=float, default=0)
    parser.add_argument("--delta", type=float, default=1)

    # bath inverse temperature
    parser.add_argument("--beta", type=float, default=5)

    # maximum bond dimension
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--degree", type=int, default=1)

    # maximum bosonic hilbert space dimension
    parser.add_argument("--nbose", type=int, default=200)

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--tmax", type=float, default=10)

    # output file name
    parser.add_argument("--fname", type=str, default="sbm_thermal.h5")

    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--nbeta", type=int, default=100)

    # the minimum number of unoccupied modes for the dynamics
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-5)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax / args.dt) + 1
    sbm_dynamics(
        args.N,
        args.alpha,
        args.wc,
        args.s,
        args.eps,
        args.delta,
        args.chi,
        args.nbose,
        args.dt,
        beta=args.beta,
        nstep=nstep,
        ofname=args.fname,
        nunoccupied=args.nunoccupied,
        spawning_threshold=args.spawning_threshold,
        unoccupied_threshold=args.unoccupied_threshold,
        degree=args.degree,
        nsamples=args.nsamples,
        nbeta=args.nbeta,
    )
