import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs
from pyttn.oqs.heom import softmspace

from numba import jit


def evolve_imaginary_time(A, h, mel, sweep, betasteps):
    beta_p = 0
    rho0 = 1.0
    for i in range(betasteps.shape[0]):
        beta_p = betasteps[i]
        sweep.step(A, h)
        rho = A.normalise()
        rho0 *= rho
        #print("imstep:",i, A.maximum_bond_dimension())
        sys.stdout.flush()

#need to fix this
def sigma(w, ij, eps, kappa):
    if (kappa*np.abs(w*ij) < eps):
        return kappa*w*ij
    else:
        return w

def measurement_basis(w, eps, kappa, n):
    res = np.exp(1.0j*np.random.uniform(0, 2*np.pi, size=(n, n)))
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i, j] *= np.exp(-(w*(i-j)/sigma(w, (i-j), eps, kappa))**2)
    Q, P = np.linalg.qr(res, mode='complete')
    return Q


def Ct(t, w, g):
    T, W = np.meshgrid(t, w)
    g2 = g**2
    fourier = np.exp(-1.0j*W*T)
    return g2@fourier

def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = 5, nbeta = 100, nsamples=256, Ncut = 50, nstep = 1, Nw = 7.5, geom='star', ofname='sbm.h5', degree = 2, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(np.abs(w)/wc))

    #set up the open quantum system bath object
    bath = oqs.bosonic_bath(J, sOP("sz", 0), beta=None)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    g,w = bath.discretise(Nb, Nw*wc, method='orthopol')

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(N)
    Hb = SOP(N)

    #and add on the system parts
    H += eps*sOP("sz", 0)
    H += 2*delta*sOP("sx", 0)

    Hb += eps*sOP("sz", 0)
    Hb += 2*delta*sOP("sx", 0)

    #now add on the bath bits getting additionally getting a frequency parameter that can be used in energy based
    #truncation schemes
    H, l = oqs.add_bath_hamiltonian(H, bath.Sp, 2*g, w, geom=geom)
    if(geom == 'ipchain'):
        Hb, w = oqs.add_bath_hamiltonian(Hb, bath.Sp, 2*g, w, geom='chain')
    else:
        Hb, w = oqs.add_bath_hamiltonian(Hb, bath.Sp, 2*g, w, geom=geom)
    w=l

    #mode_dims = [nbose for i in range(Nb)]
    mode_dims = [min(max(4, int(wc*Ncut/w[i])), nbose) for i in range(Nb)]

    #setup the system information object
    sysinf = system_modes(N)
    sysinf[0] = spin_mode(2)
    for i in range(Nb):
        sysinf[i+1] = boson_mode(mode_dims[i])

    #construct the topology and capacity trees used for constructing 
    chi0 = 2

    #and add the node that forms the root of the bath.  
    #TODO: Add some better functions for handling the construction of tree structures
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi0)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi0, min(chi0, nbose))
    ntreeBuilder.sanitise(topo)

    capacity = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(capacity()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(capacity()[1], mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(capacity)

    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for x in range(N)])
    #A.random()


    #build the mixing basis operators for each bosonic mode
    Uproj=[]
    #append sigma_x basis for the spin degree of freedom
    Uproj.append(np.array([[0, 1], [1, 0]], dtype=np.complex128))

    if geom == 'star':
        eps = 4
        kappa = 0.1
    else:
        eps = 1.2*np.amin(w)
        kappa = 1/np.amin(w)
    for i in range(Nb):
        Uproj.append(measurement_basis(w[i], eps, kappa, mode_dims[i]))

    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)
    hb = sop_operator(Hb, A, sysinf, identity_opt=True, compress=True)

    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )

    sweep = None
    sweepB = None

    B = copy.deepcopy(A)

    sweep_therm = tdvp(A, hb, krylov_dim = 12, expansion='subspace')
    sweep_therm.spawning_threshold = spawning_threshold
    sweep_therm.unoccupied_threshold=unoccupied_threshold
    sweep_therm.minimum_unoccupied=nunoccupied

    sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace')
    sweep.spawning_threshold = spawning_threshold
    sweep.unoccupied_threshold=unoccupied_threshold
    sweep.minimum_unoccupied=nunoccupied

    sweepB = tdvp(B, h, krylov_dim = 12, expansion='subspace')
    sweepB.spawning_threshold = spawning_threshold
    sweepB.unoccupied_threshold=unoccupied_threshold
    sweepB.minimum_unoccupied=nunoccupied

    if(geom == 'ipchain'):
        sweep.use_time_dependent_hamiltonian = False

    res = np.zeros((nsamples, nstep+1), dtype=np.complex128)

    beta_steps = softmspace(1e-6, beta/2.0, nbeta)
    nwarmup=4
    for i in range(nwarmup):
        print("warmup step:", i)
        sys.stdout.flush()
        sweep_therm.dt = dt
        sweep_therm.coefficient = -1.0

        if(geom == 'ipchain'):
            sweep_therm.use_time_dependent_hamiltonian = False

        sweep_therm.t=0
        print(A.collapse_basis(Uproj, nchi=2), mode_dims)
        A.normalise()
        sweep_therm.prepare_environment(A, hb)
        evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)
        sweep_therm.t=0

        print(A.collapse(nchi=2), mode_dims)
        A.normalise()
        sweep_therm.prepare_environment(A, hb)
        evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)

    for sample in range(nsamples):
        print("samples step:", sample)
        sys.stdout.flush()

        sweep_therm.t=0
        
        if(geom == 'ipchain'):
            sweep_therm.use_time_dependent_hamiltonian = False
        sweep_therm.coefficient = -1.0
        for iter in range(2):
            sweep.t=0
            print(A.collapse_basis(Uproj, nchi=2), mode_dims)
            A.normalise()
            sweep_therm.prepare_environment(A, hb)
            evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)

            sweep.t=0
            print(A.collapse(nchi=2), mode_dims)
            A.normalise()
            sweep_therm.prepare_environment(A, hb)
            evolve_imaginary_time(A, hb, mel, sweep_therm, beta_steps)

        C = copy.deepcopy(A)
        B = copy.deepcopy(A)
        B.apply_one_body_operator(op)

        if(geom == 'ipchain'):
            sweep.use_time_dependent_hamiltonian = True
            sweepB.use_time_dependent_hamiltonian = True

        sweep.dt = dt
        sweep.coefficient = -1.0j
        sweepB.dt = dt
        sweepB.coefficient = -1.0j

        sweep.t=0
        sweepB.t=0

        sweep.prepare_environment(A, h)
        sweepB.prepare_environment(B, h)

        res[sample, 0] = np.real(mel(op, A, B))
        for i in range(nstep):
            t1 = time.time()
            sweep.step(A, h)
            sweepB.step(B, h)
            t2 = time.time()
            res[sample, i+1] = np.real(mel(op, B, A))
            if i % 100 == 0:
                print((i+1)*dt, res[sample, i+1], t2-t1, A.maximum_bond_dimension())
            sys.stdout.flush()

        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
        h5.create_dataset('Sz', data=res)
        h5.close()

        A = copy.deepcopy(C)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('alpha', type = float)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)

    #number of bath modes
    parser.add_argument('--N', type=int, default=48)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--eps', type = float, default=0)
    parser.add_argument('--delta', type = float, default=1)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=5)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--degree', type=int, default=1)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=200)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=10)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm_thermal.h5')

    parser.add_argument('--nsamples', type=int, default = 256)
    parser.add_argument('--nbeta', type=int, default = 100)

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, degree = args.degree, nsamples=args.nsamples, nbeta=args.nbeta)
