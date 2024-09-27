import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../../")
sys.path.append("..")
from pyttn import *

from pyttn.utils import orthopol_discretisation
from numba import jit
from chain_map import chain_map


def discretise_bath(Nb, alpha, wc, s, beta = None, Nw = 10, moment_scaling=2, atol=0, rtol=1e-10):
    #the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #the frequency dependent bath spectral function
    @jit(nopython=True)
    def S(w):
        if beta == None:
            return J(w)*np.where(w > 0, 1.0, 0.0)
        else:
            return J(w)*0.5*(1.0+1.0/np.tanh(beta*w/2.0))

    wmax = Nw*wc
    wmin = 0
    if beta != None:
        wmin = -wmax/(beta*wc+1)

    #discretise the bath using the orthopol discretisation strategy
    g, w = orthopol_discretisation.discretise( lambda x : S(x), wmin, wmax, Nb, moment_scaling=moment_scaling, atol=atol, rtol=rtol)

    #and compute the polaron transformed renormalisation of the truncated high frequency modes
    renorm = np.exp(-2.0/np.pi*scipy.integrate.quad(lambda x : J(x)/(x*x), wmax, np.inf)[0])

    return np.array(g), np.array(w), renorm

#compute the bath correlation function from the discretised frequencies
def Ct(t, w, g):
    T, W = np.meshgrid(t, w)
    g2 = g**2
    fourier = np.exp(-1.0j*W*T)
    return g2@fourier


#setup the star Hamiltonian for the spin boson model
def setup_star_hamiltonian(eps, delta, g, w, Nb):
    N = Nb+1

    H = SOP(N)
    H += eps*sOP("sz", 0)
    H += delta*sOP("sx", 0)
    for i in range(Nb):
        H += np.sqrt(2)*g[i] * sOP("sz", 0)  * sOP("q", i+1)
        H += w[i] * sOP("n", i+1)

    return H, w


import matplotlib.pyplot as plt

def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, Ncut = 20, nstep = 1, Nw = 7.5, ofname='sbm.h5', adaptive=False, spawning_threshold=1e-5, unoccupied_threshold=1e-4):
    g, w, renorm = discretise_bath(Nb, alpha, wc, s, beta=None, Nw=Nw)
    H = None

    H, l = setup_star_hamiltonian(eps, delta*renorm, 2*g, w, Nb)

    mode_dims = [min(max(4, int(wc*Ncut/l[i])), nbose) for i in range(Nb)]
    N = Nb+1
    sysinf = system_modes(N)
    sysinf[0] = spin_mode(2)
    for i in range(Nb):
        sysinf[i+1] = boson_mode(mode_dims[i])


    degree = 2

    chi0 = chi
    if adaptive:
        chi0 = 2
    #and add the node that forms the root of the bath
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi0)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, degree, chi0)
    ntreeBuilder.sanitise(topo)

    capacity = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(capacity()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(capacity()[1], mode_dims, degree, chi)
    ntreeBuilder.sanitise(capacity)

    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(Nb+1)])

    h = sop_operator(H, A, sysinf)

    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )

    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 12)
    else:
        sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace')
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=1

    sweep.dt = dt
    sweep.coefficient = -1.0j
    sweep.prepare_environment(A, h)

    res = np.zeros(nstep+1)
    runtime = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)

    res[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        res[i+1] = np.real(mel(op, A, A))
        runtime[i+1] = runtime[i]+(t2-t1)
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt, t2-t1, res[i+1], mel(A), maxchi[i+1])
        sys.stdout.flush()

        if(i % 100):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            h5.create_dataset('Sz', data=res)
            h5.create_dataset('runtime', data=runtime)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    h5.create_dataset('Sz', data=res)
    h5.create_dataset('runtime', data=runtime)
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')
    parser.add_argument('--alpha', type = float, default=1)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)
    parser.add_argument('--delta', type = float, default=0)
    parser.add_argument('--eps', type = float, default=1)
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--nbose', type=int, default=20)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--spawningthreshold', type = float, default=1e-5)
    parser.add_argument('--unoccupiedthreshold', type = float, default=1e-4)
    args = parser.parse_args()

    nstep = int(5.0/args.dt)+1
    ofname = 'sbm_subspace_expansion_'+str(args.chi)+"_"+str(args.spawningthreshold)+"_"+str(args.unoccupiedthreshold)+'.h5'
    sbm_dynamics(128, args.alpha, args.wc, args.s, args.delta, args.eps, args.chi, args.nbose, args.dt, nstep = nstep, spawning_threshold=args.spawningthreshold, unoccupied_threshold=args.unoccupiedthreshold, ofname = ofname, adaptive=True)
    #sbm_dynamics(128, args.alpha, args.wc, args.s, args.delta, args.eps, args.chi, args.nbose, args.dt, nstep = nstep, ofname = 'sbm_single_site.h5')
