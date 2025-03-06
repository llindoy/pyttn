import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../../")
from pyttn import *
from pyttn import oqs, utils
from numba import jit

def build_topology(Ns, ds, chiS,  chi, nbose, b_mode_dims, degree):
    lchi = [chiS for i in range(Ns)]
    topo = ntreeBuilder.mps_tree(lchi, chiS)

    leaf_indices=topo.leaf_indices()
    for li in leaf_indices:
        topo.at(li).insert(ds)
        if degree == 1:
            ntreeBuilder.mps_subtree(topo.at(li), b_mode_dims, chi, min(chi, nbose))
        else:
            ntreeBuilder.mlmctdh_subtree(topo.at(li), b_mode_dims, degree, chi)

    ntreeBuilder.sanitise(topo)
    return topo

def xychain_dynamics(Ns, Nb, alpha, wc, eta, chiS, chi, nbose, dt, nbose_min = None, beta = None, nstep = 1, Nw=4.0, geom='ipchain', ofname='xychain.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, use_mode_combination=True, nbmax=2, nhilbmax=1024):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(2*np.pi*alpha*w*np.exp(-np.abs(w/wc))**2)*np.where(w > 0, 1.0, -1.0)


    import matplotlib.pyplot as plt

    Nbs = [64, 128, 180]

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)
    plt.rcParams.update({'font.size':22})
    fig, ax = plt.subplots(2, 2, figsize=(3.25*2.5, 3.25*2), sharey=False, sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    Ctval = bath.Ct(t)
    ax[0,0].plot(t, Ctval, 'k', linewidth=5, label='Exact', zorder=-1)

    colors = ['red', 'blue', 'green']
    for i, Nb in enumerate(Nbs):
        #discretise the bath correleation function using the orthonormal polynomial based cutoff 
        g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))
        ax[0, 0].plot(t, oqs.BosonicBath.Ctexp(t, g*g, w), '--', color=colors[i], linewidth=3, label=r'$N_{b}='+str(Nb)+'$')
    ax[0, 0].set_xlim([0, 40])
    ax[0, 0].legend(frameon=False, prop={'size':16}, labelspacing=0)
    plt.show()

    Ks = [1, 2, 4]

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)
    ax[0, 1].plot(t, Ctval, 'k', linewidth=5, label='Exact', zorder=-1)

    colors = ['red', 'blue', 'green']
    for i, K in enumerate(Ks):
        #discretise the bath correleation function using the orthonormal polynomial based cutoff 
        dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep*dt, Nt = nstep))
        ax[0,1].plot(t, oqs.BosonicBath.Ctexp(t, dk, -1.0j*zk), '--', color=colors[i], linewidth=3, label=r'$K='+str(K)+'$')
    ax[0,0].set_xlim([0, 40])
    ax[0,1].set_xlim([0, 40])
    ax[0,0].set_ylim([-1, np.max(np.abs(Ctval))])
    ax[0,1].set_ylim([-1, np.max(np.abs(Ctval))])
    ax[0,1].legend(frameon=False, prop={'size':16}, labelspacing=0)


    Nbs = [100, 200, 300]

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=0.625)

    Ctval = bath.Ct(t)
    ax[1,0].plot(t, Ctval, 'k', linewidth=5, label='Exact', zorder=-1)

    colors = ['red', 'blue', 'green']
    for i, Nb in enumerate(Nbs):
        #discretise the bath correleation function using the orthonormal polynomial based cutoff 
        g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))
        ax[1, 0].plot(t, oqs.BosonicBath.Ctexp(t, g*g, w), '--', color=colors[i], linewidth=3, label=r'$N_{b}='+str(Nb)+'$')
    ax[1, 0].set_xlim([0, 40])
    ax[1, 0].legend(frameon=False, prop={'size':16}, labelspacing=0)

    Ks = [1, 2, 4]

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=0.625)
    ax[1,1].plot(t, Ctval, 'k', linewidth=5, label='Exact', zorder=-1)

    colors = ['red', 'blue', 'green']
    for i, K in enumerate(Ks):
        #discretise the bath correleation function using the orthonormal polynomial based cutoff 
        dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep*dt, Nt = nstep))
        ax[1,1].plot(t, oqs.BosonicBath.Ctexp(t, dk, -1.0j*zk), '--', color=colors[i], linewidth=3, label=r'$K='+str(K)+'$')
    ax[1,0].set_xlim([0, 40])
    ax[1,1].set_xlim([0, 40])
    ax[1,0].set_ylim([-1, np.max(np.abs(Ctval))])
    ax[1,1].set_ylim([-1, np.max(np.abs(Ctval))])
    ax[1,1].legend(frameon=False, prop={'size':16}, labelspacing=0)

    ax[0, 0].set_xticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_yticklabels([])
    ax[1, 1].set_yticklabels([])
    ax[1, 0].set_xticks([0, 10, 20, 30])
    ax[1, 1].set_xticks([0, 10, 20, 30, 40])

    fig.text(0.5, 0.01, r'$t$', ha='center')
    fig.text(0.01, 0.5, r'$C(t)$', va='center', rotation='vertical')


    plt.savefig('dissipative_xy_bath_discretisation_convergence.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    parser.add_argument('--N', type=int, default=80)
    #number of spins in the system
    parser.add_argument('--Ns', type=int, default=21)

    #exponential bath cutoff parameters
    parser.add_argument('--alpha', type = float, default=0.32)
    parser.add_argument('--wc', type = float, default=4)

    #number of bath modes
    parser.add_argument('--geom', type = str, default='ipchain')

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)
    parser.add_argument('--nbose_min', type=int, default=5)

    #mode combination parameters
    parser.add_argument('--nbmax', type=int, default=4)
    parser.add_argument('--nhilbmax', type=int, default=1000)


    #system hamiltonian parameters
    parser.add_argument('--eta', type = float, default=0.04)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=32)
    parser.add_argument('--chiS', type=int, default=64)
    parser.add_argument('--degree', type=int, default=1)


    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.025)
    parser.add_argument('--tmax', type=float, default=40)

    #output file name
    parser.add_argument('--fname', type=str, default='xychain.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=1)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1

    xychain_dynamics(args.Ns, args.N, args.alpha, args.wc, args.eta, args.chiS, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, nbose_min=args.nbose_min, use_mode_combination=True, nbmax=args.nbmax, nhilbmax=args.nhilbmax)

