import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
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


#setup the chain hamiltonian for the spin boson model - this is the tedopa method
def setup_chain_hamiltonian(eps, delta, g, w, Nb):
    N = Nb+1

    t, e = chain_map(g, w)

    H = SOP(N)
    H += eps*sOP("sz", 0)
    H += delta*sOP("sx", 0)

    for i in range(Nb):
        if i == 0:
            H += np.sqrt(2)*t[i]*sOP("sz", 0) * sOP("q", i+1)
        else:
            H += t[i]*sOP("adag", i)*sOP("a", i+1)  
            H += t[i]*sOP("a", i)*sOP("adag", i+1) 
        H += e[i] * sOP("n", i+1)

    return H, e


#setup the chain hamiltonian for the spin boson model - that is this implements the method described in Nuomin, Beratan, Zhang, Phys. Rev. A 105, 032406
def setup_ipchain_hamiltonian(eps, delta, g, w, Nb):
    N = Nb+1

    t, e, U = chain_map(g, w, return_unitary = True)
    t0 = t[0]

    l = w
    P = U

    H = SOP(N)
    H += eps*sOP("sz", 0)
    H += delta*sOP("sx", 0)

    class func_class:
        def __init__(self, i, t0, e0, U0, conj = False):
            self.i = i
            self.conj=conj
            self.t0 = t0
            self.e = copy.deepcopy(e0)
            self.U = copy.deepcopy(U0)

        def __call__(self, ti):
            val = self.t0*np.conj(self.U[:, 0])@(np.exp(-1.0j*ti*self.e)*self.U[:, self.i])

            if(self.conj):
                val = np.conj(val)

            return val

    for i in range(Nb):
        H += coeff(func_class(i, t0, l, P, conj=False))*sOP("sz", 0)*sOP("a", i+1) 
        H += coeff(func_class(i, t0, l, P, conj=True ))*sOP("sz", 0)*sOP("adag", i+1)  

    return H, e



import matplotlib.pyplot as plt


def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, Ncut = 20, nstep = 1, Nw = 7.5, geom='star', ofname='sbm.h5'):
    g, w, renorm = discretise_bath(Nb, alpha, wc, s, beta=None, Nw=Nw)


    H = None

    if geom  == 'chain':
        H, l = setup_chain_hamiltonian(eps, delta*renorm, 2*g, w, Nb)
    elif geom == 'ipchain':
        H, l = setup_ipchain_hamiltonian(eps, delta*renorm, 2*g, w, Nb)
    else:
        H, l = setup_star_hamiltonian(eps, delta*renorm, 2*g, w, Nb)

    for t in H:
        print(t)

    mode_dims = [nbose for i in range(Nb)]
    #mode_dims = [min(max(4, int(wc*Ncut/l[i])), nbose) for i in range(Nb)]
    N = Nb+1
    sysinf = system_modes(N)
    sysinf[0] = spin_mode(2)
    for i in range(Nb):
        sysinf[i+1] = boson_mode(mode_dims[i])


    degree = 1
    #and add the node that forms the root of the bath
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, degree, nbose)
    ntreeBuilder.sanitise(topo)

    A = ttn(topo, dtype=np.complex128)
    A.set_state([0 for i in range(Nb+1)])
    print(topo)

    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)

    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )

    sweep = tdvp(A, h, krylov_dim = 12)
    sweep.dt = dt
    sweep.coefficient = -1.0j

    if(geom == 'ipchain'):
        sweep.use_time_dependent_hamiltonian = True

    res = np.zeros(nstep+1)
    plt.ion()
    plt.ylim([0, 0.5])
    line = plt.plot(np.arange(nstep+1)*dt, res)[0]

    res[0] = np.real(mel(op, A, A))
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        res[i+1] = np.real(mel(op, A, A))

        print((i+1)*dt, t2-t1, res[i+1], mel(A))
        sys.stdout.flush()
        plt.gcf().canvas.draw()
        line.set_data(np.arange(nstep+1)*dt, res)
        plt.pause(0.1)

        if(i % 100):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            h5.create_dataset('Sz', data=res)
            h5.close()

    plt.ioff()
    plt.close()
    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    h5.create_dataset('Sz', data=res)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')
    parser.add_argument('alpha', type = float)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)
    parser.add_argument('--delta', type = float, default=0)
    parser.add_argument('--eps', type = float, default=1)
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--nbose', type=int, default=20)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--geom', type = str, default='star')
    parser.add_argument('--fname', type=str, default='sbm.h5')
    args = parser.parse_args()

    nstep = int(30.0/args.dt)+1
    sbm_dynamics(30, args.alpha, args.wc, args.s, args.delta, args.eps, args.chi, args.nbose, args.dt, nstep = nstep, geom=args.geom, ofname = args.fname)
