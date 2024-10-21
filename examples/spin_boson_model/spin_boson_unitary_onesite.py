import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs
from numba import jit


#setup the star Hamiltonian for the spin boson model
def star_hamiltonian(eps, delta, g, w, Nb):
    N = Nb+1

    H = SOP(N)
    H += eps*sOP("sz", 0)
    H += delta*sOP("sx", 0)
    for i in range(Nb):
        H += np.sqrt(2)*g[i] * sOP("sz", 0)  * sOP("q", i+1)
        H += w[i] * sOP("n", i+1)

    return H, w


#setup the chain hamiltonian for the spin boson model - this is the tedopa method
def chain_hamiltonian(eps, delta, g, w, Nb):
    N = Nb+1

    t, e = oqs.chain_map(g, w)

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
def ipchain_hamiltonian(eps, delta, g, w, Nb):
    N = Nb+1

    t, e, U = oqs.chain_map(g, w, return_unitary = True)
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


def setup_topology(chi, nbose, mode_dims, degree):
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(topo)
    return topo


def combine_modes(bath_mode_dims, bath_mode_inds, nbmax, nhilbmax):
    composite_modes = []

    all_modes_traversed = False
    cmode = []
    chilb = 1
    mode = 0
    while not all_modes_traversed:
        #if the current cmode object is empty then we just add the current mode to the composite mode and increment
        if(len(cmode) == 0):
            cmode.append(bath_mode_inds[mode])
            chilb = chilb*bath_mode_dims[mode]
            mode += 1
            if(mode == len(bath_mode_dims)):
                all_modes_traversed = True
                composite_modes.append(copy.deepcopy(cmode))
        else:
            #othewise we check to see if the composite mode could accept the current mode without exceeding the bounds
            #then we add and increment
            if len(cmode) < nbmax and chilb*bath_mode_dims[mode] <= nhilbmax:
                cmode.append(bath_mode_inds[mode])
                chilb = chilb*bath_mode_dims[mode]
                mode += 1
                if(mode == len(bath_mode_dims)):
                    all_modes_traversed = True
                    composite_modes.append(copy.deepcopy(cmode))

            else:
                #otherwise we have reached the end of the current composite mode.  We will now reset the composite 
                #mode object and we will not increment the mode so that it start a new composite mode object in the
                #next iteration
                composite_modes.append(copy.deepcopy(cmode))
                cmode = []
                chilb = 1

    return composite_modes


def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = None, Ncut = 20, nstep = 1, Nw = 10.0, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=10, nhilbmax=1024):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, S=sOP("sz", 0), beta=beta)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, 0, Nw*wc))

    import matplotlib.pyplot as plt
    plt.plot(t, oqs.BosonicBath.Ctexp(t, g*g, w))
    plt.show()

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(N)

    #now add on the bath bits getting additionally getting a frequency parameter that can be used in energy based
    #truncation schemes
    if(geom == 'star'):
        H, w = star_hamiltonian(eps, delta, 2*g, w, Nb)
    elif geom == 'chain':
        H, w = chain_hamiltonian(eps, delta, 2*g, w, Nb)
    elif geom == 'ipchain':
        H, w = ipchain_hamiltonian(eps, delta, 2*g, w, Nb)
    else:
        raise RuntimeError("Hamiltonian geometry not recognised.")


    #mode_dims = [nbose for i in range(Nb)]
    mode_dims = [min(max(4, int(wc*Ncut/w[i])), nbose) for i in range(Nb)]

    #attempt to combine modes together based on am
    composite_modes = combine_modes(mode_dims, [x+1 for x in range(Nb)], nbmax, nhilbmax)
    Nbc = len(composite_modes)

    #setup the system information object
    sysinf = system_modes(1+len(composite_modes))
    sysinf[0] = spin_mode(2)

    tree_mode_dims = []
    for ind, cmode in enumerate(composite_modes):
        print(ind, cmode)
        sysinf[ind+1] = [boson_mode(mode_dims[x-1]) for x in cmode]
        lhd = np.prod(np.array([mode_dims[x-1] for x in cmode]))
        tree_mode_dims.append(lhd)

    #construct the topology and capacity trees used for constructing 
    chi0 = chi

    topo = setup_topology(chi0, nbose, tree_mode_dims, degree)
    capacity = setup_topology(chi, nbose, tree_mode_dims, degree)


    #construct and initialise the ttn wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)
    print(A.nmodes(), Nbc+1)
    A.set_state([0 for i in range(Nbc+1)])

    print(topo)

    #set up the Hamiltonian
    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)

    #construct objects need for evaluating observables
    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )


    #set up tdvp sweeping algorithm parameters
    sweep = tdvp(A, h, krylov_dim = 12)
    sweep.dt = dt
    sweep.coefficient = -1.0j

    if(geom == 'ipchain'):
        sweep.use_time_dependent_hamiltonian = True

    #run dynamics and measure properties storing them in a file
    res = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)
    res[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()

    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
    for i in range(5):
        dti = ts[i]-tp
        print(ts, dt)
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    i=1
    res[1] =np.real(mel(op, A, A))
    maxchi[1] = A.maximum_bond_dimension()
    sweep.dt = dt

    for i in range(1,nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        res[i+1] = np.real(mel(op, A, A))
        maxchi[i+1] = A.maximum_bond_dimension()
        print(i, res[i+1], A.maximum_bond_dimension())

        if(i % 10 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            h5.create_dataset('Sz', data=res)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    h5.create_dataset('Sz', data=res)
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('alpha', type = float)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)

    #number of bath modes
    parser.add_argument('--N', type=int, default=400)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=1)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=32)
    parser.add_argument('--degree', type=int, default=1)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--tmax', type=float, default=30)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm.h5')

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, degree = args.degree)
