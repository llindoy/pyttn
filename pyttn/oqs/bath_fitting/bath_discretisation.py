from pyttn.utils import orthopol_discretisation, density_discretisation
from numba import jit
import numpy as np


class DensityDiscretisation:
    def __init__(self,  Nb, wmin, wmax, rho=None, atol=0, rtol=1e-10, nquad=100, wtol=1e-10, ftol=1e-10, niters=100, wcut=1e-8):
        self.Nb = Nb
        self.wmin = wmin
        self.wmax = wmax
        self.rho=rho
        self.wcut = wcut
        self.atol=atol
        self.rtol=rtol
        self.nquad=nquad
        self.wtol = wtol
        self.ftol=ftol
        self.niters=niters

    #discretise a bosonic bath using the density algorithm.  Here we allow for specification of the density of frequencies (rho), if this isn't specified we use a density of frequencies
    #that is given by S(w)/w for np.abs(w) > 1e-12 and S(w)/1e-12 for np.abs(w) < 1e-12. 
    #this currently doesn't seem to work
    def __call__(self, S):
        g = None
        w = None

        if self.rho is None:
            @jit(nopython=True)
            def rhofunc(w):
                return S(w)/np.where(np.abs(w) < wcut, wcut, np.abs(w))

            g, w = density_discretisation.discretise(S, rhofunc, self.wmin, self.wmax, self.Nb, atol=self.atol, rtol=self.rtol, nquad=self.nquad, wtol=self.wtol, ftol=self.ftol, niters=self.niters)

        else:
            g, w = density_discretisation.discretise(S, self.rho, self.wmin, self.wmax, self.Nb, atol=self.atol, rtol=self.rtol, nquad=self.nquad, wtol=self.wtol, ftol=self.ftol, niters=niters)

        return np.array(g), np.array(w)


class OrthopolDiscretisation:
    def __init__(self, Nb, wmin, wmax, moment_scaling = None, atol=0, rtol=1e-10, minbound=1e-25, maxbound=1e25, moment_scaling_steps = 4, nquad=100):
        self.Nb = Nb
        self.wmin = wmin
        self.wmax=wmax
        self.moment_scaling=moment_scaling
        self.atol=atol
        self.rtol=rtol
        self.minbound=minbound
        self.maxbound=maxbound
        self.moment_scaling_steps=moment_scaling_steps
        self.nquad=nquad

    #find a constant to scale the frequency axis by in order to ensure well defined scaling of the modified moments as we go to very high moments.
    #here this is done by computing the modified moments up to varying orders (with early termination if the values leave some min and max bounds)
    #and fitting the resultant decay or growth to an exponential function.  Based on this fitting we extract a constant that aims to minimise the 
    #decay or growth, and repeats this process with growing orders (up to the maximum order we need for the discretisation) until it reaches a
    #point that the moment decay does not leave the early termination bounds.  This may not always be optimal if the Nsteps variable is too large
    #and in this case either try reducing Nsteps.
    def find_moment_scaling_factor(Sw, wmin, wmax, Nb, atol=0, rtol=1e-10, minbound=1e-30, maxbound=1e30, Nsteps = 5, nquad=100):
        Nbs = (np.arange(Nsteps)+1)*(Nb//(Nsteps+1))
        ms=1
        for Nbi in Nbs:
            Nbi = max(2, Nbi)
            moments = np.array(orthopol_discretisation.moments(Sw, wmin, wmax, Nbi, moment_scaling=ms, atol=atol, rtol=rtol, minbound=1e-30,maxbound=1e30, nquad=nquad))
            nm = np.polyfit(np.arange(moments.shape[0]), np.log(np.abs(moments)), 1)[0]
            ms = ms*np.exp(-nm)
            #if this hasn't exited early then we return at this point
            if(moments.shape[0] == Nbi*2):
                return ms
        return ms

    def __call__(self, S):
        g = None
        w = None

        #if the moment scaling parameter is not zero set its value
        if self.moment_scaling is None:
            self.moment_scaling = OrthopolDiscretisation.find_moment_scaling_factor(S, self.wmin, self.wmax, self.Nb, atol=self.atol, rtol=self.rtol, minbound=self.minbound, maxbound=self.maxbound, Nsteps=self.moment_scaling_steps, nquad=self.nquad)

        g, w = orthopol_discretisation.discretise(S, self.wmin, self.wmax, self.Nb, moment_scaling=self.moment_scaling, atol=self.atol, rtol=self.rtol, nquad=self.nquad)
        return np.array(g), np.array(w)
