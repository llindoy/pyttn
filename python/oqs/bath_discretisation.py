from pyttn.utils import orthopol_discretisation, density_discretisation
from numba import jit
import numpy as np

#compute a value for wmin such that wmin is the smallest w in range [-wmax, 0) such that S(wmin) < S(wmax)
#this should ensure a reasonable lower bound for the discretisation provided wmax is reasonable. Here this
#is done by simply computing S(w) in the range [-wmax, 0) using npoints and finding wmin satisfying this
#condition
def find_wmin(Sw, wmax, npoints = 1000):
    Swmax = Sw(wmax)
    wrange = np.linspace(-wmax, 0, npoints, endpoint=False)
    Swmin = Sw(wrange)
    return wrange[np.argmax(Swmin > Swmax)-1]


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

#a function for discretising a bosonic bath (optionally allowing finite temperature baths treated using thermofield theory) using an orthonormal
#polynomial based quadrature scheme.  
def discretise_orthopol_bosonic(J, Nb, wmax, wmin = None, beta=None, moment_scaling = None, atol=0, rtol=1e-10, minbound=1e-25, maxbound=1e25, moment_scaling_steps = 4, find_wmin_points=100, nquad=100):
    g = None
    w = None
    if beta == None:
        #setup the bath spectral function for zero temperature
        @jit(nopython=True)
        def S(w):
            return J(np.abs(w))*np.where(w > 0, 1.0, 0.0)

        #if wmin is not specified set it to zero
        if wmin is None:
            wmin = 0

        #if the moment scaling parameter is not zero set its value
        if moment_scaling is None:
            moment_scaling = find_moment_scaling_factor(S, wmin, wmax, Nb, atol=atol, rtol=rtol, minbound=minbound, maxbound=maxbound, Nsteps=moment_scaling_steps, nquad=nquad)

        g, w = orthopol_discretisation.discretise(S, wmin, wmax, Nb, moment_scaling=moment_scaling, atol=atol, rtol=rtol, nquad=nquad)

    else:
        #setup the bath spectral function for finite temperature
        @jit(nopython=True)
        def S(w):
            return J(np.abs(w))*0.5*(1.0+1.0/np.tanh(beta*np.abs(w)/2.0))*np.where(w > 0, 1.0, np.exp(beta*w))

        #if wmin is not specified set it uero
        if wmin is None:
            wmin = find_wmin(S, wmax, npoints = find_wmin_points)

        #if the moment scaling parameter is not zero set its value
        if moment_scaling is None:
            moment_scaling = find_moment_scaling_factor(S, wmin, wmax, Nb, atol=atol, rtol=rtol, minbound=minbound, maxbound=maxbound, Nsteps=moment_scaling_steps, nquad=nquad)

        g, w = orthopol_discretisation.discretise(S, wmin, wmax, Nb, moment_scaling=moment_scaling, atol=atol, rtol=rtol, nquad=nquad)

    return np.array(g), np.array(w)

#discretise a bosonic bath using the density algorithm.  Here we allow for specification of the density of frequencies (rho), if this isn't specified we use a density of frequencies
#that is given by S(w)/w for np.abs(w) > 1e-12 and S(w)/1e-12 for np.abs(w) < 1e-12. 
#this currently doesn't seem to work
def discretise_density_bosonic(J, Nb, wmax, rho=None, wmin = None, beta=None, atol=0, rtol=1e-10, nquad=100, wtol=1e-10, ftol=1e-10, niters=100, find_wmin_points=100):
    g = None
    w = None

    if beta is None:
        #setup the bath spectral function for zero temperature
        @jit(nopython=True)
        def S(w):
            return J(np.abs(w))*np.where(w > 0, 1.0, 0.0)

        #if wmin is not specified set it to zero
        if wmin is None:
            wmin = 0

        if rho is None:
            @jit(nopython=True)
            def rhofunc(w):
                return J(w)/np.where(np.abs(w) < 1e-12, 1e-12, np.abs(w))
            g, w = density_discretisation.discretise(S, S, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)
        else:
            g, w = density_discretisation.discretise(S, rho, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)
    else:
        Nb = Nb//2

        #setup the bath spectral function for zero temperature
        @jit(nopython=True)
        def S(w):
            return J(np.abs(w))*0.5*(1.0+1.0/np.tanh(beta*np.abs(w)/2.0))*np.where(w > 0, 1.0, np.exp(beta*w))

        #if wmin is not specified set it to zero
        if wmin is None:
            wmin = 0

        if rho is None:
            @jit(nopython=True)
            def rhofunc(w):
                return J(w)/np.where(np.abs(w) < 1e-12, 1e-12, np.abs(w))
            g, w = density_discretisation.discretise(S, S, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)
        else:
            g, w = density_discretisation.discretise(S, rho, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)

        g = np.array(g)
        w = np.array(w)
        wr = np.zeros(2*Nb-1)
        gr = np.zeros(2*Nb-1)
        wr[Nb-1:] = w
        gr[Nb-1:] = np.array(g)

        wf = np.flip(w)[:-1]
        gf = np.flip(g)[:-1]
        wr[:(Nb-1)] = -wf
        gr[:(Nb-1)] = gf*np.exp(-beta*wf/2.0)

        g = gr
        w = wr
    return np.array(g), np.array(w)


#A function for discretising a bosonic bath allowing for selection of different algorithms.  This requires the minimal specification of:
#   J : the spectral density of the bath
#   Nb : the number of bath modes
#   wmax : a maximum cutoff frequency used for the discretisation
def discretise_bosonic(J, Nb, wmax, method='orthopol', *args, **kwargs):
    if method == 'orthopol':
        return discretise_orthopol_bosonic(J, Nb, wmax, *args, **kwargs)
    elif method == 'density':
        return discretise_density_bosonic(J, Nb, wmax, *args, **kwargs)
    else:
        raise RuntimeError("Discretisation method not recognised.")



#a function for discretising a fermionic bath (optionally allowing finite temperature baths treated using thermofield theory) using an orthonormal
#polynomial based quadrature scheme.  
def discretise_orthopol_fermionic(J, Ef, Nb, wmax, wmin = None, beta=None, moment_scaling = None, atol=0, rtol=1e-10, minbound=1e-25, maxbound=1e25, moment_scaling_steps = 4, find_wmin_points=100, nquad=100):
    g = None
    w = None
    if beta == None:
        #setup the bath spectral function for zero temperature
        @jit(nopython=True)
        def S(w):
            return J(w)*np.where(w < Ef, 1.0, 0.0)

        #if wmin is not specified set it to zero
        if wmin is None:
            wmin = -wmax

        #if the moment scaling parameter is not zero set its value
        if moment_scaling is None:
            moment_scaling = find_moment_scaling_factor(S, wmin, wmax, Nb, atol=atol, rtol=rtol, minbound=minbound, maxbound=maxbound, Nsteps=moment_scaling_steps, nquad=nquad)

        g, w = orthopol_discretisation.discretise(S, wmin, wmax, Nb, moment_scaling=moment_scaling, atol=atol, rtol=rtol, nquad=nquad)

    else:
        #setup the bath spectral function for finite temperature
        @jit(nopython=True)
        def S(w):
            return J(w)*np.exp(-beta*(w-Ef)))(1+np.exp(-beta*(w-Ef)))

        #if wmin is not specified set it uero
        if wmin is None:
            wmin = -wmax

        #if the moment scaling parameter is not zero set its value
        if moment_scaling is None:
            moment_scaling = find_moment_scaling_factor(S, wmin, wmax, Nb, atol=atol, rtol=rtol, minbound=minbound, maxbound=maxbound, Nsteps=moment_scaling_steps, nquad=nquad)

        g, w = orthopol_discretisation.discretise(S, wmin, wmax, Nb, moment_scaling=moment_scaling, atol=atol, rtol=rtol, nquad=nquad)

    return np.array(g), np.array(w)

#discretise a fermionic bath using the density algorithm.  Here we allow for specification of the density of frequencies (rho), if this isn't specified we use a density of frequencies
#that is given by S(w)/w for np.abs(w) > 1e-12 and S(w)/1e-12 for np.abs(w) < 1e-12. 
#this currently doesn't seem to work
def discretise_density_fermionic(J, Nb, wmax, rho=None, wmin = None, beta=None, atol=0, rtol=1e-10, nquad=100, wtol=1e-10, ftol=1e-10, niters=100, find_wmin_points=100):
    g = None
    w = None

    if beta is None:
        #setup the bath spectral function for zero temperature
        @jit(nopython=True)
        def S(w):
            return J(w)*np.where(w < Ef, 1.0, 0.0)

        #if wmin is not specified set it to zero
        if wmin is None:
            wmin = -wmax

        if rho is None:
            @jit(nopython=True)
            def rhofunc(w):
                return J(w)/np.where(np.abs(w) < 1e-12, 1e-12, np.abs(w))
            g, w = density_discretisation.discretise(S, S, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)
        else:
            g, w = density_discretisation.discretise(S, rho, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)
    else:
        Nb = Nb//2

        #setup the bath spectral function for zero temperature
        @jit(nopython=True)
        def S(w):
            return J(w)*np.exp(-beta*(w-Ef)))(1+np.exp(-beta*(w-Ef)))

        #if wmin is not specified set it to zero
        if wmin is None:
            wmin = -wmax

        if rho is None:
            @jit(nopython=True)
            def rhofunc(w):
                return J(w)/np.where(np.abs(w) < 1e-12, 1e-12, np.abs(w))
            g, w = density_discretisation.discretise(S, S, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)
        else:
            g, w = density_discretisation.discretise(S, rho, wmin, wmax, Nb, atol=atol, rtol=rtol, nquad=nquad, wtol=wtol, ftol=ftol, niters=niters)

        g = np.array(g)
        w = np.array(w)
        wr = np.zeros(2*Nb-1)
        gr = np.zeros(2*Nb-1)
        wr[Nb-1:] = w
        gr[Nb-1:] = np.array(g)

        wf = np.flip(w)[:-1]
        gf = np.flip(g)[:-1]
        wr[:(Nb-1)] = -wf
        gr[:(Nb-1)] = gf*np.exp(-beta*wf/2.0)

        g = gr
        w = wr
    return np.array(g), np.array(w)


def discretise_fermionic(J, Eb, Nb, wmin, wmax, method='orthopol', *args, **kwargs):
    if method == 'orthopol':
        return discretise_orthopol_fermionic(J, Nb, wmin, wmax, *args, **kwargs)
    elif method == 'density':
        return discretise_density_fermionic(J, Nb, wmin, wmax, *args, **kwargs)
    else:
        raise RuntimeError("Discretisation method not recognised.")
