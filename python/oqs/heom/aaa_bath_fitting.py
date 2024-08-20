import numpy as np
from .aaa import AAA_algorithm

def softmspace(start, stop, N, beta = 1, endpoint = True):
    start = (np.log(1-(np.exp(-beta*start))) + beta*start)/beta
    #start = np.log(np.exp(beta*start)-1)/beta
    stop = (np.log(1-(np.exp(-beta*stop))) + beta*stop)/beta
    #stop = np.log(np.exp(beta*stop)-1)/beta

    dx = (stop-start)/N
    if(endpoint):
        dx = (stop-start)/(N-1)

    return np.logaddexp(beta*(np.arange(N)*dx  + start), 0)/beta
    #return np.log(np.exp(beta*(np.arange(N)*dx  + start))+1)/beta

def generate_grid_points(N, wc, wmin=1e-9):
    Z1 = softmspace(wmin, wc, N)
    nZ1 = -np.flip(Z1)
    Z = np.concatenate((nZ1, Z1))
    return Z


def AAA_to_HEOM(p, r):
    pp = p*1.0j
    rr = -1.0j*r/(np.pi)
    inds = pp.real > 0
    pp = pp[inds]
    rr = rr[inds]
    return rr, pp

def setup_heom_correlation_functions(Sw, Z1, nmax = 500, aaa_tol = 1e-4):
    #first compute the aaa decomposition of the spectral function
    func1, p, r, z = AAA_algorithm(Sw, Z1, nmax=nmax, tol=aaa_tol)
    
    #and convert that to the heom correlation function coefficients
    dk, zk = AAA_to_HEOM(p, r)

    #return the function for optional plotting as well as the coefficients
    return func1, dk, zk


def generate_aaa_support_points(wmin=None, wmax = None, Naaa=1000, aaa_support_points = None):
    if(aaa_support_points == None):
        if(wmax == None):
            wmax = 1
    
        if(wmin == None):
            wmin = 1e-8
    
        return generate_grid_points(Naaa, wmax, wmin=wmin)
    
    elif isinstance(aaa_support_points, (list, np.ndarray)):
        if isinstance(aaa_support_points, list):
            return np.array(aaa_support_points)
        else:
            return aaa_support_points

def aaa_fit_correlation_function(Sw, wmin=None, wmax = None, aaa_tol = 1e-3, Naaa=1000, aaa_nmax=500, aaa_support_points = None):
    Z1 = generate_aaa_support_points(wmin=wmin, wmax = wmax, Naaa=Naaa, aaa_support_points = aaa_support_points)
    Sw_aaa, dk, zk = setup_heom_correlation_functions(Sw, Z1, nmax=aaa_nmax, aaa_tol=aaa_tol)
    return dk, zk, Sw_aaa

