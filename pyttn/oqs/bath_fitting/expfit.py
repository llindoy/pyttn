import numpy as np
from .aaa import AAA_algorithm
from .ESPRIT import ESPRIT

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
    Z1 = softmspace(wmin, wc, N//2)
    nZ1 = -np.flip(Z1)
    Z = np.concatenate((nZ1, Z1))
    return Z


def AAA_support_points(aaa_support_points = "linear", wmin=None, wmax = None, Naaa=1000):
    if isinstance(aaa_support_points, (list, np.ndarray)):
        if isinstance(aaa_support_points, list):
            return np.array(aaa_support_points)
        else:
            return aaa_support_points

    elif(aaa_support_points == "softm"):
        if(wmax == None):
            wmax = 1
    
        if(wmin == None or wmin <= 0):
            wmin = 1e-8
    
        return generate_grid_points(Naaa, wmax, wmin=wmin)

    elif(aaa_support_points == "linear"):
        if(wmax == None):

        if(wmin == None):
            wmin = -1
        return np.linspace(wmin, wmax, Naaa)

    else:
        raise RuntimeError("Invalid AAA support points.")


class AAA_decomposition:
    def __init__(self, support_points = "linear", nmax=500, coeff=1.0, tol = 1e-4, **kwargs):
        self.Z1 = AAA_support_points(aaa_support_points = support_points, **kwargs)
        self.aaa_nmax=nmax
        self.coeff = coeff
        sel.aaa_tol=tol

    def AAA_to_HEOM(p, r, coeff = 1.0):
        pp = coeff*p*1.0j
        rr = -1.0j*r/(np.pi)
        inds = pp.real > 0
        pp = pp[inds]
        rr = rr[inds]
        return rr, pp


    def __call__(self, Sw):
        Sw_aaa, dk, zk = setup_heom_correlation_functions(Sw, self.Z1, nmax=self.aaa_nmax, aaa_tol=self.aaa_tol, coeff=coeff)
        return dk, zk, Sw_aaa

