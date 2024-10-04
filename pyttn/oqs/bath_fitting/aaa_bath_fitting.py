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
    Z1 = softmspace(wmin, wc, N//2)
    nZ1 = -np.flip(Z1)
    Z = np.concatenate((nZ1, Z1))
    return Z

def AAA_support_points(w = "linear", wmin=None, wmax = None, Naaa=1000):
    if isinstance(w, (list, np.ndarray)):
        if isinstance(w, list):
            return np.array(w)
        else:
            return w

    elif(w == "softm"):
        if(wmax == None):
            wmax = 1
    
        if(wmin == None or wmin <= 0):
            wmin = 1e-8
    
        return generate_grid_points(Naaa, wmax, wmin=wmin)

    elif(w == "linear"):
        if(wmax == None):
            wmax = 1
        if(wmin == None):
            wmin = -1
        return np.linspace(wmin, wmax, Naaa)

    else:
        raise RuntimeError("Invalid AAA support points.")


class AAADecomposition:
    def __init__(self, tol=1e-4, w = "linear", aaa_nmax=500, wmin=None, wmax=None, Naaa= 1000, coeff=1.0, **kwargs):
        self.w = w
        self.wmin=wmin
        self.wmax=wmax
        self.Naaa=Naaa

        self.aaa_nmax=aaa_nmax
        self.coeff = coeff
        self.aaa_tol = tol
    
    def set_bounds(self, wmin, wmax):
        self.wmin=wmin
        self.wmax=wmax

    def AAA_to_HEOM(p, r, coeff = 1.0):
        pp = coeff*p*1.0j
        rr = -1.0j*r/(np.pi)
        inds = pp.real > 0
        pp = pp[inds]
        rr = rr[inds]
        return rr, pp

    def __call__(self, Sw):
        self.Z1 = AAA_support_points(w = self.w, wmin=self.wmin, wmax=self.wmax, Naaa=self.Naaa)

        #first compute the aaa decomposition of the spectral function
        func1, p, r, z = AAA_algorithm(Sw, self.Z1, nmax=self.aaa_nmax, tol=self.aaa_tol)

        
        #and convert that to the heom correlation function coefficients
        dk, zk = AAADecomposition.AAA_to_HEOM(p, r, coeff=self.coeff)

        #sort the arguments based on largest abs of the poles
        dk = dk[np.argsort(np.abs(zk))]
        zk = zk[np.argsort(np.abs(zk))]
        print(dk, zk)

        #return the function for optional plotting as well as the coefficients
        return dk, zk, func1

