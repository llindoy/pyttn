import numpy as np
from .aaa import AAA_algorithm

def softmspace(start, stop, N, beta = 1, endpoint = True):
    start = np.log(np.exp(beta*start)-1)/beta
    stop = np.log(np.exp(beta*stop)-1)/beta

    dx = (stop-start)/N
    if(endpoint):
        dx = (stop-start)/(N-1)

    return np.log(np.exp(beta*(np.arange(N)*dx  + start))+1)/beta

def generate_grid_points(N, wc, wmin=1e-9):
    Z1 = softmspace(wmin, 20*wc, N)
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

class heom_bath:
    def __init__(self, Scoup, Sw, L, Lmin = None, aaa_support_points = None, wmax = None, wmin = None, aaa_tol = 1e-3, Naaa=1000, aaa_nmax=500, scale_factor=None):
        self.Scoup = Scoup
        self.Sw = Sw
        self.L = L
        self.Lmin = None
        self._aaa_support_points = aaa_support_points
        self.wmax = wmax
        self.wmin = wmin
        self.aaa_tol = aaa_tol
        self.Naaa = Naaa
        self.aaa_nmax=500
        self.scale_factor = scale_factor

    def aaa_support_points(self):
        if(self._aaa_support_points == None):
            wmax = self.wmax
            if(self.wmax == None):
                wmax = 1

            wmin = self.wmin
            if(self.wmin == None):
                wmin = 1e-8

            return generate_grid_points(self.Naaa, wmax, wmin=wmin)

        elif isinstance(self._aaa_support_points, (list, np.ndarray)):
            if isinstance(self._aaa_support_points, list):
                return np.array(self._aaa_support_points)
            else:
                return self._aaa_support_points


    def discretise(self, output_fitting=False):
        Z1 = self.aaa_support_points()
        Sw_aaa, dk, zk = setup_heom_correlation_functions(self.Sw, Z1, nmax=self.aaa_nmax, aaa_tol=self.aaa_tol)

        bath_dict = {
                    "S" : self.Scoup,
                    "d" : dk,
                    "z" : zk,
                    "L" : self.L
                }

        if self.Lmin is not None:
            bath_dict["Lmin"]=self.Lmin

        if self.scale_factor is not None:
            bath_dict["sf"]=self.scale_factor

        if output_fitting:
            wmax = self.wmax
            if(self.wmax == None):
                wmax = 1
            Zv = np.linspace(-20*wmax, 20*wmax, 100000)
            Swa = Sw_aaa(Zv)
            Swb = self.Sw(Zv)

            bath_dict["wf"] = Zv
            bath_dict["Sw"] = Swb
            bath_dict["Sw_fit"] = Swa

        return bath_dict

