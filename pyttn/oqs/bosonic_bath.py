import numpy as np
from numba import jit
import scipy as sp
from pyttn.utils import orthopol
from . import bath_discretisation as disc

class bosonic_bath:
    def __init__(self, Jw, Sp, Sm = None, beta=None):
        self.Sp = Sp
        self.Sm = Sm
        self.Jw = Jw
        self.beta = beta

    #improve this code to make it all work a bit better
    def Ct(self, t, wmax, wmin=None, epsabs=1.49e-08, epsrel=1.49e-08, limit=50):
        if wmin == None:
            if self.beta == None:
                wmin = 0
            else:
                wmin = -wmax
        Ct = np.zeros(t.shape, dtype=np.complex128)

        if self.beta == None:
            J = self.Jw
            @jit(nopython=True)
            def S(w):
                return J(np.abs(w))*np.where(w > 0, 1.0, 0.0)
            wc = 1e-10
            if wmin < wc:
                for ti in range(t.shape[0]):
                    ctr = sp.integrate.quad(S, wc, wmax, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    ctr += sp.integrate.quad(lambda x : S(x)*np.cos(x*t[ti]), wmin, wc, points=0, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                    cti = sp.integrate.quad(S, wc, wmax, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                    cti += sp.integrate.quad(lambda x : S(x)*np.sin(x*t[ti]), wmin, wc, points=0, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                    Ct[ti] = ctr - 1.0j*cti
            else:
                for ti in range(t.shape[0]):
                    ctr = sp.integrate.quad(S, wmin, wmax, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    cti = sp.integrate.quad(S, wmin, wmax, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                    Ct[ti] = ctr - 1.0j*cti
        else:
            beta = self.beta
            J = self.Jw
            @jit(nopython=True)
            def S(w):
                return J(np.abs(w))*0.5*(1.0+1.0/np.tanh(beta*np.abs(w)/2.0))*np.where(w > 0, 1.0, np.exp(beta*w))

            wc = 1e-10
            for ti in range(t.shape[0]):
                ctr = sp.integrate.quad(S, wc, wmax, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                ctr += sp.integrate.quad(lambda x : S(x)*np.cos(x*t[ti]), -wc, wc, points=0, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                ctr += sp.integrate.quad(S, wmin, -wc, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                cti = sp.integrate.quad(S, wc, wmax, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                cti += sp.integrate.quad(lambda x : S(x)*np.sin(x*t[ti]), -wc, wc, points=0, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                cti += sp.integrate.quad(S, wmin, -wc, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                Ct[ti] = ctr - 1.0j*cti
        return Ct/np.pi


    def Sw(self, w):
        if self.beta == None:
            return self.Jw(np.abs(w))*np.where(w > 0, 1.0, 0.0)
        else:
            return self.Jw(np.abs(w))*0.5*(1.0+1.0/np.tanh(self.beta*np.abs(w)/2.0))*np.where(w > 0, 1.0, np.exp(self.beta*w))


    def discretise(self, Nb, wmax, method='orthopol', *args, **kwargs):
        return disc.discretise_bosonic(self.Jw, Nb, wmax, method=method, beta=self.beta, *args, **kwargs)

    def fitCt(self, wmax, wmin=None, aaa_tol = 1e-3, Naaa=1000, aaa_nmax=500, aaa_support_points = "softm"):
        if wmin == None:
            if self.beta == None:
                wmin = 0
            else:
                wmin = -wmax

        from .heom.aaa_bath_fitting import aaa_fit_correlation_function
        return aaa_fit_correlation_function(lambda x : self.Sw(x), wmax = wmax, wmin=wmin, aaa_tol = aaa_tol, Naaa=Naaa, aaa_nmax=aaa_nmax, aaa_support_points = aaa_support_points)
