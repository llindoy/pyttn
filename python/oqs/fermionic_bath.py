import numpy as np
from numba import jit
import scipy as sp
from pyttn.utils import orthopol
from . import bath_discretisation as disc

class fermionic_bath:
    def __init__(self, Jw, Sp, Sm, beta=None):
        self.Sp = Sp
        self.Sm = Sm
        self.Jw = Jw
        self.beta = beta

    def Ct(self, t, wmax, wmin=None, Ef = 0):
        if wmin == None:
            wmin = -wmax
        Ct = np.zeros(t.shape, dtype=np.complex128)

        if self.beta == None:
            J = self.Jw
            @jit(nopython=True)
            def S(w):
                return J(w)*np.where(w < Ef, 1.0, 0.0)
            wc = 1e-10
            if wmin < wc:
                for ti in range(t.shape[0]):
                    ctr = sp.integrate.quad(S, wc, wmax, weight='cos', wvar = t[ti])[0] 
                    ctr += sp.integrate.quad(lambda x : S(x)*np.cos(x*t[ti]), wmin, wc, points=0)[0]
                    cti = sp.integrate.quad(S, wc, wmax, weight='sin', wvar = t[ti])[0]
                    cti += sp.integrate.quad(lambda x : S(x)*np.sin(x*t[ti]), wmin, wc, points=0)[0]
            else:
                for ti in range(t.shape[0]):
                    ctr = sp.integrate.quad(S, wmin, wmax, weight='cos', wvar = t[ti])[0] 
                    cti = sp.integrate.quad(S, wmin, wmax, weight='sin', wvar = t[ti])[0]
                Ct[ti] = ctr - 1.0j*cti
        else:
            beta = self.beta
            J = self.Jw
            @jit(nopython=True)
            def S(w):
                return J(w)*np.exp(-beta*(w-Ef)))(1+np.exp(-beta*(w-Ef)))

            wc = 1e-10
            for ti in range(t.shape[0]):
                ctr = sp.integrate.quad(S, wc, wmax, weight='cos', wvar = t[ti])[0]
                ctr += sp.integrate.quad(lambda x : S(x)*np.cos(x*t[ti]), -wc, wc, points=0)[0]
                ctr += sp.integrate.quad(S, wmin, -wc, weight='cos', wvar = t[ti])[0]
                cti = sp.integrate.quad(S, wc, wmax, weight='sin', wvar = t[ti])[0]
                cti += sp.integrate.quad(lambda x : S(x)*np.sin(x*t[ti]), -wc, wc, points=0)[0]
                cti += sp.integrate.quad(S, wmin, -wc, weight='sin', wvar = t[ti])[0]
                Ct[ti] = ctr - 1.0j*cti
        return Ct/np.pi


    def Sw(self, w, Ef = 0):
        if self.beta == None:
            return self.Jw(w)*np.where(w < Ef, 1.0, 0.0)
        else:
            return self.Jw(w)*np.exp(-beta*(w-Ef)))(1+np.exp(-beta*(w-Ef)))


    def discretise(self, Nb, wmax, wmin=None, Ef = 0, method='orthopol', *args, **kwargs):
        if(wmin == None):
            wmin = -wmax
        return disc.discretise_fermionic(self.Jw, Ef, Nb, wmin=wmin, wmax, method=method, beta=self.beta, *args, **kwargs)

    def fitCt(self, wmax = None, wmin=None, aaa_tol = 1e-3, Naaa=1000, aaa_nmax=500, aaa_support_points = None, Ef = 0):
        from .heom.aaa_bath_fitting import aaa_fit_correlation_function
        return aaa_fit_correlation_function(lambda x : self.Sw(x, Ef=Ef), wmax = wmax, wmin=wmin, aaa_tol = aaa_tol, Naaa=Naaa, aaa_nmax=aaa_nmax, aaa_support_points = aaa_support_points)
