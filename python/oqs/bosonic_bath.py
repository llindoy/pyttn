import numpy as np
from numba import jit
import scipy as sp
from pyttn.utils import orthopol
from . import bath_discretisation as disc

class bosonic_bath:
    def __init__(self, Jw, Sp, Sm = None, beta=None):
        self.Sp = Sp
        self.Sm = None
        self.Jw = Jw
        self.beta = beta

    def Ct(self, t, wmax, wmin=None):
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
                return J(np.abs(w))*0.5*(1.0+1.0/np.tanh(beta*np.abs(w)/2.0))*np.where(w > 0, 1.0, np.exp(beta*w))

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


    def Sw(self, w):
        if self.beta == None:
            return self.Jw(np.abs(w))*np.where(w > 0, 1.0, 0.0)
        else:
            return self.Jw(np.abs(w))*0.5*(1.0+1.0/np.tanh(self.beta*np.abs(w)/2.0))*np.where(w > 0, 1.0, np.exp(self.beta*w))


    def discretise(self, Nb, wmax, method='orthopol', *args, **kwargs):
        return disc.discretise_bosonic(self.Jw, Nb, wmax, method=method, beta=self.beta, *args, **kwargs)


