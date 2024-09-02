import numpy as np
from numba import jit
import scipy as sp
from pyttn.utils import orthopol
from . import bath_discretisation as disc

class fermionic_bath:
    def __init__(self, Jw, Sp=None, Sm=None, beta=None):
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
                return J(w)*np.where(w <= Ef, 1.0, 0.0)
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
            J = self.Jw
            @jit(nopython=True)
            def S(w):
                return J(w)*np.exp(-self.beta*(w-Ef))/(1+np.exp(-self.beta*(w-Ef)))

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

    def fermi_distrib(self, w, Ef):
        if self.beta == None:
            return np.where(w <= Ef, 1.0, 0.0)
        else:
            if isinstance(w, np.ndarray):
                res = 0.0*w
                res[w < Ef] = 1/(1+np.exp(self.beta*(w[w<Ef]-Ef)))
                res[w >= Ef] = np.exp(-self.beta*(w[w>=Ef]-Ef))/(1+np.exp(-self.beta*(w[w>=Ef]-Ef)))
            else:
                if(w < Ef):
                    return 1/(1+np.exp(self.beta*(w-Ef)))
                else:
                    return np.exp(-self.beta*(w-Ef))/(1+np.exp(-self.beta*(w-Ef)))

            return res

    def Sw(self, w, Ef = 0):
        return self.Jw(w)*self.fermi_distrib(w, Ef)

    def Sw_filled(self, w, Ef=0):
        return self.Sw(w, Ef)

    def Sw_empty(self, w, Ef=0):
        return self.Jw(w)*(1-self.fermi_distrib(w, Ef))

    def discretise(self, Nb, wmax, wmin=None, wmin_empty = None, wmax_filled = None, Ef = 0, method='orthopol', wmin_tol = None, *args, **kwargs):
        Nm = Nb//2
        if Nm == 0:
            Nm = 1
        if(self.beta == None):
            if(wmin == None):
                wmin = -wmax
            gf, wf = disc.discretise_fermionic(lambda x : self.Sw_filled(x), Nm, wmin, Ef, method=method, *args, **kwargs)
            ge, we =  disc.discretise_fermionic(lambda x : self.Sw_empty(x), Nm, Ef, wmax, method=method, *args, **kwargs)
            return gf, wf, ge, we
        else:
            if(wmin == None):
                wmin = -wmax
            if wmin_tol == None:
                if(wmin_empty == None):
                    wmin_empty = wmin
                if wmax_filled == None:
                    wmax_filled = wmax
            else:
                Ef = 0
                tol = 1e-10
                win = 1.0/self.beta*np.log(1/tol-1)+Ef
                if(wmin_empty == None):
                    wmin_empty = -win
                if wmax_filled == None:
                    wmax_filled = win

            if(wmax_filled > wmax):
                wmax_filled = wmax
            if(wmin_empty < wmin):
                wmin_empty = wmin

            gf, wf = disc.discretise_fermionic(lambda x : self.Sw_filled(x), Nm, wmin, wmax_filled, method=method, *args, **kwargs)
            ge, we =  disc.discretise_fermionic(lambda x : self.Sw_empty(x), Nm, wmin_empty, wmax, method=method, *args, **kwargs)
            return gf, wf, ge, we


