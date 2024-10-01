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

    def Ct(self, t, wmax=np.inf, wmin=None, wtol=None, Ef = 0, sigma = '+'):
        wmin, wmax = self.estimate_bounds(wmax, wmin=wmin, wtol=wtol, Ef=Ef, sigma = sigma)
        Ct = np.zeros(t.shape, dtype=np.complex128)

        coeff = 1
        if sigma == '-':
            coeff = -1

        if(wmax == np.inf or wmin == -np.inf):
            ctr = sp.integrate.quad_vec(lambda x : self.Sw(x, Ef=Ef, sigma=sigma)*np.cos(x*t), wmin, wmax, limit=5000)[0] 
            cti = sp.integrate.quad_vec(lambda x : self.Sw(x, Ef=Ef, sigma=sigma)*np.sin(x*t), wmin, wmax, limit=5000)[0]
            Ct = ctr + coeff*1.0j*cti
        else:
            for ti in range(t.shape[0]):

                ctr = sp.integrate.quad(lambda x : self.Sw(x, Ef=Ef, sigma=sigma), wmin, wmax, weight='cos', wvar = t[ti])[0] 
                cti = sp.integrate.quad(lambda x : self.Sw(x, Ef=Ef, sigma=sigma), wmin, wmax, weight='sin', wvar = t[ti])[0]

                Ct[ti] = ctr + coeff*1.0j*cti
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

    def Sw(self, w, Ef = 0, sigma = '+'):
        if(sigma == '+'):
            return self.Jw(w)*self.fermi_distrib(w, Ef)
        else:
            return self.Jw(w)*(1-self.fermi_distrib(w, Ef))

    def estimate_bounds(self, wmax, wmin=None, wtol=None, Ef = 0, sigma = '+'): 

        wmax = np.abs(wmax)
        if self.beta is None:
            if(sigma == '+'):
                wmin = -wmax
                wmax = Ef
            else:
                wmin = Ef
        else:
            if(wtol == None):
                wmin = -wmax
            else:
                if(sigma == '+'):
                    wmin = -wmax
                    wmax = min(wmax, (1.0/self.beta*np.log(1/wtol-1)) + Ef)
                else:
                    wmin = min(wmax, (Ef-1.0/self.beta*np.log(1/wtol-1)) )

        return wmin, wmax

    def discretise(self, Nm, wmax, wmin=None, wtol = None, Ef = 0, sigma = '+', method='orthopol', *args, **kwargs):
        wmin, wmax = self.estimate_bounds(wmax, wmin=wmin, wtol=wtol, Ef=Ef, sigma = sigma)
        print(wmin, wmax)

        gf, wf = disc.discretise_fermionic(lambda x : self.Sw(x, Ef=Ef, sigma=sigma), Nm, wmin, wmax, method=method, *args, **kwargs)
        return gf, wf

    def fitCt(self, wmax, wmin=None, wtol=None, Ef = 0, sigma='+', aaa_tol = 1e-3, Naaa=1000, aaa_nmax=500, aaa_support_points = "linear"):
        wmin, wmax = self.estimate_bounds(wmax, wmin=wmin, wtol=wtol, Ef=Ef)

        from .heom.aaa_bath_fitting import aaa_fit_correlation_function
        return aaa_fit_correlation_function(lambda x : self.Sw(x, Ef=Ef, sigma=sigma), wmax = wmax, wmin=wmin, aaa_tol = aaa_tol, Naaa=Naaa, aaa_nmax=aaa_nmax, aaa_support_points = aaa_support_points)
