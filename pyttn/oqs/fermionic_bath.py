import numpy as np
from numba import jit
import scipy as sp
from pyttn.utils import orthopol
from .bath_fitting.aaa_bath_fitting import AAADecomposition
from .bath_fitting.ESPRIT_bath_fitting import ESPRITDecomposition

class FermionicBath:
    """A class for managing a continuous fermionic gaussian bath.  This provides
    functions for computing non-interacting bath correlation functions, as well as
    decomposing the correlation function into a linear combination of
    complex valued exponentials (expfit) or oscillator terms (discretise)
    :param Jw: The bath spectral function defining the non-interacting correlation function
    :param beta: The inverse temperature of the bath, defaults to None
    :type beta: float, optional 
    :param wmax: the maximum frequency bound, default to np.inf
    :type wmax: float, optional
    :param wmin: the minimum frequency bound, default to np.inf
    :type wmin: float, optional
    :param wtol: a value for determining wmin based on wmax.  See fermionic.bath.estimate_bounds, default to None
    :type wtol: float, optional
    """
    def __init__(self, Jw, Sp=None, Sm=None, beta=None, wmax=np.inf, wmin=None, wtol=None):
        self.Jw = Jw
        self.Sp = Sp
        self.Sm = Sm
        self.beta = beta
        self.wmin = wmin
        self.wmax = wmax
        self.wtol=wtol

    def Ct(self, t, Ef = 0, sigma = '+', epsabs=1.49e-12, epsrel=1.49e-12, limit=2000):
        """Returns the value of the non-interacting bath correlation function evaluated 
        at the time points t:

        .. math::
            C^{\\sigma}(t) = \\frac{1}{pi}\int_{wmin}^{wmax} J(\\omega) f_F(\\sigma\\beta(\\omega - Ef)) exp(\\sigma i \\omega t)

        :param t: time
        :type t: np.ndarray
        :param Ef: The fermi energy, default to 0
        :type Ef: float, optional
        :param sigma: Whether to compute greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        wmin, wmax = self.estimate_bounds(Ef=Ef, sigma = sigma)
        Ct = np.zeros(t.shape, dtype=np.complex128)

        coeff = 1
        if sigma == '-':
            coeff = -1

        if(wmax == np.inf or wmin == -np.inf):
            ctr = sp.integrate.quad_vec(lambda x : self.Sw(x, Ef=Ef, sigma=sigma)*np.cos(x*t), wmin, wmax, epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
            cti = sp.integrate.quad_vec(lambda x : self.Sw(x, Ef=Ef, sigma=sigma)*np.sin(x*t), wmin, wmax, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
            Ct = ctr + coeff*1.0j*cti
        else:
            for ti in range(t.shape[0]):
                ctr = sp.integrate.quad(lambda x : self.Sw(x, Ef=Ef, sigma=sigma), wmin, wmax, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                cti = sp.integrate.quad(lambda x : self.Sw(x, Ef=Ef, sigma=sigma), wmin, wmax, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                Ct[ti] = ctr + coeff*1.0j*cti
        return Ct/np.pi


    def Ctexp(t, dk, zk, sigma='+'):
        """Returns the value of the non-interacting bath correlation function evaluated 
        at the time points t using the results of discretisation or expfit:
        :param t: time
        :type t: np.ndarray
        :param dk: the weights of each term in the fit
        :type dk: np.ndarray
        :param zk: the (complex) frequencies of each term in the fit
        :type zk: np.ndarray
        :param sigma: Whether to compute greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        coeff = 1
        if sigma == '-':
            coeff = -1

        ret = np.zeros(t.shape, dtype=np.complex128)
        for i in range(len(dk)):
            ret += dk[i]*np.exp(coeff*1.0j*zk[i]*t)
        return ret

    def fermi_distrib(self, w, Ef):        
        """Returns the value fermi function at w and fermi energy Ef:
        :param w: frequency
        :type w: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :return: The bath correlation function
        :rtype: np.ndarray
        """
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
        """Returns the non-interacting bath spectral function at w and fermi energy Ef
        :param w: frequency
        :type w: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        if(sigma == '+'):
            return self.Jw(w)*self.fermi_distrib(w, Ef)
        else:
            return self.Jw(w)*(1-self.fermi_distrib(w, Ef))

    def estimate_bounds(self, wmax=None, Ef = 0, sigma = '+'): 
        """Returns estimates for the upper and lower bounds of the spectral density to be used for the
        discretisation function
        :param wmax: the maximum frequency bound, defaults to self.wmax
        :type wmax: float, optional
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: the maximum and minimum frequency bounds
        :rtype: float, float 
        """
        if wmax is None:
            wmax = self.wmax
        wmin = self.wmin
        wtol = self.wtol
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

    def discretise(self, discretisation_engine, Ef = 0, sigma = '+'):
        from .bath_fitting import OrthopolDiscretisation, DensityDiscretisation
        wmin, wmax = self.estimate_bounds(Ef=Ef, sigma = sigma)
        if(discretisation_engine.wmin is None):
            discretisation_engine.wmin = wmin
        if(discretisation_engine.wmax is None):
            discretisation_engine.wmin = wmax
        return discretisation_engine(lambda x : self.Sw(x, Ef=Ef, sigma=sigma))

    def expfit(self, fitting_engine, Ef = 0, sigma = '+'):
        from .bath_fitting import AAADecomposition, ESPRITDecomposition
        dk = None
        zk = None
        if isinstance(fitting_engine, AAADecomposition):
            wmin, wmax = self.estimate_bounds(Ef=Ef, sigma = sigma)
            wav = (wmax-wmin)/2
            if(fitting_engine.wmin is None):
                fitting_engine.wmin = wav - 2*(wav - wmin)
            if(fitting_engine.wmax is None):
                fitting_engine.wmin = wav + 2*(wmax-wav)
            dk, zk, _ = fitting_engine(lambda x : self.Sw(x, Ef=Ef, sigma=sigma))

        elif isinstance(fitting_engine, ESPRITDecomposition):
            dk, zk, _ = fitting_engine(lambda x : self.Ct(x, Ef=Ef, sigma=sigma))
        return dk, zk
            
