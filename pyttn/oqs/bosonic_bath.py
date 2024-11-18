import numpy as np
from numba import jit
import scipy as sp

class BosonicBath:
    """A class for managing a continuous bosonic gaussian bath.  This provides
    functions for computing non-interacting bath correlation functions, as well as
    decomposing the correlation function into a linear combination of
    complex valued exponentials (expfit) or oscillator terms (discretise)
    :param Jw: The bath spectral function defining the non-interacting correlation function
    :param S: The system operator
    :type S: SOP, optional
    :param beta: The inverse temperature of the bath, defaults to None
    :type beta: float, optional 
    :param wmax: the maximum frequency bound, defaults to np.inf
    :type wmax: float, optional
    :param wmin: the minimum frequency bound, defaults to None
    :type wmin: float, optional
    """
    def __init__(self, Jw, S=None, beta=None, wmax=np.inf, wmin=None):
        self.Jw = Jw
        self.S = S
        self.beta = beta
        if wmin == None:
            wmin = self.find_wmin(wmax)

        self.wmin = wmin
        self.wmax = wmax

    def find_wmin(self, wmax, npoints = 1000):
        if(self.beta is None):
            return 0
        else:
            if ( wmax == np.inf):
                return -np.inf
            else:
                Swmax = self.Sw(wmax)
                wrange = np.linspace(-wmax, 0, npoints, endpoint=False)
                Swmin = self.Sw(wrange)
                return wrange[np.argmax(Swmin > Swmax)-1]

    def estimate_bounds(self, wmax=None):
        """Returns estimates for the upper and lower bounds of the spectral density to be used for the
        discretisation function
        :param wmax: the maximum frequency bound, defaults to self.wmax
        :type wmax: float, optional
        :return: the maximum and minimum frequency bounds
        :rtype: float, float 
        """
        if wmax is None:
            wmax = self.wmax
        wmax = np.abs(wmax)
        wmin = self.find_wmin(wmax)

        return wmin, wmax

    #improve this code to make it all work a bit better
    def Ct(self, t, epsabs=1.49e-12, epsrel=1.49e-12, limit=2000):
        """Returns the value of the non-interacting bath correlation function evaluated 
        at the time points t:
        .. math::
            C^{\\sigma}(t) = \\frac{1}{pi}\int_{wmin}^{wmax} J(\\omega) f_B(\\beta\\omega) exp(- i \\omega t)
        :param t: time
        :type t: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        wmin = self.wmin
        wmax = self.wmax

        if self.beta == None and wmin < 0:
            wmin = 0
        Ctr = np.zeros(t.shape, dtype=np.complex128)
        Cti = np.zeros(t.shape, dtype=np.complex128)

        print(wmin, wmax)
        wc = 1e-10
        #if wmin > wc we don't span zero
        if(wmin >= wc):
            #if wmax is infinite then we do a standard integral 
            if wmax == np.inf:
                Ctr = sp.integrate.quad_vec(lambda x : self.Sw(x)*np.cos(x*t), wmin, wmax, epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                Cti = sp.integrate.quad_vec(lambda x : self.Sw(x)*np.sin(x*t), wmin, wmax, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
            #otherwise we can make use of the weight function
            else:
                for ti in range(t.shape[0]):
                    Ctr[ti] = sp.integrate.quad(lambda x : self.Sw(x), wmin, wmax, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    Cti[ti] = sp.integrate.quad(lambda x : self.Sw(x), wmin, wmax, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        #if wmin < wc then we need to split the integral
        else:
            #if wmin is negative we will split it into either two or three regions 
            #depending on the value of wmin and -wc.  
            #Handle the region [wc, wmax]
            if wmax == np.inf:
                Ctr += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.cos(x*t), wc, wmax, epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                Cti += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.sin(x*t), wc, wmax, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
            #otherwise we can make use of the weight function
            else:
                for ti in range(t.shape[0]):
                    Ctr[ti] += sp.integrate.quad(lambda x : self.Sw(x), wc, wmax, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    Cti[ti] += sp.integrate.quad(lambda x : self.Sw(x), wc, wmax, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]

            #Handle the two regions [wmin, -wc], (-wc, wc)
            if(wmin < -wc):
                if wmin == -np.inf:
                    Ctr += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.cos(x*t), wmin, -wc, epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    Cti += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.sin(x*t), wmin, -wc, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                #otherwise we can make use of the weight function
                else:
                    for ti in range(t.shape[0]):
                        Ctr[ti] += sp.integrate.quad(lambda x : self.Sw(x), wmin, wc, weight='cos', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                        Cti[ti] += sp.integrate.quad(lambda x : self.Sw(x), wmin, wc, weight='sin', wvar = t[ti], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                    Ctr+= sp.integrate.quad_vec(lambda x : self.Sw(x)*np.cos(x*t), -wc, wc, points=[0], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    Cti+= sp.integrate.quad_vec(lambda x : self.Sw(x)*np.sin(x*t), -wc, wc, points=[0], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
            #Handle the regions (wmin, wc)
            else:
                if(wmin <= 0):
                    Ctr += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.cos(x*t), wmin, wc, points=[0], epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    Cti += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.sin(x*t), wmin, wc, points=[0], epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                else:
                    Ctr += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.cos(x*t), wmin, wc, epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
                    Cti += sp.integrate.quad_vec(lambda x : self.Sw(x)*np.sin(x*t), wmin, wc, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]

        return (Ctr - 1.0j*Cti)/np.pi

    def Ctexp(t, dk, zk):
        """Returns the value of the non-interacting bath correlation function evaluated 
        at the time points t using the results of discretisation or expfit:
        :param t: time
        :type t: np.ndarray
        :param dk: the weights of each term in the fit
        :type dk: np.ndarray
        :param zk: the (complex) frequencies of each term in the fit
        :type zk: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        ret = np.zeros(t.shape, dtype=np.complex128)
        for i in range(len(dk)):
            ret += dk[i]*np.exp(-1.0j*zk[i]*t)
        return ret

    def Sw(self, w):
        """Returns the non-interacting bath spectral function at w
        :param w: frequency
        :type w: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the
            greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        if self.beta == None:
            return self.Jw(np.abs(w))*np.where(w > 0, 1.0, 0.0)
        else:
            return np.where(w>0, self.Jw(w), -self.Jw(-w))*0.5*(1.0+1.0/np.tanh(self.beta*w/2.0))

    def discretise(self, discretisation_engine):
        """Returns the coupling constants and frequencies associated with a discretised representation of the bath
        :param discretisation_energy: The object used to discretise the c
        :type w: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the
            greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """

        from .bath_fitting import OrthopolDiscretisation, DensityDiscretisation
        if(discretisation_engine.wmin is None):
            if self.wmin is None:
                discretisation_engine.wmin = -self.wmax
            else:
                discretisation_engine.wmin = self.wmin
        if(discretisation_engine.wmax is None):
            discretisation_engine.wmin = 2*self.wmax
        return discretisation_engine(self.Sw)

    def expfit(self, fitting_engine):
        from .bath_fitting import AAADecomposition, ESPRITDecomposition
        dk = None
        zk = None
        if isinstance(fitting_engine, AAADecomposition):
            if self.wmin is None:
                fitting_engine.wmin = -self.wmax
            else:
                fitting_engine.wmin = self.wmin
            if(fitting_engine.wmax is None):
                fitting_engine.wmin = 2*self.wmax
            dk, zk, _ = fitting_engine(self.Sw)

        elif isinstance(fitting_engine, ESPRITDecomposition):
            dk, zk, _ = fitting_engine(lambda x : self.Ct(x))
        return dk, zk
            
