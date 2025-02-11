import numpy as np
from .ESPRIT import ESPRIT
from .aaa import AAA_algorithm


class ExpFitDecomposition:
    """Base class for exponential fit decompositions
    """

    def __init__(self):
        pass


def ESPRIT_support_points(t="linear", tmax=None, Nt=1000):
    """A function for automatically generating support points to be used within the ESPRIT algorithm.  

    :param t: Either the support points or a key word used to generate the support points. (Default: "linear") This parameter can be either a

        - list/np.ndarray: In this case the function simply returns these points as the support points ignoring all other inputs
        - str: In this case the function generates points according to a rule using the user specified tmax, and Nt. Possible options for this argument are

            - linear: Use a linear set of grid points between wmin and wmax

    :type t: str or np.ndarray or list, optional
    :param tmax: The maximum allowed time. (Default: None)
    :type tmax: float or None, optional
    :param Nt: The number of time points to use with the ESPRIT algorithm. (Default: 1000)
    :type Nt: int or None, optional
    """
    if isinstance(ESPRIT_support_points, (list, np.ndarray)):
        if isinstance(ESPRIT_support_points, list):
            return np.array(ESPRIT_support_points)
        else:
            return ESPRIT_support_points
    else:
        if tmax is None:
            raise RuntimeError("Invalid tmax.")
        return np.linspace(0, np.abs(tmax), Nt)


class ESPRITDecomposition(ExpFitDecomposition):
    """A class providing an easy to use interface for applying the ESPRIT decomposition to 
    construct a sum-of-exponential approximation for a bath correlation function

    Constructor arguments: 

    :param K: The number of terms in the sum-of-exponential decomposition
    :type K: int
    :param t: Either the support points orr a key word used to generate the support points.  For details see the ESPRIT_support_points function.  (Default: "linear")
    :type t: str or np.ndarray or list, optional
    :param \*\*kwargs: Keyword arguments to pass to ESPRIT_support_points

    Callable arguments:

    :param Ct: The non-interacting bath correlation function to be decomposed.
    :type Ct: callable
    :returns:
        - dk (np.ndarray) - The coefficients in the sum-of-exponential decomposition of the bath correlation function
        - zk (np.ndarray) - The exponents in the sum-of-exponential decomposition of the bath correlation function
        - Ctres (callable) - The ESPRIT fit of the bath correlation function

    """

    def __init__(self, K, t="linear", **kwargs):
        self.t = ESPRIT_support_points(t=t, **kwargs)
        self.K = K

    def __call__(self, Ct):
        dt = self.t[1]-self.t[0]
        C = Ct(self.t)
        dk, zk, Ctres = ESPRIT(C, self.K)
        zk = zk/dt
        return dk, zk, Ctres


def __softmspace(start, stop, N, beta=1, endpoint=True):
    start = (np.log(1-(np.exp(-beta*start))) + beta*start)/beta
    # start = np.log(np.exp(beta*start)-1)/beta
    stop = (np.log(1-(np.exp(-beta*stop))) + beta*stop)/beta
    # stop = np.log(np.exp(beta*stop)-1)/beta

    dx = (stop-start)/N
    if (endpoint):
        dx = (stop-start)/(N-1)

    return np.logaddexp(beta*(np.arange(N)*dx + start), 0)/beta
    # return np.log(np.exp(beta*(np.arange(N)*dx  + start))+1)/beta


def __generate_grid_points(N, wc, wmin=1e-9):
    Z1 = __softmspace(wmin, wc, N//2)
    nZ1 = -np.flip(Z1)
    Z = np.concatenate((nZ1, Z1))
    return Z


def AAA_support_points(w="linear", wmin=None, wmax=None, Naaa=1000):
    """A function for automatically generating support points to be used within the AAA algorithm.  

    :param w: Either the support points orr a key word used to generate the support points. (Default: "linear") This parameter can be either a

        - list/np.ndarray: In this case the function simply returns these points as the support points ignoring all other inputs
        - str: In this case the function generates points according to a rule using the user specified wmin, wmax, Naaa. Possible options for this argument are

            - linear: Use a linear set of grid points between wmin and wmax
            - softm: Use a softmspace (interpolates between linear and logarithmic) for positive frequencies and a mirrored set of points for negative frequencies

    :type w: str or np.ndarray or list, optional
    :param wmin: The minimum allowed frequency. (Default: None)
    :type wmin: float or None, optional
    :param wmax: The maximum allowed frequency. (Default: None)
    :type wmax: float or None, optional
    :param Naaa: The number of support points used by the AAA algorithm. (Default: 1000)
    :type Naaa: int or None, optional
    """
    if isinstance(w, (list, np.ndarray)):
        if isinstance(w, list):
            return np.array(w)
        else:
            return w

    elif (w == "softm"):
        if (wmax == None):
            wmax = 1

        if (wmin == None or wmin <= 0):
            wmin = 1e-8

        return __generate_grid_points(Naaa, wmax, wmin=wmin)

    elif (w == "linear"):
        if (wmax == None):
            wmax = 1
        if (wmin == None):
            wmin = -1

        return np.linspace(wmin, wmax, Naaa)

    else:
        raise RuntimeError("Invalid AAA support points.")


class AAADecomposition:
    """A class providing an easy to use interface for applying the AAA based rational function 
    decomposition to approximate a bath spectral density.  This class includes tools used for
    automatic generation of support points, as well as manually specified support points to 
    allow for both easy general use and flexibility when handling more complex spectral densities.

    Constructor arguments: 

    :param tol: The tolerance used within the AAA algorithm. (Default: 1e-4)
    :type tol: float, optional
    :param K: The maximum number of poles to fit. (default: None)
    :type K: int or None, optional
    :param w: Either the support points or a key word used to generate the support points.  For details see the AAA_support_points function.  (Default: "linear")
    :type w: str or np.ndarray or list, optional
    :param aaa_nmax: The maximum number of poles to allow within the AAA algorithm. (Default:500)
    :type aaa_nmax: int, optional
    :param coeff: A coefficient to go in front of the frequency terms. (Default: 1)
    :type coeff: float, optional
    :param wmin: The minimum allowed frequency. (Default: None)
    :type wmin: float or None, optional
    :param wmax: The maximum allowed frequency. (Default: None)
    :type wmax: float or None, optional
    :param Naaa: The number of support points used by the AAA algorithm. (Default: 1000)
    :type Naaa: int or None, optional

    Callable arguments:

    :param S: The spectral density function to be decomposed.
    :type S: callable
    :returns:
        - dk (np.ndarray) - The coefficients in the sum-of-exponential decomposition of the bath correlation function
        - zk (np.ndarray) - The exponents in the sum-of-exponential decomposition of the bath correlation function
        - func (callable) - The AAA rational function fit of the spectral density

    """

    def __init__(self, tol=1e-4, K = None, w="linear", aaa_nmax=500, coeff=1.0, wmin=None, wmax=None, Naaa=1000):
        self.Z1 = None
        self.w = w
        self.wmin=wmin
        self.wmax=wmax
        self.Naaa = Naaa

        self.aaa_nmax = aaa_nmax
        self.coeff = coeff
        self.aaa_tol = tol
        self.K = K

    def __AAA_to_HEOM(p, r, coeff=1.0):
        """Convert the poles and residues from the AAA algorithm into the coefficients and frequencies needed
        for the HEOM algorithms

        :param p: The poles extracted from the AAA algorithm
        :type p: np.ndarray
        :param r: The residues extracted from the AAA algorithm
        :type r: np.ndarray
        :param coeff: A coefficient to go in front of the frequency terms. (Default: 1)
        :type coeff: float, optional
        """
        pp = coeff*p*1.0j
        rr = -1.0j*r/(np.pi)
        inds = pp.real > 0
        pp = pp[inds]
        rr = rr[inds]
        return rr, pp

    def __call__(self, S):
        """Perform the AAA decomposition on the function S.

        :param S: The spectral density function to be decomposed.
        :type S: callable

        :returns:
            - dk (np.ndarray) - The coefficients in the sum-of-exponential decomposition of the bath correlation function
            - zk (np.ndarray) - The exponents in the sum-of-exponential decomposition of the bath correlation function
            - func (callable) - The AAA rational function fit of the spectral density
        """
        #generate the support points for the AAA decomposition
        if not isinstance(self.Z1, (np.ndarray, list)):
            self.Z1 = AAA_support_points(w=self.w, wmin=self.wmin, wmax=self.wmax, Naaa=self.Naaa)

        # first compute the aaa decomposition of the spectral function
        func1, p, r, z = AAA_algorithm(
            S, self.Z1, nmax=self.aaa_nmax, tol=self.aaa_tol, K = self.K)

        # and convert that to the heom correlation function coefficients
        dk, zk = AAADecomposition.__AAA_to_HEOM(p, r, coeff=self.coeff)

        # sort the arguments based on largest abs of the poles
        dk = dk[np.argsort(np.abs(zk))]
        zk = zk[np.argsort(np.abs(zk))]

        # return the function for optional plotting as well as the coefficients
        return dk, zk, func1
