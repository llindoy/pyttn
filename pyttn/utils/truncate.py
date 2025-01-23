import numpy as np


class TruncationBase:
    """Base class for local Hilbert space mode decomposition
    """
    def __init__(self):
        return 

class DepthTruncation(TruncationBase):
    """A class for truncating system modes to a fixed number of levels   

    Constructor arguments

    :param Lmax: The number of states to include in the local Hilbert space (default:2)
    :type Lmax: int, optional


    Callable arguments

    :param gk: The interaction strength
    :type gk: np.ndarray
    :param wk: The mode frequency
    :type wk: np.ndarray
    :param is_fermion: Whether or not the modes to be truncated are fermionic
    :type is_fermion: bool

    :returns: A list containing the local Hilbert space dimension of each mode
    :rtype: list 
    """
    def __init__(self, Lmax = 2):
        self.Lmax = Lmax

    def __call__(self, gk, wk, is_fermion):
        if is_fermion:
            return [2 for i in range(len(wk))]
        else:
            return [self.Lmax for i in range(len(wk))]

class EnergyTruncation(TruncationBase):
    """A class for truncating system modes to a fixed number of levels   

    Constructor arguments

    :param ecut: The maximum energy to include in the problem (defaul: 0)
    :type ecut: float, optional
    :param Lmax: The maximum number of states to include in the local Hilbert space (default:2)
    :type Lmax: int, optional
    :param Lmin: The minimum number of states to include in the local Hilbert space (default:1)
    :type Lmin: int, optional
    :param func: Whether to use the absolute value or real part of the frequency. (default: "abs")
    :type func: {"abs", "real"}, optional


    Callable arguments

    :param gk: The interaction strength
    :type gk: np.ndarray
    :param wk: The mode frequency
    :type wk: np.ndarray
    :param is_fermion: Whether or not the modes to be truncated are fermionic
    :type is_fermion: bool

    :returns: A list containing the local Hilbert space dimension of each mode
    :rtype: list 
    """

    def __init__(self, ecut = 0, Lmax = 2, Lmin = 1, func='abs'):
        self.ecut = ecut
        self.Lmax = Lmax
        self.Lmin = Lmin
        self.func = func

    def truncate_bosonic(self, gk, wk):
        """Truncate a bosonic mode using coupling constants and frequencies   

        :param gk: The interaction strength
        :type gk: np.ndarray
        :param wk: The mode frequency
        :type wk: np.ndarray

        :returns: A list containing the local Hilbert space dimension of each mode
        :rtype: list 
        """
        _wk = np.zeros(wk.shape, dtype=float)
        if self.func == 'abs':
            for i in range(len(wk)):
                _wk[i] = np.abs(wk[i])
        else:
            for i in range(len(wk)):
                _wk[i] = np.real(wk[i])

        Nb = []
        for i in range(len(wk)):
            nbose = self.Lmax
            if not self.ecut is None:
                nbose = int(self.ecut/np.real(_wk[i]))
            if nbose < self.Lmin:
                nbose = self.Lmin

            if(nbose > self.Lmax):
                nbose = self.Lmax
            Nb.append(nbose)
        return Nb

    def __call__(self, gk, wk, is_fermion):
        if is_fermion:
            return [2 for i in range(len(wk))]
        else:
            return self.truncate_bosonic(gk, wk)
