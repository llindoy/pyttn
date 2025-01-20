import numpy as np


class TruncationBase:
    def __init__(self):
        return 

class DepthTruncation:
    def __init__(self, Lmax = 2):
        self.Lmax = Lmax

    def __call__(self, ck, wk, is_fermion):
        if is_fermion:
            return [2 for i in range(len(wk))]
        else:
            return [self.Lmax for i in range(len(wk))]

class EnergyTruncation:
    def __init__(self, ecut = 0, Lmax = 2, Lmin = 1, func='abs'):
        self.ecut = ecut
        self.Lmax = Lmax
        self.Lmin = Lmin
        self.func = func

    def truncate_bosonic(self, ck, wk):
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

    def __call__(self, ck, wk, is_fermion):
        if is_fermion:
            return [2 for i in range(len(wk))]
        else:
            return self.truncate_bosonic(ck, wk)
