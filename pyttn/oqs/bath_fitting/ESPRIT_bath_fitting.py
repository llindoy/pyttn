import numpy as np
from .ESPRIT import ESPRIT

def ESPRIT_support_points(t = "linear", tmax = None, Nt=1000):
    if isinstance(ESPRIT_support_points, (list, np.ndarray)):
        if isinstance(ESPRIT_support_points, list):
            return np.array(ESPRIT_support_points)
        else:
            return ESPRIT_support_points
    else:
        if tmax is None:
            raise RuntimeError("Invalid tmax.")
        return np.linspace(0, np.abs(tmax), Nt)

class ESPRITDecomposition:
    def __init__(self, K, t = "linear", **kwargs):
        self.t = ESPRIT_support_points(t = t, **kwargs)
        self.K = K

    def __call__(self, Ct):
        dt = self.t[1]-self.t[0]
        C = Ct(self.t)
        dk, zk, Ctres = ESPRIT(C, self.K)
        zk = zk/dt
        return dk, zk, Ctres

