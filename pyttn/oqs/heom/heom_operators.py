import numpy as np

def commutator(L):
    return np.kron(L, np.identity(L.shape[0])) - np.kron(np.identity(L.shape[0]), L.T)


def anti_commutator(L):
    return np.kron(L, np.identity(L.shape[0])) + np.kron(np.identity(L.shape[0]), L.T)


def bkp(nbose, dk):
    ret = []
    # b1 = np.zeros((nbose-1), dtype=np.complex128)
    # row = np.zeros((nbose-1), dtype=int)
    # col = np.zeros((nbose-1), dtype=int)
    for i in range(nbose - 1):
        row = i
        col = i + 1
        b1 = np.sqrt((i + 1.0)) * np.sqrt(np.abs(dk))
        print(row, col, b1)
        ret.append((row, col, (1.0 + 0.0j) * b1))
    return ret
    # return sp.sparse.csr_matrix(b1, (row, col), shape=(nbose, nbose))


def bkm(nbose, dk, mind=True):
    ret = []
    # b1 = np.zeros((nbose-1), dtype=np.complex128)
    # row = np.zeros((nbose-1), dtype=int)
    # col = np.zeros((nbose-1), dtype=int)
    coeff = 1

    if not mind:
        coeff = -1.0
    for i in range(nbose - 1):
        row = i + 1
        col = i
        b1 = coeff * np.sqrt((i + 1.0)) * dk / np.sqrt(np.abs(dk))
        ret.append((row, col, (1.0 + 0.0j) * b1))
    # return b1, row, col
    return ret
    # return sp.sparse.csr_matrix(b1, (row, col), shape=(nbose, nbose))


def Sl(S):
    return np.kron(S, np.identity(S.shape[0]))


def Sr(S):
    return np.kron(np.identity(S.shape[0]), S.T)


def Sp(S):
    Scomm = commutator(S)
    return Scomm


def Sm(S, mind=True):
    Sop = None
    if mind:
        Sop = np.kron(S, np.identity(S.shape[0]))
    else:
        Sop = np.kron(np.identity(S.shape[0]), S.T)
    return Sop


def nop(nbose, zk):
    ns = np.arange(nbose)
    return -1.0j * (ns) * zk


def compute_dimensions(S, dk, zk, L, Lmin=None):
    ds = np.ones(2 * len(dk) + 1, dtype=int)
    ds[0] = S.shape[0] * S.shape[1]

    minzk = np.amin(np.real(zk))
    if Lmin is None:
        for i in range(len(dk)):
            nb = L
            ds[2 * i + 1] = nb
            ds[2 * i + 2] = nb
    else:
        for i in range(len(dk)):
            nb = max(int(L * minzk / np.real(zk[i])), Lmin)
            ds[2 * i + 1] = nb
            ds[2 * i + 2] = nb

    return ds
