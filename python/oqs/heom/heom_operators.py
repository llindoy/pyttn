
import numpy as np
import scipy as sp

def commutator(L):
    return np.kron(L, np.identity(L.shape[0])) - np.kron(np.identity(L.shape[0]), L.T)

def anti_commutator(L):
    return np.kron(L, np.identity(L.shape[0])) + np.kron(np.identity(L.shape[0]), L.T)

def bkp(nbose, dk):
    b1 = np.zeros((nbose-1), dtype=np.complex128)
    row = np.zeros((nbose-1), dtype=int)
    col = np.zeros((nbose-1), dtype=int)
    for i in range(nbose-1):
        row[i] = i
        col[i] = i+1
        b1[i] = np.sqrt((i+1.0))*np.sqrt(np.abs(dk))

    return sp.sparse.csr_matrix(b1, (row, col), shape=(nbose, nbose))

def bkm(nbose, dk, mind=True):
    b1 = np.zeros((nbose-1), dtype=np.complex128)
    row = np.zeros((nbose-1), dtype=int)
    col = np.zeros((nbose-1), dtype=int)
    coeff = 1

    if not mind:
        coeff = -1.0
    for i in range(nbose-1):
        row[i]= i+1
        col[i] = i
        b1[i] = coeff*np.sqrt((i+1.0))*dk/np.sqrt(np.abs(dk))
    return sp.sparse.csr_matrix(b1, (row, col), shape=(nbose, nbose))

def Sp(S):
    Scomm = commutator(S)
    return Scomm

def Sm(S, mind = True):
    Sop = None
    if mind:
        Sop = np.kron(S, np.identity(S.shape[0]))
    else:
        Sop = np.kron(np.identity(S.shape[0]), S.T)
    return Sy;


def nop(nbose, zk):
    ns = np.arange(nbose)
    return sp.sparse.csr_matrix(-1.0j*(ns)*zk, (ns, ns), shape=(nbose, nbose))
    return 

def compute_dimensions(S, dk, zk, L, Lmin = None):
    ds = np.ones(2*len(dk)+1, dtype = int)
    ds[0] = S.shape[0]*S.shape[1]

    minzk = np.amin(np.real(zk))
    if(Lmin == None):
        for i in range(len(dk)):
            nb = L
            ds[2*i+1] = nb
            ds[2*i+2] = nb
    else:
        for i in range(len(dk)):
            nb = max(int(L*minzk/np.real(zk[i])), Lmin)
            ds[2*i+1] = nb
            ds[2*i+2] = nb

    return ds


