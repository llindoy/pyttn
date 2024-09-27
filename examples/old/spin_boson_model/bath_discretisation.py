import numpy as np
import scipy.linalg as spla
from scipy.optimize import nnls
import matplotlib.pyplot as plt


def construct_interpolative(A, eps_or_k):
    B = None
    P = None
    k = None
    if(eps_or_k < 1):
        k, idx, proj = spla.interpolative.interp_decomp(A, eps_or_k)
        P = spla.interpolative.reconstruct_interp_matrix(idx, proj)
        B = spla.interpolative.reconstruct_skel_matrix(A, k, idx)
    else:
        idx, proj = spla.interpolative.interp_decomp(A, eps_or_k)
        P = spla.interpolative.reconstruct_interp_matrix(idx, proj)
        B = spla.interpolative.reconstruct_skel_matrix(A, eps_or_k, idx)
        k = eps_or_k
    return B, P, idx, k

def discretise_bath(S, win, zin, tmax, nt, tol = 1e-6):
    t = np.linspace(0, tmax, nt)
    Sw = S(win)

    W, T = np.meshgrid(win, t)
    fourier = np.exp(-1.0j*W*T)

    f = fourier*(Sw[np.newaxis, :])
    Cf = fourier*(Sw*zin)[np.newaxis, :]

    B, P, idx, k = construct_interpolative(f, tol)
    print(B)
    
    #extract the selected frequencies
    ws = win[idx[:k]]

    #get the correlation function by summing over the frequencies
    C = np.sum(Cf, axis=1)

    #now solver for the weights 
    zs, rnorm = nnls(B, C)

    gs = np.sqrt(zs*Sw[idx[:k]])

    return ws, gs, t, C


def discretise_bath_uniform(S, wmin, wmax, nw, tmax, nt, tol=1e-6):
    w = np.linspace(wmin, wmax, nw)
    zin = np.ones(w.shape)*(w[1]-w[0])
    return discretise_bath(S, w, zin, tmax, nt, tol=tol)

def Sw(w):
    return w*np.exp(-np.abs(w)/5)*(1+1/np.tanh(w/2))


ws, gs, t, C = discretise_bath_uniform(Sw, -50, 50, 10000, 100, 1000, 1e-6)
plt.plot(t, np.real(C), 'k')
res = np.zeros(t.shape, dtype=np.complex128)
for i in range(len(ws)):
    res += np.abs(gs[i])**2*np.exp(-1.0j*ws[i]*t)
plt.plot(t, np.real(res), 'r--')
plt.show()
wv = np.linspace(-20, 100, 1000)


plt.plot(wv, Sw(wv))
plt.scatter(ws, gs*gs)
plt.show()
