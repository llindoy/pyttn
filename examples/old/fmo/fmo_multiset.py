import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *
from pyttn.utils import density_discretisation, orthopol_discretisation
from pyttn._pyttn import operator_dictionary_complex
from numba import jit


meV = 0.000036749304951208
cmn1 = 0.0000045563352812
fs = 41.341374575751

#fmo spectral density from Phys. Rev. Lett., 132, 100403
@jit(nopython=True)
def Jar(w):
    res = 0*w
    S = 0.29
    s = np.array([0.8, 0.5])
    omega = np.array([0.069, 0.24])*meV
    for i in range(2):
        res += S/(s[0]+s[1]) * s[i]/(7*6*5*4*3*2*1*2*omega[i]) * np.power(w, 5)*np.exp(-np.sqrt(np.abs(w)/omega[i]))
    return res

@jit(nopython=True)
def Jh(w):
    wk = np.array([46, 68, 117, 167, 180, 191, 202, 243, 263, 284, 291, 327, 366, 385, 404, 423, 440, 481, 541, 568, 582, 597, 630, 638, 665, 684, 713, 726, 731, 750, 761, 770, 795, 821, 856, 891, 900, 924, 929, 946, 966, 984, 1004, 1037, 1058, 1094, 1104, 1123, 1130, 1162, 1175, 1181, 1201, 1220, 1283, 1292, 1348, 1367, 1386, 1431, 1503, 1545])*cmn1
    gk = 5*cmn1*np.ones(wk.shape)
    sk = np.array([0.011, 0.011, 0.009, 0.009, 0.010, 0.011, 0.011, 0.012, 0.003, 0.008, 0.008, 0.003, 0.006, 0.002, 0.002, 0.002, 0.001, 0.002, 0.004, 0.007, 0.004, 0.004, 0.003, 0.006, 0.004, 0.003, 0.007, 0.01, 0.005, 0.004, 0.009, 0.018, 0.007, 0.006, 0.007, 0.003, 0.004, 0.001, 0.001, 0.002, 0.002, 0.003, 0.001, 0.002, 0.002, 0.001, 0.001, 0.003, 0.003, 0.009, 0.007, 0.01, 0.003, 0.005, 0.002, 0.004, 0.007, 0.002, 0.004, 0.002, 0.003, 0.003])
    res = w*0.0
    for i in range(62):
        res += 4*wk[i]*sk[i]*gk[i]*(wk[i]**2+gk[i]**2)*w/(np.pi*((w+wk[i])**2+gk[i]**2)*((w-wk[i])**2+gk[i]**2))
    return res

def spectral_density(w):
    return Jar(w)+Jh(w)


def S(w, T):
    if T is None:
        return spectral_density(w)*np.where(w > 0, 1.0, 0.0)
    else:
        kb = 3.166811563e-6
        beta = 1.0/(kb*T)
        return spectral_density(w)*0.5*(1.0+1.0/np.tanh(beta*w/2.0))

def Hsys():
    return np.array([ [410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9], [-87.7, 530, 30.8, 8.2, 0.7, 11.8, 4.3], [5.5, 30.8, 210, -53.5, -2.2, -9.6, 6.0], [-5.9, 8.2, -53.5, 320, -70.7, -17.0, -63.3],  [6.7, 0.7, -2.2, -70.7, 480, 81.1, -1.3],[-13.7, 11.8, -9.6, -17.0, 81.1, 630, 39.7], [-9.9, 4.3, 6.0, -53.3, -1.3, 39.7, 440]])*cmn1

def Ct(t, w, g):
    T, W = np.meshgrid(t, w)
    g2 = g**2
    fourier = np.exp(-1.0j*W*T)
    return g2@fourier

def multiset_fmo_dynamics(T, Nb, chi, nbose, dt, nstep):

    N = Nb*7
    mode_dims = [nbose for i in range(Nb*7)]
    b_mode_dims = [nbose for i in range(Nb)]


    #construct the topology tree 
    topo = ntree("(1(%d(%d(%d)(%d))(%d(%d)(%d)))(%d(%d(%d)(%d))(%d)))" %( chi, chi, chi,  chi, chi, chi,  chi, chi, chi,  chi, chi, chi ))
    print(topo)
    
    #and add the node that forms the root of the bath
    degree = 2
    ntreeBuilder.mlmctdh_subtree(topo()[0][0][0], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[0][0][1], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[0][1][0], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[0][1][1], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][0][0], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][0][1], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][1], b_mode_dims, degree, chi)
    ntreeBuilder.sanitise(topo)
    print(topo)

    wmin = -2000*cmn1
    if T == 0:
        wmin=0

    g, w = orthopol_discretisation.discretise( lambda x : S(x, T), wmin, 2000*cmn1, Nb, moment_scaling=2, atol=0, rtol=1e-10)
    g = np.array(g)
    w = np.array(w)


    H = multiset_SOP(7, N)
    sysinf = system_modes(N)

    for i in range(7):
        for m in range(Nb):
            sysinf[i *Nb + m] = boson_mode(nbose)

    Hs = Hsys()
    for i in range(7):
        for j in range(7):
            H[i, j] += Hs[i,j]

    for i in range(7):
        for m in range(Nb):
            #add on the bath coupling terms
            H[i,i] += np.sqrt(2)*g[m]*sOP("q", i*Nb+m)

            #add on the identity bath terms
            for j in range(7):
                H[i, i] += w[m]*sOP("n", j*Nb+m)

    A = ms_ttn(topo, 7, dtype=np.complex128)
    coeff = np.zeros(7)
    coeff[5] = 1.0
    state = [[0 for i in range(N)] for j in range(7)]
    A.set_state(coeff, state)

    observables = []
    for i in range(7):
        op = multiset_SOP(7, N)
        op[i, i] += 1.0
        observables.append(multiset_sop_operator(op, A, sysinf))

    mel = matrix_element(A)

    h = multiset_sop_operator(H, A, sysinf)

    sweep = tdvp(A, h, krylov_dim = 8)
    sweep.dt = dt*fs
    sweep.coefficient = -1.0j

    res = np.zeros((7, nstep+1))

    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=[0, 1])
    lines = []
    lines2 = []
    for j in range(7):
        lines.append(ax.plot(np.arange(nstep+1)*dt, res[j, :], 'k')[0])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$n(t)$")

    #for j in range(7):
    #    res[j, 0] = np.real(mel(observables[j], A, A))
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        print((i+1)*dt, end=' ')
        for j in range(7):
            res[j, i+1] = np.real(mel(observables[j], A))
            print(res[j, i+1], end=' ')
        print(t2-t1)
        
        if(plt.fignum_exists(num)):
            plt.gcf().canvas.draw()
            for j in range(7):
                lines[j].set_data(np.arange(nstep+1)*dt, res[j, :])
            plt.pause(0.01)

    plt.ioff()
    plt.show()

T = 300
Nb = 128
multiset_fmo_dynamics(T, Nb, 1, 10, 0.5, 1000)
