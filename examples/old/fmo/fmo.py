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


def single_set_fmo_dynamics(T, Nb, chi, nbose, dt, nstep):

    N = Nb*7
    mode_dims = [nbose for i in range(Nb*7)]
    b_mode_dims = [nbose for i in range(Nb)]


    #construct the topology tree 
    topo = ntree("(1(7(7))(1(%d(%d(%d)(%d))(%d(%d)(%d)))(%d(%d(%d)(%d))(%d))))" %( chi, chi, chi,  chi, chi, chi,  chi, chi, chi,  chi, chi, chi ))
    print(topo)
    
    #and add the node that forms the root of the bath
    degree = 2
    ntreeBuilder.mlmctdh_subtree(topo()[1][0][0][0], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][0][0][1], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][0][1][0], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][0][1][1], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][1][0][0], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][1][0][1], b_mode_dims, degree, chi)
    ntreeBuilder.mlmctdh_subtree(topo()[1][1][1], b_mode_dims, degree, chi)
    ntreeBuilder.sanitise(topo)

    wmin = -2000*cmn1
    if T == 0:
        wmin=0

    g, w = orthopol_discretisation.discretise( lambda x : S(x, T), wmin, 2000*cmn1, Nb, moment_scaling=2, atol=0, rtol=1e-10)


    H = SOP(N+1)
    sysinf = system_modes(N+1)

    sysinf[0] = generic_mode(7)
    for i in range(7):
        for m in range(Nb):
            sysinf[1 + i *Nb + m] = boson_mode(nbose)


    opdict = operator_dictionary_complex(N+1)

    H += sOP("Hsys", 0)
    for i in range(7):
        for m in range(Nb):
            H += np.sqrt(2)*g[m]*sOP("S"+str(i), 0)*sOP("q", 1+i*Nb+m)
            H += w[m]*sOP("n", 1+i*Nb+m)

    A = ttn(topo, dtype=np.complex128)
    state = np.zeros(N+1, dtype=int)
    state[0] =5
    A.set_state(state)

    observables = []
    opdict.insert(0, "Hsys", site_operator(Hsys(), optype="matrix", mode=0))
    for i in range(7):
        v = np.zeros(7)
        v[i] = 1.0
        op =  site_operator(v, optype="diagonal_matrix", mode=0)
        observables.append(op)
        opdict.insert(0, "S"+str(i),op)

    h = sop_operator(H, A, sysinf, opdict)

    mel = matrix_element(A)

    sweep = tdvp(A, h, krylov_dim = 8)
    sweep.dt = dt*fs
    sweep.coefficient = -1.0j
    sweep.use_time_dependent_hamiltonian = True

    res = np.zeros((7, nstep+1))

    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=[0, 1])
    lines = []
    for j in range(7):
        lines.append(ax.plot(np.arange(nstep+1)*dt, res[j, :])[0])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$n(t)$")

    res[0] = np.real(mel(op, A, A))
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
Nb = 16
single_set_fmo_dynamics(T, Nb, 1, 10, 0.5, 1000)
