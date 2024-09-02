import numpy as np
import time
import sys
import h5py

sys.path.append("../../")
from pyttn import *
from pyttn import oqs
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
        res += S/(s[0]+s[1]) * s[i]/(7*6*5*4*3*2*1*2*omega[i]**4) * np.power(w, 5)*np.exp(-np.sqrt(np.abs(w)/omega[i]))
    return res

def Ct(t, w, g):
    T, W = np.meshgrid(t, w)
    g2 = g**2
    fourier = np.exp(-1.0j*W*T)
    return g2@fourier

@jit(nopython=True)
def lorentzian(w, wk, gk, sk):
    return 4*wk*sk*gk*(wk**2+gk**2)*w/(np.pi*((w+wk)**2+gk**2)*((w-wk)**2+gk**2))#4*sk*gk*wk*w/np.pi/(((w-wk)**2+gk**2)*((w+wk)**2+gk**2))

@jit(nopython=True)
def Jh(w):
    wk = np.array([46, 68, 117, 167, 180, 191, 202, 243, 263, 284, 291, 327, 366, 385, 404, 423, 440, 481, 541, 568, 582, 597, 630, 638, 665, 684, 713, 726, 731, 750, 761, 770, 795, 821, 856, 891, 900, 924, 929, 946, 966, 984, 1004, 1037, 1058, 1094, 1104, 1123, 1130, 1162, 1175, 1181, 1201, 1220, 1283, 1292, 1348, 1367, 1386, 1431, 1503, 1545])*cmn1
    gk = 5*cmn1*np.ones(wk.shape)
    sk = np.array([0.011, 0.011, 0.009, 0.009, 0.010, 0.011, 0.011, 0.012, 0.003, 0.008, 0.008, 0.003, 0.006, 0.002, 0.002, 0.002, 0.001, 0.002, 0.004, 0.007, 0.004, 0.004, 0.003, 0.006, 0.004, 0.003, 0.007, 0.01, 0.005, 0.004, 0.009, 0.018, 0.007, 0.006, 0.007, 0.003, 0.004, 0.001, 0.001, 0.002, 0.002, 0.003, 0.001, 0.002, 0.002, 0.001, 0.001, 0.003, 0.003, 0.009, 0.007, 0.01, 0.003, 0.005, 0.002, 0.004, 0.007, 0.002, 0.004, 0.002, 0.003, 0.003])
    res = w*0.0
    for i in range(wk.shape[0]):
        res += lorentzian(w, wk[i], gk[i], sk[i])
        #res += 4*wk[i]*sk[i]*gk[i]*(wk[i]**2+gk[i]**2)*w/(np.pi*((w+wk[i])**2+gk[i]**2)*((w-wk[i])**2+gk[i]**2))
    return res

@jit(nopython=True)
def J2(w):
    wk = np.array([36, 70, 117, 173, 185, 195, 237, 260, 284, 327, 365, 381, 479, 541, 565, 580, 635, 714, 723, 730, 747, 759, 768, 777, 819, 859, 896, 1158, 1176, 1216])*cmn1
    sk = np.array([0.01, 0.01, 0.0055, 0.008, 0.008, 0.011, 0.005, 0.0025, 0.005, 0.0015, 0.002, 0.002, 0.001, 0.001, 0.002, 0.001, 0.003, 0.002, 0.003, 0.001, 0.002, 0.002, 0.004, 0.0015, 0.002, 0.0025, 0.002, 0.004, 0.003, 0.002])
    gk = 7.5*cmn1*np.ones(wk.shape)

    res = w*0.0
    for i in range(wk.shape[0]):
        res += lorentzian(w, wk[i], gk[i], sk[i])
        #res += 4*wk[i]*sk[i]*gk[i]*(wk[i]**2+gk[i]**2)*w/(np.pi*((w+wk[i])**2+gk[i]**2)*((w-wk[i])**2+gk[i]**2))

    return res
@jit(nopython=True)
def Jdebye(w):
    gj = 106.1*cmn1
    lj = 35*cmn1
    return (2.0/np.pi)*lj*gj*w/(gj*gj+w*w)


def Hsys():
    #return np.array([ [410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9], [-87.7, 530, 30.8, 8.2, 0.7, 11.8, 4.3], [5.5, 30.8, 210, -53.5, -2.2, -9.6, 6.0], [-5.9, 8.2, -53.5, 320, -70.7, -17.0, -63.3],  [6.7, 0.7, -2.2, -70.7, 480, 81.1, -1.3],[-13.7, 11.8, -9.6, -17.0, 81.1, 630, 39.7], [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 440]])*cmn1
    H = np.array([[310, -94.8, 5.5, -5.9, 7.1, -15.1, -12.2, 39.5 ], 
        [0, 230, 29.8, 7.6, 1.6, 13.1, 5.7, 7.9], 
        [0, 0, 0, -58.9, -1.2, -9.3, 3.4, 1.4],
        [0, 0, 0, 180, -64.1, -17.4, -62.3, -1.6],
        [0, 0, 0, 0, 405, 89.5, -4.6, 4.4], 
        [0, 0, 0, 0, 0, 320, 35.1, -9.1],
        [0, 0, 0, 0, 0, 0, 270, -11.1], 
        [0, 0, 0, 0, 0, 0, 0, 505]]
    )
    H = H+H.T
    np.fill_diagonal(H, H.diagonal()/2)
    print(H)
    return H*cmn1

def build_tree(degree, b_mode_dims, chi, nbose):
    ret = ntree("(1(8(8))(8))")

    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(ret()[1], [chi for i in range(8)], degree, chi)
    else:
        ntreeBuilder.mps_subtree(ret()[1], [chi for i in range(8)], chi, chi)
    

    leaf_indices = ret.leaf_indices()
    for li in leaf_indices[1:]:
        if(degree > 1):
            ntreeBuilder.mlmctdh_subtree(ret.at(li), b_mode_dims, degree, chi)
        else:
            ntreeBuilder.mps_subtree(ret.at(li), b_mode_dims, chi, min(chi, nbose))
        
    ntreeBuilder.sanitise(ret)
    return ret


#TODO: Need to figure out why the ipchain version of this function randomly changes the normalisation of the state we are interested in.
#TODO: Need to fix spectral density
#TODO: Need to figure fmo hamiltonian parameters
import copy
def single_set_fmo_dynamics(T, Nb, chi, nbose, dt, nstep, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, Ncut=4):
    @jit(nopython=True)
    def spectral_density(w):
        return (Jar(w)+Jh(w))

    import matplotlib.pyplot as plt
    wmax = 2000*cmn1
    w = np.linspace(0*cmn1, wmax,1000)
    plt.plot(w/cmn1, spectral_density(w)/cmn1)
    #plt.plot(w/cmn1, (J2(w))/cmn1)


    #set up the bath terms
    beta = None
    if T > 0:
        kb = 3.166811563e-6
        beta = 1.0/(kb*T)

    wmax = 2000*cmn1
    bath = oqs.bosonic_bath(spectral_density, sOP("S"+str(0), 0), beta=beta)
    g,w = bath.discretise(Nb, wmax, method='orthopol', rtol=1e-12)
    t = np.arange(nstep+1)*dt
    print(g, w)
    Ctb = bath.Ct(t, wmax=wmax, epsabs=1e-12, epsrel=1e-12, limit=100)
    Ct2 = Ct(t, w, g)
    plt.figure(2)
    plt.plot(t, np.real(Ctb), linewidth=4)
    plt.plot(t, np.imag(Ctb), linewidth=4)
    plt.plot(t, np.real(Ct2), '--', linewidth=2)
    plt.plot(t, np.imag(Ct2), '--', linewidth=2)
    plt.figure(1)


    N = Nb*8
    b_mode_dims = [max(4, min(nbose, int(wmax*Ncut/np.abs(w[i])))) for i in range(Nb)]

    mode_dims = [b_mode_dims[j] for i in range(8) for j in range(Nb)]
    print(mode_dims)

    #set up the operator dictionary defining operators acting on the system degree of freedom
    opdict = operator_dictionary_complex(N+1)
    observables = []
    opdict.insert(0, "Hsys", site_operator(Hsys(), optype="matrix", mode=0))
    for i in range(8):
        v = np.zeros(8)
        v[i] = 1.0
        op =  site_operator(v, optype="diagonal_matrix", mode=0)
        observables.append(copy.deepcopy(op))
        opdict.insert(0, "S"+str(i),op)


    chi0 = chi
    if adaptive:
        chi0 = 4

    #build the tree structures used to define the wavefunction
    topo = build_tree(degree, b_mode_dims, chi0, nbose)
    capacity = build_tree(degree, b_mode_dims, chi, nbose)
    print(topo)

    #set up the system Hamiltonian
    H = SOP(N+1)
    H += sOP("Hsys", 0)



    #set up the initial wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)
    state = np.zeros(N+1, dtype=int)
    state[0] = 0
    A.set_state(state)
    A.normalise()

    #set up the system information
    sysinf = system_modes(N+1)
    sysinf[0] = generic_mode(8)
    for i in range(8):
        for m in range(Nb):
            sysinf[1 + i *Nb + m] = boson_mode(b_mode_dims[m])


    plt.scatter(w/cmn1, g*g/np.amax(g*g)*np.amax(spectral_density(w)/cmn1))
    plt.show()
    for i in range(8):
        bath = oqs.bosonic_bath(spectral_density, sOP("S"+str(0), 0), beta=beta)
        Sp= sOP("S"+str(0), 0)
        #add the bath hamiltonian on
        H, w = oqs.add_bath_hamiltonian(H, bath.Sp, g, w, geom=geom, binds = [i*Nb+1+ x for x in range(Nb)])

    #construct the Hamiltonian
    h = sop_operator(H, A, sysinf, opdict)

    mel = matrix_element(A, 10)

    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 12)
    else:
        sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace')
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    if(geom == 'ipchain'):
        sweep.use_time_dependent_hamiltonian = True

    res = np.zeros((8, nstep+1))
    maxchi = np.zeros(nstep+1)
    for j in range(8):
        res[j, 0] = np.real(mel(observables[j], A, A))

    maxchi[0] = A.maximum_bond_dimension()
    for i in range(nstep):
        print(i)
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        print(i, t2-t1, mel(A, A))
        for j in range(8):
            res[j, i+1] = np.real(mel(observables[j], A, A))
        sys.stdout.flush()
        maxchi[i+1] = A.maximum_bond_dimension()

        if(i % 10 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt/fs))
            for j in range(8):
                h5.create_dataset('n_'+str(j), data=res[j, :])
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()
        
    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt/fs))
    for j in range(8):
        h5.create_dataset('n_'+str(j), data=res[j, :])
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('--T', type = float, default = 0)

    #number of bath modes
    parser.add_argument('--N', type=int, default=48)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=36)
    parser.add_argument('--degree', type=int, default=2)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--tmax', type=float, default=1000)

    #output file name
    parser.add_argument('--fname', type=str, default='fmo.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)
    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1

    single_set_fmo_dynamics(args.T, args.N, args.chi, args.nbose, args.dt*fs, nstep, geom=args.geom, ofname=args.fname, degree = args.degree, adaptive=args.subspace, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold)

