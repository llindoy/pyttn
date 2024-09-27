import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs
from pyttn._pyttn import operator_dictionary_complex
from pyttn._pyttn.linalg import csr_matrix_complex, diagonal_matrix_complex

from numba import jit

#a function for mapping to a block chain form
def block_chain_map(M, bs = 1, tol = 1e-14):
    s = M.shape[0]
    if(s % bs != 0):
        raise RuntimeError("block chain mapping expects matrices that have dimension that is a multiple of the block size.")

    Niter = s//bs
    Vm = np.zeros((s, bs), dtype=np.complex128)
    Q = np.zeros((s, s), dtype=np.complex128)

    Hm = np.zeros((s+bs, s), dtype=np.complex128)

    for i in range(bs):
        Vm[i,i] = 1

    def ind(i):
        return i*bs

    Niter = 2
    #something is going horrible wrong in the orthogonalisation step of this code
    for j in range(0, Niter):
        print("m ortho", j, np.conj(Vm).T@Vm)

        #step 2 - append the jth column
        Q[:,ind(j):ind(j+1)] = copy.deepcopy(Vm)

        #step 3 - compute action of the operator on our current vector
        W = np.zeros((s, bs), dtype=np.complex128)
        W = M@Q[:, ind(j):ind(j+1)]

        #step 4 - compute the hamiltonian terms
        #this is the H_{j, j}
        Hm[ind(j):ind(j+1), ind(j):ind(j+1)] = np.conj(Q[:, ind(j):ind(j+1)]).T@W

        #step 5 set the rest of the Hamiltonian terms - set all of the rows of the final column
        Hm[:ind(j+1),ind(j):ind(j+1)] = np.conj(Q[:, :ind(j+1)]).T@W

        for i in range(j):
            print(W.shape, bs)
            print(i, j, np.conj(Q[:, i*bs:(i+1)*bs]).T@W)

        print("val", W)
        print("val2", Q[:,:ind(j+1)]@Hm[:ind(j+1), ind(j):ind(j+1)])
        W -= Q[:,:ind(j+1)]@Hm[:ind(j+1), ind(j):ind(j+1)]
        for i in range(j):
            print(i, j, np.conj(Q[:, i*bs:(i+1)*bs]).T@W)
#           W -= Q[:, i*bs:(i+1)*bs]@Hm[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
        
        Vm, Rj = np.linalg.qr(W)
        print(j, j, np.conj(Vm).T@Vm)
        Hm[ind(j+1):ind(j+2), ind(j):ind(j+1)] = copy.deepcopy(Rj)

    exit(1)
    return Hm[:s, :], Q

def remove_zeros(M, tol=1e-14):
    M[np.abs(M) < tol] = 0.0
    return M


def visualise_time_dep_coeffs(dk, zk, dt, nstep):
    t = np.arange(nstep+1)*dt

    Nb = dk.shape[0]*2
    M = np.zeros((Nb+2, Nb+2), dtype = np.complex128)
    for i in range(Nb):
        gk = 0
        hk = 0
        wk = 0

        if(i%2 == 0):
            wk = -1.0j*zk[i//2]
            gk = dk[i//2]/np.sqrt(np.abs(dk[i//2]))
            hk = np.sqrt(np.abs(dk[i//2]))
            M[0, i+2] = gk
        else:
            wk = -1.0j*np.conj(zk[i//2])
            gk = -np.conj(dk[i//2])/np.sqrt(np.abs(dk[i//2]))
            hk = np.sqrt(np.abs(dk[i//2]))
            M[1, i+2] = gk
        #add on the S^L mapping term
        M[i+2, 0] = hk
        M[i+2, 1] =-hk
        M[i+2, i+2] = wk
    import matplotlib.pyplot as plt


    M2, Q = block_chain_map(M, bs=2)

    HI_tilde = copy.deepcopy(M2)
    HI_tilde[2:, 2:] *= 0.0
    plt.spy(HI_tilde)
    plt.show()
    T = Q@np.conj(Q).T
    T = remove_zeros(T, tol=1e-6)
    print(T)
    plt.spy(T)
    plt.show()

    #print(M2, Q)
    exit()

    #class func_class:
    #    def __init__(self, i, t0, e0, X, Xp, conj = False):
    #        self.i = copy.deepcopy(i)
    #        self.conj=conj
    #        self.t0 = copy.deepcopy(t0)
    #        self.e = copy.deepcopy(e0)
    #        self.U = copy.deepcopy(U0)

    #    def __call__(self, ti):
    #        val = self.t0*np.conj(self.U[:, 0])@(np.exp(-1.0j*ti*self.e)*self.U[:, self.i])

    #        if(self.conj):
    #            val = np.conj(val)

    #        return val

def sbm_dynamics(alpha, wc, s, eps, delta, chi, nbose, dt, nbose_min = None, beta = None, Ncut = 20, nstep = 1, Nw = 20, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, aaa_tol=1e-4):

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.bosonic_bath(J, sOP("sz", 0), beta=beta)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    dk,zk, Sw_aaa = bath.fitCt(wmax=Nw*wc, aaa_tol=aaa_tol)
    w = np.linspace(-Nw*wc, Nw*wc, 1000)

    Nb = 2*dk.shape[0]
    N = Nb+1

    Nt = Nb
    N = Nt+1

    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    b_mode_dims = [8 for x in range(Nt)]
    mode_dims = [4] + b_mode_dims#oqs.heom.compute_dimensions(sz, dk, zk, nbose, Lmin=nbose_min)
    mode_dims =np.array(mode_dims)
    b_mode_dims = np.array(b_mode_dims)
    print(N)
    print(mode_dims.shape)

    #set up the total Hamiltonian
    H = SOP(N)

    #and add on the system parts
    Hsys = eps*sz + delta *sx
    H += sOP("lsys", 0)
    
    visualise_time_dep_coeffs(dk, zk, dt, nstep)
    exit()
    M = np.zeros((Nb+2, Nb+2), dtype = np.complex128)
    for i in range(Nb):
        gk = 0
        hk = 0
        wk = 0

        if(i%2 == 0):
            wk = -1.0j*zk[i//2]
            gk = dk[i//2]/np.sqrt(np.abs(dk[i//2]))
            hk = np.sqrt(np.abs(dk[i//2]))
            M[0, i+2] = gk
        else:
            wk = -1.0j*np.conj(zk[i//2])
            gk = -np.conj(dk[i//2])/np.sqrt(np.abs(dk[i//2]))
            hk = np.sqrt(np.abs(dk[i//2]))
            M[1, i+2] = gk
        #add on the S^L mapping term
        M[i+2, 0] = hk
        M[i+2, 1] =-hk
        M[i+2, i+2] = wk

    M = M.T
    M2, Q = block_chain_map(M, bs=2)
    M = M.T
    M2 = M2.T
    M2 = remove_zeros(M2, 1e-12)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(np.abs(M))
    plt.figure(2)
    plt.imshow(np.abs(M2))
    plt.show()
    #plt.figure(1)
    #plt.spy(M)
    #plt.figure(2)
    #plt.spy(M2)

    #plt.figure(3)
    #plt.plot(np.abs(np.diag(M)))
    #plt.plot(np.abs(np.diag(M2)))

    #plt.figure(4)
    #plt.plot(np.abs(M[2:, 0]), 'k')
    #plt.plot(np.abs(M2[2:, 0]), 'r')

    #plt.plot(np.abs(M[0, 2:]+M[1, 2:]), 'k--')
    #plt.plot(np.abs(M2[0, 2:]+M2[1, 2:]), 'r--')
    #plt.show()
    #for i in range(Nb):
    #    print(i, M[i+2, i+2], M2[i+2, i+2])
    
    v = []
    vc = []
    v.append(sOP("sm_f", 0))
    v.append(sOP("sm_b", 0))
    for i in range(Nb):
        v.append(sOP("a", i+1))

    vc.append(sOP("sm_f", 0))
    vc.append(sOP("sm_b", 0))
    for i in range(Nb):
        vc.append(sOP("adag", i+1))

    for i in range(Nt+2):
        for j in range(Nt+2):
            if(np.abs(M[i, j]) > 1e-12):
                H += M[i, j]*vc[i]*v[j]
    #for i in range(Nb):
    #    gk = 0
    #    hk = 0
    #    wk = 0

    #    if(i%2 == 0):
    #        wk = -1.0j*zk[i//2]
    #        gk = dk[i//2]/np.sqrt(np.abs(dk[i//2]))
    #        hk = np.sqrt(np.abs(dk[i//2]))
    #    else:
    #        wk = -1.0j*np.conj(zk[i//2])
    #        gk = -np.conj(dk[i//2])/np.sqrt(np.abs(dk[i//2]))
    #        hk = np.sqrt(np.abs(dk[i//2]))

    #    H += hk*sOP("sp", 0)*sOP("a", i+1)

    #    if i % 2 == 0:
    #        H += gk*sOP("sm_f", 0)*sOP("adag", i+1)
    #    else:
    #        H += gk*sOP("sm_b", 0)*sOP("adag", i+1)
    #    H += wk*sOP("n", i+1)

    #set up the operator dictionary information
    opdict = operator_dictionary_complex(N)

    #set up the different system operators
    opdict.insert(0, "lsys", site_operator(oqs.heom.commutator(Hsys), optype="matrix", mode=0))
    opdict.insert(0, "sp", site_operator(oqs.heom.Sp(sz), optype="matrix", mode=0))
    opdict.insert(0, "sm_f", site_operator(oqs.heom.Sm(sz, True), optype="matrix", mode=0))
    opdict.insert(0, "sm_b", site_operator(oqs.heom.Sm(sz, False), optype="matrix", mode=0))

    #setup the system information object
    sysinf = system_modes(N)
    sysinf[0] = generic_mode(mode_dims[0])
    for i in range(Nt):
        sysinf[i+1] = boson_mode(b_mode_dims[i])

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

    #and add the node that forms the root of the bath.  
    #TODO: Add some better functions for handling the construction of tree structures
    topo = ntree("(1(%d(%d))(%d))"%(mode_dims[0], mode_dims[0], mode_dims[0]))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], b_mode_dims, degree, chi0)
    else:
        ntreeBuilder.mps_subtree(topo()[1], b_mode_dims, chi0, min(chi0, nbose))
    ntreeBuilder.sanitise(topo)

    capacity = ntree("(1(%d(%d))(%d))"%(mode_dims[0], mode_dims[0], mode_dims[0]))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(capacity()[1], b_mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(capacity()[1], b_mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(capacity)


    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(Nt+1)])
    print(topo)

    h = sop_operator(H, A, sysinf, opdict)

    #set up ttns storing the observable to be measured.  Here these are just system observables
    #so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = ntree("(1(%d(%d))(1))"%(mode_dims[0], mode_dims[0]))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(obstree()[1], b_mode_dims, degree, 1)
    else:
        ntreeBuilder.mps_subtree(obstree()[1], b_mode_dims, 1, min(chi0, nbose))
    ntreeBuilder.sanitise(obstree)


    Sz_ttn = ttn(obstree, dtype=np.complex128)
    prod_state = [sz.flatten()]
    for i in range(Nt):
        state_vec = np.zeros(b_mode_dims[i], dtype=np.complex128)
        state_vec[0] = 1.0
        prod_state.append(state_vec)
    Sz_ttn.set_product(prod_state)

    nops = []

    for i in range(Nt):
        nop_tree = ttn(obstree, dtype=np.complex128)
        nprod_state = [np.identity(2).flatten()]
        for j in range(Nt):
            state_vec = np.zeros(b_mode_dims[j], dtype=np.complex128)
            if j == i:
                state_vec = np.arange(b_mode_dims[j])*(1.0+0.0j)
            else:
                state_vec[0] = 1.0
            nprod_state.append(state_vec)
        nop_tree.set_product(nprod_state)
        nops.append(copy.deepcopy(nop_tree))

    mel = matrix_element(A)

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

    res = np.zeros(nstep+1)
    resN = np.zeros((nstep+1, Nt))
    maxchi = np.zeros(nstep+1)
    res[0] = np.real(mel(Sz_ttn, A))
    maxchi[0] = A.maximum_bond_dimension()

    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        res[i+1] = np.real(mel(Sz_ttn, A))
        maxchi[i+1] = A.maximum_bond_dimension()
        for j in range(Nt):
            resN[i+1, j] = np.abs(mel(nops[j], A))

        print((i+1)*dt, res[i+1], maxchi[i+1], np.real(mel(A, A)))
        sys.stdout.flush()
        if(i % 100 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            h5.create_dataset('Sz', data=res)
            for j in range(Nt):
                h5.create_dataset('N_'+str(j), data=resN[:, j])
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    h5.create_dataset('Sz', data=res)
    for j in range(Nt):
        h5.create_dataset('N_'+str(j), data=resN[:, j])
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('alpha', type = float)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)

    #number of bath modes
    parser.add_argument('--aaatol', type=float, default=1e-1)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=0.5)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=32)
    parser.add_argument('--degree', type=int, default=1)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=30)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-6)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, aaa_tol=args.aaatol, nbose_min=5)
