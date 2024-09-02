import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs

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

    for j in range(0, Niter):
        Q[:,j*bs:(j+1)*bs] = copy.deepcopy(Vm)
        Qj = np.zeros((s, bs), dtype=np.complex128)
        Qj = M@Q[:, j*bs:(j+1)*bs]

        Hm[j*bs:(j+1)*bs, j*bs:(j+1)*bs] = np.conj(Q[:, j*bs:(j+1)*bs]).T@Qj
        for i in range(j):
            Hm[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = np.conj(Q[:, i*bs:(i+1)*bs]).T@Qj 
            Qj -= Q[:, i*bs:(i+1)*bs]@Hm[i*bs:(i+1)*bs, j*bs:(j+1)*bs]

        Qj -= Q[:, j*bs:(j+1)*bs]@Hm[j*bs:(j+1)*bs, j*bs:(j+1)*bs]

        
        Qj, Rj = np.linalg.qr(Qj)
        Vm = copy.deepcopy(Qj)
        Hm[(j+1)*bs:(j+2)*bs, (j+0)*bs:(j+1)*bs] = copy.deepcopy(Rj)

    return Hm[:s, :], Q

def remove_block_chain_zeros(M, bs):
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            if(np.abs(i-j) > bs):
                M[i, j] = 0.0
    return M

def setup_bath_tree(node, mode_dims, chi, nbose, degree):
    node.insert(2)
    node[0].insert(2)

    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(node, mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(node, mode_dims, chi, min(chi, nbose))

def setup_tree(Ns, mode_dims, chis, chib, nbose, degree_spins = 2, degree_bath = 2):
    topo = None
    if(Ns == 1):
        topo = ntree("(1(2(2))(2))")
        if(degree_bath > 1):
            ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree_bath, chib)
        else:
            ntreeBuilder.mps_subtree(topo()[1], mode_dims, chib, min(chib, nbose))
    else:
        if(degree_spins > 1):
            topo = ntreeBuilder.mlmctdh_tree([chis for i in range(Ns)], degree_spins, chis)
        else:
            topo = ntreeBuilder.mps_tree([chis for i in range(Ns)], chis, chis)

        leaf_indices = topo.leaf_indices()
        for li in leaf_indices:
            setup_bath_tree(topo.at(li), mode_dims, chib, nbose, degree_bath)
    ntreeBuilder.sanitise(topo)
    return topo

def msb_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, nu=1, xi = None, beta = None, Ncut = 20, nstep = 1, Nw = 7.5, geom='star', ofname='msb.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.bosonic_bath(J, sOP("sz", 0), beta=beta)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    gs,ws = bath.discretise(Nb, Nw*wc, method='orthopol')


    gks = None
    Ns = 1
    if isinstance(xi, np.ndarray):
        Ns = xi.shape[0]
        Jij = np.zeros((Nb, Ns, Ns))
        for i in range(Ns):
            Jij[:, i, i] = gs*gs
            for j in range(i+1, Ns):
                print(np.cos(ws*(xi[i]-xi[j])/nu))
                Jij[:, i, j] = gs*gs*np.cos(ws*(xi[i]-xi[j])/nu)
                Jij[:, j, i] = gs*gs*np.cos(ws*(xi[i]-xi[j])/nu)
        gks = np.zeros((Nb, Ns, Ns))
        for i in range(Nb):
            u, v = np.linalg.eigh(Jij[i, :, :])
            gks[i, :, :] = v@np.diag(np.sqrt(np.abs(u)))@np.conj(v).T
    else:
        Ns = 1
        gks = np.zeros((Nb, Ns, Ns))
        gks[:, 0, 0] = gs


    M = np.zeros((Ns*(Nb + 1), Ns*(Nb + 1)))
    for j in range(Ns):
        for i in range(Nb):
            M[Ns*(i+1)+j, Ns*(i+1)+j] = ws[i]

    for i in range(Ns):
        for j in range(Ns):
            for k in range(Nb):
                M[i, Ns*(k + 1) + j] = gks[k, i, j]
                M[Ns*(k+1) + j, i] = gks[k, i, j]



    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.spy(M)

    M2, Q = block_chain_map(M, bs=Ns)
    M2 = remove_block_chain_zeros(M2, Ns)
    plt.figure(2)
    plt.spy(M2)
    plt.show()

    chi0 = chi
    if adaptive:
        chi0 = 4

    mode_dims = [min(max(4, int(wc*Ncut/ws[i])), nbose) for i in range(Nb)]
    topo = setup_tree(Ns, mode_dims, chi0, chi0, nbose, degree, degree)
    capacity = setup_tree(Ns, mode_dims, chi, chi, nbose, degree, degree)

    #ct = bath.Ct(t, Nw*wc)
    tau = 1.0

    N = Ns*(Nb+1)

    #setup the system information object
    sysinf = system_modes(N)
    vs = []
    vsd = []
    mode_ordering = []
    for i in range(Ns):
        sysinf[i*(Nb+1)] = spin_mode(2)
        vs.append(sOP("sz", i*(Nb+1)))
        vsd.append(sOP("sz", i*(Nb+1)))

    for i in range(Ns):
        for k in range(Nb):
            mind = i*(Nb+1)+k+1
            sysinf[mind] = boson_mode(mode_dims[k])
            vs.append(sOP("a", mind))
            vsd.append(sOP("adag", mind))

    #TODO Need to fix the mode indices types
    print(sysinf.mode_indices)

    #set up the total Hamiltonian
    H = SOP(N)

    #and add on the system parts
    for i in range(Ns):
        H += eps*sOP("sz", i*(Nb+1))
        H += delta*sOP("sx", i*(Nb+1))

    for i in range(N):
        for j in range(N):
            if( np.abs(M[i, j]) > 1e-12):
                H += M[i, j]*vs[i]*vsd[j]
    print(H)


    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(N)])

    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)
    mel = matrix_element(A)

    ops = []
    for i in range(Ns):
        op = site_operator_complex(
            sOP("sz", i*(Nb+1)),
            sysinf
        )
        ops.append(op)

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

    res = np.zeros((nstep+1, Ns))
    maxchi = np.zeros(nstep+1)
    for i in range(Ns):
        res[0, i] = np.real(mel(ops[i], A, A))
    maxchi[0] = A.maximum_bond_dimension()

    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        for j in range(Ns):
            res[i+1, j] = np.real(mel(ops[j], A, A))
        maxchi[i+1] = A.maximum_bond_dimension()
        print(i)
        sys.stdout.flush()

        if(i % 10 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            for j in range(Ns):
                h5.create_dataset('Sz_'+str(j), data=res[:, j])
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    for j in range(Ns):
        h5.create_dataset('Sz_'+str(j), data=res[:, j])
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
    parser.add_argument('--N', type=int, default=256)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=1)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--degree', type=int, default=2)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=30)

    #output file name
    parser.add_argument('--fname', type=str, default='msb.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()
    xi = np.arange(2)

    nstep = int(args.tmax/args.dt)+1
    msb_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, xi=xi, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree)
