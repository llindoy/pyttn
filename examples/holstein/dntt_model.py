import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import h5py

from pyttn import ntree, ntreeBuilder
from pyttn import system_modes, boson_mode
from pyttn import ttn, sop_operator, matrix_element, tdvp, site_operator, sOP
from pyttn.utils import visualise_tree, ModeCombination
from pyttn import ms_ttn, ms_sop_operator, multiset_SOP

import matplotlib.pyplot as plt

def gen_lattice(a, b, N):
    res = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            if j % 2 == 0:
                res[i, j, :] = np.array( [ i*a, (j//2)*b, i*N+j])
            else:
                res[i, j, :] = np.array( [ i*a+a/2, (j//2)*b+b/2, i*N+j])
    return res

def partition_points_x(x):
    if x.shape[0] % 2 == 0:
        skip = x.shape[0]//2
    else:
        skip = (x.shape[0]+1)//2
    return x[:skip, :], x[skip:, :]

def partition_points_y(x):
    if x.shape[1] % 2 == 1:
        skip = x.shape[1]//2
    else:
        skip = (x.shape[1]+1)//2
    return x[:, :skip], x[:, skip:]


def partition_and_get_path(x, path = None):
    if path is None:
        path = []

    lists = []

    #partition the two sets of points
    if x.shape[0] > 1:
        xs = [*partition_points_x(x)]
    elif x.shape[1] > 1:
        xs = [*partition_points_y(x)]
    else:
        return [ [path, x[0,0,2]]]


    for i in range(2):
        curr_path = path + [i]
        partition = False

        if(xs[i].shape[1] > 1):
            x2s = [*partition_points_y(xs[i])]
            partition=True
        elif xs[i].shape[0] > 1:
            x2s = [*partition_points_x(xs[i])]
            partition=True
        else:
            lists.append([curr_path, xs[i][0,0,2]])
        
        if partition:
            for j in range(2):
                fpath = curr_path + [j]
                lists = lists + partition_and_get_path(x2s[j], fpath)

    return lists

def plot_points(x):
    for xi in x:
        plt.scatter(xi[:, :, 0], xi[:, :, 1], 2+xi[:,:, 2]/10)


def tree_index_to_site_index(lists):
    return [int(li[1]) for li in lists]


def invert_indexing(inds):
    res = [0 for i in inds]
    for i in range(len(inds)):
        res[inds[i]] = i
    return res

def expand_nodes(inds, N):
    res = []
    for i in inds:
        for j in range(N):
            res.append(i*N+j)
    return res


def run_initial_step(A, h, sweep, dt, nstep=10):
    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), nstep)
    for i in range(nstep):
        dti = ts[i]-tp
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    return A, h, sweep

def output_results(ofname, timepoints, res, runtime):
    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=timepoints)

    N2 = res.shape[1]
    N = int(np.sqrt(N2))
    for i in range(N):
        for j in range(N):
            h5.create_dataset("|%d,%d><%d,%d|"%(i, j, i, j), data=res[:, i*N+j])
    h5.create_dataset('runtime', data=runtime*np.ones(1))
    h5.close()


def dntt_dynamics(N, chi1, beta = None, chi2 = None, tmax=200, dt=0.25, nbose=10, nbosep = 40, ofname='dntt.h5', output_skip=1):
    #here we are considering a model in which each site couples to three neighbours and includes
    #both holstein modes and peirels modes to each of its neighbours

    a = 7.662
    b = 6.187

    g = 0.1
    w0 = 1

    gp = [0.2, 0.2, 0.2]
    wp =[0.1, 0.1, 0.1]
    J = [1, 0.3, -1.2]

    #set up the time evolution parameters
    nsteps = int(tmax/(dt))+1
    t = np.arange(nsteps+1)*dt


    #set up the system information
    sysinf = system_modes(4*N*N)

    for i in range(N*N):
        if beta is None:
            sysinf[4*i] = boson_mode(nbose)
            sysinf[4*i+1] = boson_mode(nbosep)
            sysinf[4*i+2] = boson_mode(nbosep)
            sysinf[4*i+3] = boson_mode(nbosep)
        else:
            sysinf[4*i] = [boson_mode(nbose), boson_mode(nbose)]
            sysinf[4*i+1] = [boson_mode(nbosep), boson_mode(nbosep)]
            sysinf[4*i+2] = [boson_mode(nbosep), boson_mode(nbosep)]
            sysinf[4*i+3] = [boson_mode(nbosep), boson_mode(nbosep)]

    vib_dims = [sysinf.mode(i).lhd() for i in range(4)]

    #set up the Hamiltonian
    if beta is None:
        H = multiset_SOP(N*N, 4*N*N)
    else:
        H = multiset_SOP(N*N, 8*N*N)

    #generate the mapping from square lattice to tree indices
    lattice_points = gen_lattice(a, b, N)
    
    #add on the electronic coupling and peirel terms
    for i in range(N):
        for j in range(N):
            i0 = i*N+j

            if j%2 == 0:
                n1 = ((i+1)%N)*N+j
                n2 = i*N+((j+1)%N)
                n3 = ((i-1+N)%N)*N+((j+1)%N)
            else:
                n1 = ((i+1)%N)*N+j
                n2 = ((i+1)%N)*N+((j+1)%N)
                n3 = i*N+((j+1)%N)

            #add on the  
            #o       o
            #    
            #    o ----- o 
            #terms
            H[i0, n1] += J[0]
            H[n1, i0] += J[0]
            if beta is None:
                H[i0, n1] += gp[0]*(sOP("adag", 4*i+1)+sOP("a", 4*i+1))
                H[n1, i0] += gp[0]*(sOP("adag", 4*i+1)+sOP("a", 4*i+1))
            else:
                gpp = gp[0]*(0.5*(1+1.0/np.tanh( beta*wp[0]/2)))
                gpm = gp[0]*(0.5*(1+1.0/np.tanh(-beta*wp[0]/2)))
                H[i0, n1] += float(gpp)*(sOP("adag", 8*i+2)+sOP("a", 8*i+2))
                H[n1, i0] += float(gpp)*(sOP("adag", 8*i+2)+sOP("a", 8*i+2))
                H[i0, n1] += float(gpm)*(sOP("adag", 8*i+3)+sOP("a", 8*i+3))
                H[n1, i0] += float(gpm)*(sOP("adag", 8*i+3)+sOP("a", 8*i+3))
            #add on the  
            #o       o
            #      -
            #    o       o 
            #terms
            H[i0, n2] += J[1]
            H[n2, i0] += J[1]
            if beta is None:
                H[i0, n2] += gp[1]*(sOP("adag", 4*i+2)+sOP("a", 4*i+2))
                H[n2, i0] += gp[1]*(sOP("adag", 4*i+2)+sOP("a", 4*i+2))
            else:
                gpp = gp[1]*(0.5*(1+1.0/np.tanh( beta*wp[1]/2)))
                gpm = gp[1]*(0.5*(1+1.0/np.tanh(-beta*wp[1]/2)))
                H[i0, n2] += float(gpp)*(sOP("adag", 8*i+4)+sOP("a", 8*i+4))
                H[n2, i0] += float(gpp)*(sOP("adag", 8*i+4)+sOP("a", 8*i+4))
                H[i0, n2] += float(gpm)*(sOP("adag", 8*i+5)+sOP("a", 8*i+5))
                H[n2, i0] += float(gpm)*(sOP("adag", 8*i+5)+sOP("a", 8*i+5))
            #add on the  
            #o       o
            #  -
            #    o       o 
            #terms
            H[i0, n3] += J[2]
            H[n3, i0] += J[2]
            if beta is None:
                H[i0, n3] += gp[2]*(sOP("adag", 4*i+3)+sOP("a", 4*i+3))
                H[n3, i0] += gp[2]*(sOP("adag", 4*i+3)+sOP("a", 4*i+3))
            else:
                gpp = gp[2]*(0.5*(1+1.0/np.tanh( beta*wp[2]/2)))
                gpm = gp[2]*(0.5*(1+1.0/np.tanh(-beta*wp[2]/2)))
                H[i0, n2] += float(gpp)*(sOP("adag", 8*i+6)+sOP("a", 8*i+6))
                H[n2, i0] += float(gpp)*(sOP("adag", 8*i+6)+sOP("a", 8*i+6))
                H[i0, n2] += float(gpm)*(sOP("adag", 8*i+7)+sOP("a", 8*i+7))
                H[n2, i0] += float(gpm)*(sOP("adag", 8*i+7)+sOP("a", 8*i+7))

    #add on the purely bosonic terms
    for i in range(N*N):
        for j in range(N*N):
            if beta is None:
                H[i, i] += w0*sOP("n", 4*j+0)
                for x in range(3):
                    H[i, i] += wp[x]*sOP("n", 4*j+x+1)
            else:
                H[i, i] +=  w0*sOP("n", 8*j+0)
                H[i, i] += -w0*sOP("n", 8*j+1)
                for x in range(3):
                    H[i, i] += wp[x]*sOP("n", 8*j+2*x+2)
                    H[i, i] +=-wp[x]*sOP("n", 8*j+2*x+3)

    #now add on the holstein mode coupling terms
    for i in range(N*N):
        if beta is None:
            H[i, i] += g*(sOP("adag", 4*i)+sOP("a", 4*i))
        else:
            gp = g*(0.5*(1+1.0/np.tanh(beta*w0/2)))
            gm = g*(0.5*(1+1.0/np.tanh(-beta*w0/2)))   
            H[i, i] += float(gp)*(sOP("adag", 8*i)+sOP("a", 8*i))
            H[i, i] += float(gm)*(sOP("adag", 8*i+1)+sOP("a", 8*i+1))

    #build the ML-MCTDH tree for the vibrational modes
    class chi_step:
        def __init__(self, chimax, chimin, N, degree = 2):
            self.chimin = chimin
            if N%degree == 0:
                self.Nl = int(int(np.log(N)/np.log(degree)))
            else:
                self.Nl = int(int(np.log(N)/np.log(degree))+1)

            self.nx = int((chimax-chimin)//(self.Nl-1))

        def __call__(self, l):
            ret=int((self.Nl-l)*self.nx+self.chimin)
            return ret
    topo = ntreeBuilder.mlmctdh_tree([chi2 for i in range(N*N)], 2, chi_step(chi1, chi2, N*N))

    #add on nodes for the vibrational modes associated with each site.  This is the holstein modes
    #plus the peirels modes coupling to the right and the two modes coupling upwards
    linds = topo.leaf_indices()
    for li in linds:
        ntreeBuilder.mlmctdh_subtree(topo.at(li), vib_dims, 2, chi2)

    lists = partition_and_get_path(lattice_points)
    conv = tree_index_to_site_index(lists)
    site_to_tree = invert_indexing(conv)
    site_to_tree = expand_nodes(site_to_tree, 4)
    sysinf.mode_indices = site_to_tree

    #set up the wavefunction for simulating the dynamics
    A = ms_ttn(N*N, topo, dtype=np.complex128)
    state = [[0 for i in range(4*N*N)] for j in range(N*N)]
    coeff = np.zeros(N*N, dtype=np.float64)
    coeff[(N//2)*(N+1)] = 1
    A.set_state(coeff, state)
    
    #setup the matrix element calculation object
    mel = matrix_element(A)
    
    #set up the sum of product operator object for evolution
    h = ms_sop_operator(H, A, sysinf)

    #setup the evolution object
    sweep = tdvp(A, h, krylov_dim = 12)
    sweep.expmv_tol=1e-10
    sweep.dt = dt
    sweep.coefficient = -1.0j
    
    #setup buffers for storing the results
    res = np.zeros((nsteps+1, N*N), dtype=np.complex128)
    maxchi = np.zeros((nsteps+1))
    
    for j in range(N*N):
        res[0, j] = mel(A.slice(j))

    t1 = time.time()
    
    #perform a set of steps with a logarithmic timestep discretisation
    A, h, sweep = run_initial_step(A, h, sweep, dt)
    
    for j in range(N*N):
        res[1, j] = mel(A.slice(j))

    for j in range(N*N):
        res[1, j] = mel(A.slice(j))
    sweep.dt = dt
    
    timepoints = (np.arange(nsteps+1)*dt)
    
    #perform the remaining dynamics steps
    for i in range(1, nsteps):
        print(i, nsteps, flush=True)
        sweep.step(A, h)
        
        t2 = time.time()
    
        for j in range(N*N):
            res[i+1, j] = mel(A.slice(j))
    
        if(i % output_skip == 0):
            output_results(ofname, timepoints, res, (t2-t1))
    
    t2 = time.time()
    output_results(ofname, timepoints, res, (t2-t1))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyttn test")

    parser.add_argument('N', type=int)

    parser.add_argument('chi', type=int)
    parser.add_argument('--chi2', type=int, default=None)
    parser.add_argument('--nbose', type=int, default=8)
    parser.add_argument('--nbosep', type=int, default=8)
    parser.add_argument('--beta', type=float, default=None)

    parser.add_argument('--fname', type=str, default='dntt.h5')

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--tmax', type=float, default=5)

    parser.add_argument('--output_skip', type=int, default=1)
    args = parser.parse_args()

    dntt_dynamics(
            args.N,
            args.chi, 
            nbose = args.nbose,
            beta = args.beta,
            chi2=args.chi2, 
            tmax=args.tmax,
            dt=args.dt,
            output_skip=args.output_skip,
            ofname=args.fname
    )


