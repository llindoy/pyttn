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
    for j in range(res.shape[1]):
        h5.create_dataset("|%d><%d|"%(j, j), data=res[:, j])
    h5.create_dataset('runtime', data=runtime*np.ones(1))
    h5.close()


def holstein_dynamics(ansatz, g, w0, J, N, chi1, beta = None, chi2 = None, tmax=200, dt=0.25, nbose=10, ofname='holstein_1d.h5', output_skip=1, use_mode_combination=True, nbmax=2, nhilbmax=1024):
    #set up the time evolution parameters
    nsteps = int(tmax/(dt))+1
    t = np.arange(nsteps+1)*dt

    if beta is None:
        Nmodes = N  
    else:
        Nmodes = 2*N

    #set up the system information
    sysinf = system_modes(Nmodes)
    for i in range(Nmodes):
        sysinf[i] = boson_mode(nbose)

    #use mode combination on the bosonic modes
    if(use_mode_combination):
        mode_comb = ModeCombination(nbmax, nhilbmax)
        sysinf = mode_comb(sysinf)

    #set up the Hamiltonian
    H = multiset_SOP(N, Nmodes)
    
    #add on the electronic coupling terms
    for i in range(N):
        j = (i+1)%N
        H[i, j] += J
        H[j, i] += J

    #add on the purely bosonic terms
    for i in range(N):
        for j in range(N):
            if beta is None:
                H[i, i] += w0*sOP("n", j)
            else:
                H[i, i] += w0*sOP("n", 2*j)
                H[i, i] += -w0*sOP("n", 2*j+1)

    #now add on the system bath coupling terms
    for i in range(N):
        if beta is None:
            H[i, i] += g*(sOP("adag", i)+sOP("a", i))
        else:
            gp = g*(0.5*(1+1.0/np.tanh(beta*w0/2)))
            gm = g*(0.5*(1+1.0/np.tanh(-beta*w0/2)))   
            H[i, i] += gp*(sOP("adag", 2*i)+sOP("a", 2*i))
            H[i, i] += gm*(sOP("adag", 2*i+1)+sOP("a", 2*i+1))

    if ansatz == "mps":
        topo = ntreeBuilder.mps_tree(sysinf.mode_dimensions(), chi1)
        
    elif ansatz == "ttn":
        if chi2 is None:
            topo = ntreeBuilder.mlmctdh_tree(sysinf.mode_dimensions(), 2, chi1)
        else:
            #vary the bond dimension throughout the tree
            class chi_step:
                def __init__(self, chimax, chimin, N, degree = 2):
                    self.chimin = chimin
                    if N%degree == 0:
                        self.Nl = int(int(np.log(N)/np.log(degree))+1)
                    else:
                        self.Nl = int(int(np.log(N)/np.log(degree))+2)
            
                    self.nx = int((chimax-chimin)//self.Nl)
            
                def __call__(self, l):
                    ret=int((self.Nl-l)*self.nx+self.chimin)
                    return ret
            topo = ntreeBuilder.mlmctdh_tree(sysinf.mode_dimensions(), 2, chi_step(chi1, chi2, N))
    else:
        raise RuntimeError("Ansatz argument not recognized.  Valid options are \"mps\" or\"ttn\"")

    #set up the wavefunction for simulating the dynamics
    A = ms_ttn(N, topo, dtype=np.complex128)
    state = [[0 for i in range(N)] for j in range(N)]
    coeff = np.zeros(N, dtype=np.float64)
    coeff[int(N//2)] = 1
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
    res = np.zeros((nsteps+1, N), dtype=np.complex128)
    maxchi = np.zeros((nsteps+1))
    
    for j in range(N):
        res[0, j] = mel(A.slice(j))

    t1 = time.time()
    
    #perform a set of steps with a logarithmic timestep discretisation
    A, h, sweep = run_initial_step(A, h, sweep, dt)
    
    for j in range(N):
        res[1, j] = mel(A.slice(j))

    for j in range(N):
        res[1, j] = mel(A.slice(j))
    sweep.dt = dt
    
    timepoints = (np.arange(nsteps+1)*dt)
    
    #perform the remaining dynamics steps
    for i in range(1, nsteps):
        print(i, nsteps, flush=True)
        sweep.step(A, h)
        
        t2 = time.time()
    
        for j in range(N):
            res[i+1, j] = mel(A.slice(j))
    
        if(i % output_skip == 0):
            output_results(ofname, timepoints, res, (t2-t1))
    
    t2 = time.time()
    output_results(ofname, timepoints, res, (t2-t1))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyttn test")

    parser.add_argument('ansatz', type = str, default="mps")

    parser.add_argument('g', type=float)
    parser.add_argument('w0', type=float)
    parser.add_argument('J', type=float)
    parser.add_argument('N', type=int)

    parser.add_argument('chi', type=int)
    parser.add_argument('--chi2', type=int, default=None)
    parser.add_argument('--nbose', type=int, default=8)
    parser.add_argument('--beta', type=float, default=None)

    parser.add_argument('--fname', type=str, default='holstein.h5')

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--tmax', type=float, default=20)

    parser.add_argument('--output_skip', type=int, default=10)
    args = parser.parse_args()

    holstein_dynamics(
            args.ansatz, 
            args.g,
            args.w0, 
            args.J, 
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
