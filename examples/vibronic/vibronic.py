from itertools import product
import numpy as np
import pickle
import os
import sys
import h5py

from pyttn import sOP, SOP, system_modes, nlevel_mode, boson_mode, ntree
from pyttn import ttn, sop_operator, matrix_element, site_operator, tdvp
from pyttn import set_topology_properties, NodeIncrementSetter
from pyttn import ntreeBuilder as ntB

os.environ['OPENBLAS_NUM_THREADS']='1'


def vc_hamiltonian(ls, ws, alps, bs):
    N = alps.shape[0] #number of electronic states
    M = ws.shape[0] #number of vibrational modes    
    rN = range(N)
    rM = range(M)   
    H = SOP(M+1)

    #onsite energies
    for i, j in product(rN, rN):
        ij = "|"+str(i)+"><"+str(j)+"|"
        if np.abs(ls[i, j]) > 1e-12:
            H+=ls[i,j]*sOP(ij,0)    

    #vibrational terms
    for k in rM:
        if np.abs(ws[k]) > 1e-12:
            H+=ws[k]*sOP("n",k+1)   

    #linear vibronic terms
    for i, j, k in product(rN, rN, rM):
        ij = "|"+str(i)+"><"+str(j)+"|"
        if np.abs(alps[i,j,k]) > 1e-12:
            H += alps[i,j,k]*sOP(ij,0)*sOP("q",k+1)   

    #quadratic vibronic terms
    for i, j, k, kp in product(rN, rN, rM, rM):
        ij = "|"+str(i)+"><"+str(j)+"|"
        if np.abs(bs[i,j,k,kp]) > 1e-12:
            H += bs[i,j,k,kp]*sOP(ij,0)*sOP("q",k+1)*sOP("q",kp+1)

    return H

def output_results(ofname, t, res):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=t)
    h5.create_dataset("res", data=res)
    h5.close()

def vibronic_dynamics(ls, ws, alps, bs):
    H=vc_hamiltonian(ls,ws,alps,bs)
    N=alps.shape[0] #number of electronic states
    M=ws.shape[0] #number of vibrational modes
    d=16 #vibrational mode Hilbert space dimension 

    #parameters for tree definition
    dims=[d for _ in range(M)] #hilbert space dims
    Nspf=16   #number of single particle functions
    degree=2  #degree of tree (2=binary)

    #set up system information
    sysinf=system_modes(M+1)
    #electronic mode is a N-level system
    sysinf[0]=nlevel_mode(N) 
    #vibrational modes are bosons
    for k in range(M):
        sysinf[k+1]=boson_mode(d)

    #set up tree containing root and electron dof
    topo=ntree("(1(%d(%d)))"%(N, N))
    capacity=ntree("(1(%d(%d)))"%(N, N))

    #attach a binary tree to the root treating for 
    #handling vibrational modes
    ntB.mlmctdh_subtree(topo(),dims,degree,4)
    ntB.mlmctdh_subtree(capacity(),dims,degree,Nspf)

    #update the capacity tree structure 
    set_topology_properties(capacity, NodeIncrementSetter(4, maxchi=Nspf), sysinf.mode_dimensions(), chi_local_transform=sysinf.mode_dimensions())

    #now construct the wavefunction
    psi=ttn(topo, capacity)
    #and set its current state to |0>_e x |000...>
    state=[0]+[0 for _ in range(M)]
    psi.set_state(state)

    #prepare the hSOP form of the Hamiltonian
    Hop=sop_operator(H,psi,sysinf)

    #prepare the list of operators
    ops=[]
    for i in range(N):
        for j in range(N):
            op=sOP("|"+str(i)+"><"+str(j)+"|", 0)
            ops.append(site_operator(op, sysinf))

    #object for evaluating expectation values
    mel=matrix_element(psi)

    #integration parameters
    dt=0.1      #integration timestep
    nsteps=1000  #number of steps

    #set up the TDVP engine
    engine = tdvp(psi, Hop, krylov_dim=8, subspace_neigs=4, subspace_krylov_dim=8, expansion="subspace")
    engine.spawning_threshold = 1e-5    #energy variance based expansion
    engine.unoccupied_threshold = 1e-5  #unoccupied SPF based expansion
    engine.minimum_unoccupied = 1       #unoccupied SPF based expansion
    engine.dt=dt   #set timestep
    engine.coefficient = -1.0j #set coeff in exp(coeff*H*t)

    #set up an array for storing the results
    res=np.zeros((N*N,nsteps+1),dtype=np.complex128)

    #evaluate the initial expectation values
    for i in range(N*N):
        res[i, 0] = mel(ops[i], psi)

    #now run dynamic and compute expectation values
    for step in range(nsteps):
        engine.step(psi, Hop)
        print(step, (step+1)*dt, end=" ")
        for i in range(N*N):
            res[i, step+1] = mel(ops[i], psi)
            print(res[i, step+1], end=" ")
        print()
        sys.stdout.flush()
        if step % 10 == 0:
            output_results("res.h5", np.arange(nsteps+1)*dt, res)

    output_results("res.h5", np.arange(nsteps+1)*dt, res)


#run vibronic coupling dynamics for the anthracene/C60 complex model presented
#in J. Chem. Phys. 142, 084706 (2015)
filehandler = open('params.pkl', 'rb')
omegas, couplings = pickle.load(filehandler)
filehandler.close()
omegas = np.array(omegas)
lambdas = couplings[0]
alphas = couplings[1]
N = len(lambdas)
M = len(omegas)
betas = np.zeros((N, N, M, M))

vibronic_dynamics(lambdas, omegas, alphas, betas)