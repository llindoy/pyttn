import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

sys.path.append("../../")
from pyttn import *


def hyperfines(N):
    return np.sqrt(6.0*N/(2.0*N*N+3*N+1)) * (N-np.arange(N))/N

def hamiltonian(B, A):
    Nb = len(A)
    N = Nb+1
    #set up the Hamiltonian
    H = SOP(N)

    H += B*sOP("sz", 0)

    for i in range(Nb):
        H += A[i] * (sOP("sx", 0) * sOP("sx", i+1) )
        H += A[i] * (sOP("sy", 0) * sOP("sy", i+1) )
        H += A[i] * (sOP("sz", 0) * sOP("sz", i+1) )

    return H

def csm_test(Nb, B, chi, dt, nstep = 1, degree = 2, yrange=[-0.25, 0.25]):
    A = hyperfines(Nb)

    N = Nb+1

    H = hamiltonian(B, A)
    print(H)
    bath_dims = [2 for i in range(Nb)]

    #set up the system information
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = spin_mode(2)

    #construct the tree topology used for these calculations.  Here we are using a binary tree to partition the up and down spin sectors
    #with the impurity orbitals taking the first site on each subtree
    topo = ntree("(1(2(2))(2))")
    
    #then we are adding on trees for the bath degrees of freedom
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], bath_dims, degree, chi)
    else:
        raise RuntimeError("mps subtree currently doesn't work correctly.")
    print(topo)
    ntreeBuilder.sanitise(topo)
    print(topo)

    A = ttn(topo, dtype=np.complex128, purification=True)
    A.set_identity_purification()
    B = ttn(topo, dtype=np.complex128, purification=True)
    B.set_identity_purification()

    sqrsz_op = site_operator(1.0/np.sqrt(2.0)*np.array([1, 1, 1.0j, 1.0j]), optype="diagonal_matrix", mode=0)
    sqrszb_op = site_operator(1.0/np.sqrt(2.0)*np.array([1, 1, -1.0j, -1.0j]), optype="diagonal_matrix", mode=0)\

    sx = np.array([ [0, 0.5], [0.5, 0]])
    sx_op = site_operator(np.kron(sx, np.identity(2)), optype="matrix", mode=0)
    sz_op = site_operator([0.5, 0.5, -0.5, -0.5], optype="diagonal_matrix", mode=0)


    A.apply_one_body_operator(sqrsz_op)
    B.apply_one_body_operator(sqrszb_op)
    A.orthogonalise()
    B.orthogonalise()

    h = sop_operator(H, A, sysinf)
    hB = sop_operator(H, B, sysinf)

    mel = matrix_element(A)

    #set up the tdvp engine
    sweep = tdvp(A, h, krylov_dim = 8)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)
    sweepB = tdvp(B, hB, krylov_dim = 8)
    sweepB.dt = dt
    sweepB.coefficient = 1.0j
    sweepB.prepare_environment(B, hB)

    res = np.zeros(nstep+1)
    resx = np.zeros(nstep+1)

    #set up interactive plotting
    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    l1 = ax.plot(np.arange(nstep+1)*dt, res)[0]
    l2 = ax.plot(np.arange(nstep+1)*dt, resx)[0]
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=yrange)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$<S_z(t) S_z(0)>$")

    print(mel(A, A))
    res[0] = np.real(mel(sz_op, A, B))
    resx[0] = np.real(mel(sx_op, A, B))
    #do the time evolution
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        sweepB.step(B, hB)
        t2 = time.time()
        res[i+1] = np.real(mel(sz_op, A, B))
        resx[i+1] = np.real(mel(sx_op, A, B))
        print((i+1)*dt, res[i+1], resx[i+1], t2-t1)
        if(plt.fignum_exists(num)):
            plt.gcf().canvas.draw()
            l1.set_data(np.arange(nstep+1)*dt, res)
            l2.set_data(np.arange(nstep+1)*dt, resx)
            plt.pause(0.01)

    plt.ioff()
    plt.show()

csm_test(49, 0.5, 32, 0.05, nstep = 200)

