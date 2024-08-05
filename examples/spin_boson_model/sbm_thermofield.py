import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *

from sbm_core import sbm_discretise

def spin_boson_test(Nb, alpha, wc, eps, delta, beta, chi, nbose, dt, nstep = 1, degree = 2, yrange=[0, 0.5]):
    g, w = sbm_discretise(Nb, alpha, wc)

    gtherm = np.zeros(2*Nb) 
    wtherm = np.zeros(2*Nb) 
    for i in range(Nb):
        wtherm[Nb-(i+1)] = -w[i]
        wtherm[Nb+i] = w[i]

        gtherm[Nb-(i+1)] = g[i] * 0.5*(1 + 1.0/np.tanh(beta*wtherm[Nb-(i+1)]/2.0))
        gtherm[Nb+i] = g[i] * 0.5*(1 + 1.0/np.tanh(beta*wtherm[Nb+i]/2.0))

    sbg = spin_boson(2*eps, 2*delta, wtherm, gtherm, geom="star")
    sbg.mode_dims = [nbose for i in range(2*Nb)]
    H = sbg.hamiltonian()
    sysinf = sbg.system_info()

    #construct the topology tree 
    topo = ntree("(1(2(2))(2))")
    
    #and add the node that forms the root of the bath
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], sbg.mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], sbg.mode_dims, chi)
    ntreeBuilder.sanitise(topo)
    print(topo)

    A = ttn(topo, dtype=np.complex128)
    A.set_state([0 for i in range(2*Nb+1)])

    h = sop_operator(H, A, sysinf)

    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )

    print(type(op))

    sweep = tdvp(A, h, krylov_dim = 8)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)

    res = np.zeros(nstep+1)

    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=yrange)
    line = ax.plot(np.arange(nstep+1)*dt, res)[0]
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$S_z(t)$")

    res[0] = np.real(mel(op, A, A))
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        res[i+1] = np.real(mel(op, A, A))
        print((i+1)*dt, res[i+1], t2-t1)
        if(plt.fignum_exists(num)):
            plt.gcf().canvas.draw()
            line.set_data(np.arange(nstep+1)*dt, res)
            plt.pause(0.01)
    plt.ioff()
    plt.show()

beta = 1
alphas = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
spin_boson_test(64, alphas[1], 5, 0.0, 1.0, beta, 32, 20, 0.01, degree = 2, nstep = 500, yrange=[-0.5, 0.5])
