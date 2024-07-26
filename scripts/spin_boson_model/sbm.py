import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *


def sbm_discretise(N, alpha, wc):
    w = -wc * np.log(1.-np.arange(N)/(N+1.0))
    g = np.sqrt(4*alpha*w)
    return g, w


def spin_boson_test(Nb, alpha, wc, eps, delta, chi, nbose, dt, nstep = 1, degree = 2, yrange=[0, 0.5]):
    g, w = sbm_discretise(Nb, alpha, wc)


    sbg = spin_boson(eps, delta, w, g, geom="star")
    sbg.mode_dims = [nbose for i in range(Nb)]
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
    A.set_state([0 for i in range(Nb+1)])

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
            line.remove()
            del line
            line = ax.plot(np.arange(nstep+1)*dt, res)[0]
            plt.pause(0.1)


spin_boson_test(8, 4.0, 25, 0.0, 2.0, 20, 10, 0.001, degree = 1, nstep = 100, yrange=[0.4996, 0.5])
