import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *

from sbm_core import sbm_discretise


def spin_boson_test(Nb, alpha, wc, eps, delta, chi, nbose, dt, nstep = 1, degree = 2, yrange=[0, 0.5]):
    g, w = sbm_discretise(Nb, alpha, wc)

    H = SOP(Nb+1)
    H += 2.0*eps*sOP("sz", 0) + 2.0*delta*sOP("sx", 0)
    for i in range(Nb):
        H += np.sqrt(2.0)*g[i]*sOP("sz", 0)*sOP("q", i+1)
        H += w[i]*sOP("n", i+1)

    sysinf = system_modes(Nb+1)
    sysinf[0] = spin_mode(2)
    mode_dims = [nbose for i in range(Nb)]
    for i in range(Nb):
        sysinf[i+1] = boson_mode(mode_dims[i])

    #construct the topology tree 
    topo = ntree("(1(2(2))(2))")
    
    #and add the node that forms the root of the bath
    ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi)
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

spin_boson_test(64, 2.0, 25, 0.0, 1.0, 20, 10, 0.0025, nstep = 400, yrange=[0.4965, 0.5])
