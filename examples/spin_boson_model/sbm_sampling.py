import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *

from sbm_core import sbm_discretise

def spin_boson_test(Nb, alpha, wc, eps, delta, beta, chi, dt, nstep = 1, degree = 2, yrange=[0, 0.5], mode_trunc_tol=1e-5, nbmin=10, nbmax = None, nsamples = 100):
    g, w = sbm_discretise(Nb, alpha, wc)
    print(w)

    nbose = None
    if not nbmax == None:
        nbose = [min(nbmax, max(int(np.ceil(-(np.log(mode_trunc_tol)/(beta*wi) + 1))), nbmin)) for wi in w]
    else:
        nbose = [max(int(np.ceil(-(np.log(mode_trunc_tol)/(beta*wi) + 1))), nbmin) for wi in w]
    print(nbose)

    
    dist = [np.exp(-beta*wi*np.arange(nb))/np.sum(np.exp(-beta*wi*np.arange(nb))) for nb, wi in zip(nbose, w)]
    dist.insert(0, np.array([1, 0]))

    sbg = spin_boson(2*eps, 2*delta, w, g, geom="star")

    sbg.mode_dims = nbose
    H = sbg.hamiltonian()
    sysinf = sbg.system_info()

    #construct the topology tree 
    topo = ntree("(1(2(2))(2))")
    
    #and add the node that forms the root of the bath
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], sbg.mode_dims, degree, chi)
    else:
        raise RuntimeError("mps subtree currently doesn't work")

    ntreeBuilder.sanitise(topo)
    print(topo)
    A = ttn(topo, dtype=np.complex128)
    h = sop_operator(H, A, sysinf)

    res = np.zeros((nsamples, nstep+1))
    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=yrange)
    lav = ax.plot(np.arange(nstep+1)*dt, res[0,:], 'k')[0]
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$S_z(t)$")

    for sample in range(nsamples):
        A.sample_product_state(dist)

        mel = matrix_element(A)

        op = site_operator_complex(
            sOP("sz", 0),
            sysinf
        )

        sweep = tdvp(A, h, krylov_dim = 8)
        sweep.dt = dt
        sweep.coefficient = 1.0j
        sweep.prepare_environment(A, h)

        line = ax.plot(np.arange(nstep+1)*dt, res[sample,:], 'grey')[0]


        res[sample, 0] = np.real(mel(op, A, A))
        for i in range(nstep):
            t1 = time.time()
            sweep.step(A, h, dt)
            t2 = time.time()
            res[sample, i+1] = np.real(mel(op, A, A))
            print((i+1)*dt, res[sample, i+1], t2-t1)
            if(plt.fignum_exists(num)):
                plt.gcf().canvas.draw()
                line.set_data(np.arange(nstep+1)*dt, res[sample, :])
                lav.set_data(np.arange(nstep+1)*dt, np.mean(res[:(sample+1), :], axis=0))
                plt.pause(0.01)
    plt.ioff()
    plt.show()

alphas = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
spin_boson_test(64, alphas[1], 5, 0.0, 1.0, 1, 10, 0.01, degree = 2, nstep = 500, yrange=[-0.5, 0.5], nbmax=200, nbmin=10, mode_trunc_tol=1e-5)
