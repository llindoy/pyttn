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


def spin_boson_test(Nb, alpha, wc, eps, delta, chi, nbose, dt, nstep = 1, degree = 2, compress = True):
    g, w = sbm_discretise(Nb, alpha, wc)

    sbg = spin_boson(eps, delta, w, g, geom="star")
    sbg.mode_dims = [nbose for i in range(Nb)]
    H = sbg.hamiltonian()
    sysinf = sbg.system_info()

    #construct the topology tree 
    topo = ntree("(1(2(2))(2))")
    
    #and add the node that forms the root of the bath
    ntreeBuilder.mlmctdh_subtree(topo()[1], sbg.mode_dims, degree, chi)
    ntreeBuilder.sanitise(topo)

    A = ttn(topo, dtype=np.complex128)
    A.set_state([0 for i in range(Nb+1)])

    h = sop_operator(H, A, sysinf, compress=compress)

    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )

    sweep = tdvp(A, h, krylov_dim = 12)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)

    timings = np.zeros(nstep)
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        timings[i] = t2-t1

    stdev = 0
    if(nstep > 1):
        stdev = np.std(timings)
    return np.mean(timings), stdev
