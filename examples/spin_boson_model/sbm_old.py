import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *

from sbm_core import discretise_bath

def Ct(t, w, g):
    T, W = np.meshgrid(t, w)
    g2 = g**2
    fourier = np.exp(-1.0j*W*T)
    return g2@fourier

def spin_boson_test(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta=None, nstep = 1, degree = 2, yrange=[0, 0.5], Nw =12):
    g, w, renorm = discretise_bath(Nb, alpha, wc, s, beta=beta, Nw=Nw)
    print(renorm)

    sbg = models.spin_boson(eps, delta*renorm, w, 2*g, geom="star")
    sbg.mode_dims = [nbose for i in range(Nb)]
    H = sbg.hamiltonian()
    sysinf = sbg.system_info()

    N = Nb+1
    mode_dims = [2 for i in range(Nb)]
    #and add the node that forms the root of the bath
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

    sweep = tdvp(A, h, krylov_dim = 12)
    sweep.dt = dt
    sweep.coefficient = -1.0j
    sweep.prepare_environment(A, h)

    res = np.zeros(nstep+1)

    res[0] = np.real(mel(op, A, A))
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        res[i+1] = np.real(mel(op, A, A))
        print((i+1)*dt, t2-t1, res[i+1])
        sys.stdout.flush()

spin_boson_test(32, 0.1, 5, 1.0, 0.0, 1.0, 16, 20, 0.00125, degree = 2, nstep = 6000, yrange=[-0.5, 0.5])
