import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import scipy
sys.path.append("../../")
from pyttn import *
from pyttn.utils import density_discretisation, orthopol_discretisation

from numba import jit

def discretise_bath(Nb, alpha, wc, s, beta = None, Nw = 10, moment_scaling=2, atol=0, rtol=1e-10):
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    @jit(nopython=True)
    def S(w):
        if beta == None:
            return J(w)*np.where(w > 0, 1.0, 0.0)
        else:
            return J(w)*0.5*(1.0+1.0/np.tanh(beta*w/2.0))

    wmax = Nw*wc
    wmin = 0
    if beta != None:
        wmin = -wmax/(beta*wc+1)
    g, w = orthopol_discretisation.discretise( lambda x : S(x), wmin, wmax, Nb, moment_scaling=moment_scaling, atol=atol, rtol=rtol)

    renorm = np.exp(-2.0/np.pi*scipy.integrate.quad(lambda x : J(x)/(x*x), wmax, np.inf)[0])

    return np.array(g), np.array(w), renorm

def spin_boson_test(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta=None, nstep = 1, degree = 2, compress = True, adaptive=False):
    g, w, renorm = discretise_bath(Nb, alpha, wc, s, beta=beta)

    sbg = models.spin_boson(2*eps, 2*delta*renorm, w, g, geom="star")
    mode_dims = [nbose for i in range(Nb)]
    sbg.mode_dims = mode_dims
    H = sbg.hamiltonian()
    sysinf = sbg.system_info()

    #construct the topology tree 
    topo = ntree("(1(2(2))(2))")
    
    #and add the node that forms the root of the bath
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi, nbose)
    ntreeBuilder.sanitise(topo)

    #construct the topology tree 
    capacity = ntree("(1(2(2))(2))")
    
    #and add the node that forms the root of the bath
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(capacity()[1], mode_dims, degree, chi+8)
    else:
        ntreeBuilder.mps_subtree(capacity()[1], mode_dims, chi+8, min(chi+8, nbose))
    ntreeBuilder.sanitise(capacity)

    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(Nb+1)])

    h = sop_operator(H, A, sysinf, compress=compress)

    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )

    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 12)
    else:
        sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace')
        sweep.spawning_threshold = 1e-5
        sweep.unoccupied_threshold=1e-4
        sweep.minimum_unoccupied=1
        sweep.eval_but_dont_apply=True

    sweep.dt = dt
    sweep.coefficient = -1.0j
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
