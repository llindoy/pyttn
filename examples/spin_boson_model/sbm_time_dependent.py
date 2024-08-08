import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *
from pyttn.utils import density_discretisation, orthopol_discretisation

from sbm_core import sbm_discretise

def spin_boson_test(Nb, alpha, wc, eps, delta, chi, nbose, dt, nstep = 1, degree = 2, yrange=[0, 0.5]):
    g, w = density_discretisation.discretise( lambda w : np.pi*alpha/2*w*np.exp(-w/wc), lambda w : np.exp(-w/wc), 0.0, wc*5, Nb)
    #g, w = orthopol_discretisation.discretise( lambda w : np.pi*alpha/2.0*w*np.exp(-w/wc), 0.0, wc*5, Nb, moment_scaling=1.95, atol=0, rtol=1e-10)
    w = np.array(w, dtype=np.complex128)
    g = np.array(g, dtype=np.complex128)
    print(w)
    print(g)
    plt.plot(w, g)
    plt.show()


    sbg = models.spin_boson(eps, delta, w, g, geom="star")
    sbg.mode_dims = [nbose for i in range(Nb)]
    sysinf = sbg.system_info()

    H = SOP(Nb+1)
    H += 2.0*delta*sOP("sx", 0)#coeff(lambda t : 0.0 if np.mod(t, 0.2) < 0.1 else delta)*sOP("sx", 0)
    for i in range(Nb):
        H += 2*np.sqrt(2.0)*g[i]*sOP("sz", 0)*sOP("q", i+1)
        H += w[i]*sOP("n", i+1)

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
    sweep.coefficient = -1.0j
    sweep.use_time_dependent_hamiltonian = True

    res = np.zeros(nstep+1)

    plt.ion()
    fig, ax = plt.subplots()
    num = fig.number
    ax.set(xlim=[0, dt*nstep])
    ax.set(ylim=yrange)
    line = ax.plot(np.arange(nstep+1)*dt, res*2)[0]
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
            line.set_data(np.arange(nstep+1)*dt, res*2)
            plt.pause(0.01)

    plt.ioff()
    plt.show()

alphas = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
spin_boson_test(64, 2.0, 25, 0.0, 1.0, 8, 10, 0.0025, degree = 2, nstep = 200, yrange=[0.993, 1])
