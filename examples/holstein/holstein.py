import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("../../")
from pyttn import *

def setup_interactive_plots(res):
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    num = fig.number
    im = ax.imshow(res, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    return fig, ax, num, im

def update_interactive_plots(num, im,  res):
    if(plt.fignum_exists(num)):
        plt.gcf().canvas.draw()
        im.set_data(res)
        plt.pause(0.1)

def holstein(N, J, gh, wh, chi, nbose, gp=0, wp=0, dt = 0.01, nstep = 100, geom='1d', periodic=True, degree=2):
    include_peirels = (np.abs(gp) > 1e-12 and np.abs(wp) > 0)
    nmodes = N
    if(include_peirels):
        if(periodic):
            nmodes = 2*N
        else:
            nmodes = 2*N-1

    H = multiset_SOP(N, nmodes)

    #first add on the holstein terms
    for i in range(N):
        ind = i
        if(include_peirels):
            ind = 2*i
        H[i, i] += np.sqrt(2)*gh * sOP("q", ind)
        for j in range(N):
            indj = j
            if(include_peirels):
                indj = 2*j
            H[i, i] += wh * sOP("n", indj)
            if(include_peirels):
                H[i, i] += wp * sOP("n", indj+1)

    for i in range(N-1):
        #add on the electron hopping terms
        H[i, i+1] += J
        H[i+1, i] += J
        if(include_peirels):
            ind = 2*i+1
            H[i, i+1] += np.sqrt(2)*gp*sOP("q", ind)
            H[i+1, i] += np.sqrt(2)*gp*sOP("q", ind)

    if periodic:
        H[N-1, 0] += J
        H[0, N-1] += J
        if(include_peirels):
            ind = 2*N-1
            H[N-1, 0] += np.sqrt(2)*gp*sOP("q", ind)
            H[0, N-1] += np.sqrt(2)*gp*sOP("q", ind)

    mode_dims = [nbose for i in range(nmodes)]
    #and add the node that forms the root of the bath
    topo = ntreeBuilder.mlmctdh_tree(mode_dims, degree, chi)
    ntreeBuilder.sanitise(topo)

    A = ms_ttn(topo, N, dtype=np.complex128)
    A.nthreads = 1
    coeffs = np.zeros(N)
    coeffs[N//2] = 1.0
    state = [np.zeros(nmodes, dtype=int) for i in range(N)]
    A.set_state(coeffs, state)

    sysinf = system_modes(nmodes)
    for i in range(nmodes):
        sysinf[i] = boson_mode(nbose)

    h = multiset_sop_operator(H, A, sysinf)

    sweep = tdvp(A, h, krylov_dim = 8, numthreads=1)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)

    As = ttn(topo, dtype=np.complex128)
    print(type(As))
    mel = matrix_element(As)

    print("setup")
    res = np.zeros((N, nstep+1))

    fig, ax, num, im = setup_interactive_plots(res)

    for ind in range(N):
        As.assign(A.slice(ind))
        res[ind, 0] = np.real(mel(As))

    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        print((i+1)*dt, end=' ')
        for ind in range(N):
            As.assign(A.slice(ind))
            res[ind, i+1] = np.real(mel(As))
            print(res[ind, i+1], end=' ')
        print(t2-t1)
        sys.stdout.flush()
        update_interactive_plots(num, im, res)
    plt.ioff()
    plt.show()

holstein(32, 1, 0.1, 1, 24, 24, dt = 0.05, degree=2, nstep = 1000, gp=0.0, wp = 0.0)
