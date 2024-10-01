from engine.heom import *

import numpy as np
import scipy as sp
import h5py

import sys

#current function used to define the support points for the aaa algorithm. The best choice of support points will depend on the nature of the spectral function you are considering (e.g. where are the interesting features of the spectral function, discontinuities, derivative discontinuities, sharp peaks)
#This choice works rather well for exponential cutoffs with challenging points at zero but won't work well for generic spectral densities.

def heom_dynamics(rho0, H, baths, dt, tmax, chimax, tol = 1e-6, plot_aaa = False, fname=None, output_stride =10):
    nhilb = rho0.shape[0]
    nliouv = nhilb*nhilb

    bs = [x.discretise(output_fitting = plot_aaa) for x in baths]

    if plot_aaa:
        import matplotlib.pyplot as plt
        for b in bs:
            plt.plot(b["wf"], b["Sw"])
            plt.plot(b["wf"], b["Sw_fit"])
        plt.show()

    nt = int(tmax/dt)+1

    #now build the propagator matrices for HEOM
    print("Build HEOM propagator matrices", file=sys.stderr)
    prop = HEOM_propagator(bs, dt)
     
    A = setup_HEOM_ados(rho0, prop)

    print("Build HEOM propagator MPO", file=sys.stderr)

    rhos = np.zeros((nt+1, nliouv), dtype = np.complex128)

    rhos[0, :] = rho0.flatten()

    print(0, end=' ' )
    for j in range(rhos.shape[1]):
        print(np.real(rhos[0, j]), np.imag(rhos[0, j]), end=' ')
    print(A.maximum_bond_dimension())

    n = 1
    for i in range(nt):
        #update the time dependent system propagator at the midpoint of the next step
        Hsys = H(i*dt+dt/2)
        Lsys = commutator(Hsys)
        Us = sp.linalg.expm(-1.0j*dt*Lsys)    

        #apply the propagator through a time step
        apply_propagator(Us, prop, A, tol=tol, nbond=chimax)

        #compute rho from the object
        rho = extract_rho(A)

        #and save it in the rhos array
        rhos[i+1, :] = rho
        print((i+1)*dt, end=' ' )
        for j in range(len(rho)):
            print(np.real(rho[j]), end=' ')
        print(np.real(rho[0]+rho[3]), end=' ')
        print(A.maximum_bond_dimension())
        sys.stdout.flush()

        if (i % output_stride == 0 and not fname == None):
            h5out = h5py.File(fname, 'w')
            h5out.create_dataset('t', data=np.arange(nt+1)*dt)
            h5out.create_dataset('rho', data=rhos)
            h5out.close()

    if not fname == None:
        h5out = h5py.File(fname, 'w')
        h5out.create_dataset('t', data=np.arange(nt+1)*dt)
        h5out.create_dataset('rho', data=rhos)
        h5out.close()
    return rhos


def main():
    sx = np.array([[0, 1], [1, 0]], dtype = np.complex128)
    sy = np.array([[0, -1.0j], [1.0j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)

    #and the system part of the system bath coupling operator
    S = sz

    #function defining the bath spectral function
    beta = 5
    s = 1
    zeta = 0.5
    wc = 7.5
    def Jw(w):
        return np.sign(w)*np.pi/2.0*zeta*wc*np.power(np.abs(w)/wc, s)*np.exp(-np.abs(w)/wc)

    def Sw(w):
        if beta == None:
            return Jw(w)*np.where(w > 0, 1.0, 0.0)
        else:
            return Jw(w)*0.5*(1+1.0/np.tanh(beta*w/2.0))


    #the initial value of the system density operator
    rho0 = (np.identity(2) + sz)/2

    #set up the evolution parameters
    dt = 0.05
    tmax = 15.0

    #setup the ados parameters - accuracy of representation of auxiliary density operator
    nbose = 15
    Lmin = 5
    tol = 1e-8
    chimax = 100

    #setup the aaa parameters - controls accuracy of representation of bath correlation function 
    #convergence with respect to the aaa_tol parameter depends significantly on the choice of support points used, and also on the form of the bath correlation function.  
    #Here I am using a softmspace grid with a rather tight zero.  This approach seems to work quite well for baths with an exponential cutoff at zero temperature but doesn't necessarily work the best for e.g. Debye baths.
    #The choice of support points is somewhat an art and the accuracy can really only be determined by monitoring both the accuracy of the aaa fitting of the spectral function and also the correlation function.  
    #To change the 
    aaa_tol = 1e-3
    wmax = wc

    #construct the HEOM baths object
    baths = [heom_bath(S, Sw, nbose, Lmin=Lmin, wmax = wc, aaa_tol=aaa_tol)]

    eps = 0.0
    delta = 1.0

    def H(t):
        return eps*sz + delta*sx
    

    rhos = heom_dynamics(rho0, H, baths, dt, tmax, chimax, tol=tol, plot_aaa=True, fname='sbm_dynamics.h5')

if __name__ == "__main__":
    main()

