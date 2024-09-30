import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs

from pyttn._pyttn import liouville_space
from pyttn._pyttn import operator_dictionary_complex


from numba import jit


#setup the star Hamiltonian for the spin boson model
def star_hamiltonian(eps, delta, g, w, Nb):
    N = Nb+1

    H = SOP(N)
    H += eps*sOP("sy", 0)
    H += delta*sOP("sx", 0)
    for i in range(Nb):
        H += np.sqrt(2)*g[i] * sOP("sz", 0)  * sOP("q", i+1)
        H += w[i] * sOP("n", i+1)

    return H, w


def combine_modes(bath_mode_dims, bath_mode_inds, nbmax, nhilbmax):
    composite_modes = []

    all_modes_traversed = False
    cmode = []
    chilb = 1
    mode = 0
    while not all_modes_traversed:
        #if the current cmode object is empty then we just add the current mode to the composite mode and increment
        if(len(cmode) == 0):
            cmode.append(bath_mode_inds[mode])
            chilb = chilb*bath_mode_dims[mode]
            mode += 1
            if(mode == len(bath_mode_dims)):
                all_modes_traversed = True
                composite_modes.append(copy.deepcopy(cmode))
        else:
            #othewise we check to see if the composite mode could accept the current mode without exceeding the bounds
            #then we add and increment
            if len(cmode) < nbmax and chilb*bath_mode_dims[mode] <= nhilbmax:
                cmode.append(bath_mode_inds[mode])
                chilb = chilb*bath_mode_dims[mode]
                mode += 1
                if(mode == len(bath_mode_dims)):
                    all_modes_traversed = True
                    composite_modes.append(copy.deepcopy(cmode))

            else:
                #otherwise we have reached the end of the current composite mode.  We will now reset the composite 
                #mode object and we will not increment the mode so that it start a new composite mode object in the
                #next iteration
                composite_modes.append(copy.deepcopy(cmode))
                cmode = []
                chilb = 1

    return composite_modes


def default_op_test():
    Nb=4
    g = np.sqrt(np.arange(Nb)+1)
    w = np.arange(Nb)+1

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(N)

    eps = 0.1
    delta = 1

    #now add on the bath bits getting additionally getting a frequency parameter that can be used in energy based
    #truncation schemes
    H, w = star_hamiltonian(eps, delta, 2*g, w, Nb)

    print(H)

    #mode_dims = [nbose for i in range(Nb)]
    mode_dims = [10 for i in range(Nb)]

    nbmax = 1
    nhilbmax = 100
    #attempt to combine modes together based on am
    composite_modes = combine_modes(mode_dims, [x+1 for x in range(Nb)], nbmax, nhilbmax)
    Nbc = len(composite_modes)

    #setup the system information object
    sysinf = system_modes(1+len(composite_modes))
    sysinf[0] = spin_mode(2)

    for ind, cmode in enumerate(composite_modes):
        sysinf[ind+1] = [boson_mode(mode_dims[x-1]) for x in cmode]
        lhd = np.prod(np.array([mode_dims[x-1] for x in cmode]))

    Hl = SOP(N)
    #liouville_space.left_superoperator(H, sysinf, Hl, 1.0)
    liouville_space.anticommutator_superoperator(H, sysinf, Hl, 1.0)
    print(Hl)


def operator_dictionary_op_test():
    Nb=4
    g = np.sqrt(np.arange(Nb)+1)
    w = np.arange(Nb)+1

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(N)

    eps = 0.1
    delta = 1

    #now add on the bath bits getting additionally getting a frequency parameter that can be used in energy based
    #truncation schemes
    H, w = star_hamiltonian(eps, delta, 2*g, w, Nb)

    print(H)

    #mode_dims = [nbose for i in range(Nb)]
    mode_dims = [3 for i in range(Nb)]

    #setup the system information object
    sysinf = system_modes(1+Nb)
    sysinf[0] = generic_mode(2)

    for i in range(Nb):
        sysinf[i+1] = generic_mode(mode_dims[i])

    print(H)
    #set up the operator dictionary information
    opdict = operator_dictionary_complex(N)
    #set up the different system operators
    opdict.insert(0, "sx", site_operator(np.random.rand(2,2), optype="matrix", mode=0))
    opdict.insert(0, "sy", site_operator(np.random.rand(2), optype="diagonal_matrix", mode=0))
    opdict.insert(0, "sz", site_operator(scipy.sparse.random(2, 2, density=0.5, format='csr'), optype="sparse_matrix", mode=0))

    for i in range(Nb):
        opdict.insert(i+1, "q", site_operator(scipy.sparse.random(mode_dims[i], mode_dims[i], density=0.1, format='csr'), optype="sparse_matrix", mode=i+1))
        opdict.insert(i+1, "n", site_operator(scipy.sparse.random(mode_dims[i], mode_dims[i], density=0.1, format='csr'), optype="sparse_matrix", mode=i+1))

    for i in range(opdict.nmodes()):
        print("mode: ", i)
        for k,v in opdict[i].items():
            print(k, v)


    Hl = SOP(N)
    #liouville_space.left_superoperator(H, sysinf, Hl, 1.0)
    opdict2 = operator_dictionary_complex(N)
    liouville_space.anticommutator_superoperator(H, sysinf, opdict, Hl, opdict2, 1.0)
    print(Hl)

    for i in range(opdict2.nmodes()):
        print("mode: ", i)
        for k,v in opdict2[i].items():
            print(k, v)
    exit()

if __name__ == "__main__":
    operator_dictionary_op_test()
