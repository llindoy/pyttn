import numpy as np
import scipy as sp


def bosonic_bath_properties(
    dk, zk, L, Lmin=None, combine_real=False, tol=1e-12, nbmax=1, nhilbmax=1000
):
    ck = []
    nuk = []
    ds = []
    real_mode = []

    mode_dims = []
    mode_inds = []

    minzk = np.amin(np.real(zk))

    # first iterate through each of the nodes and determine if any of them are purely real.
    # in the case of a purely real mode we will only append a single term
    counter = 0
    for i in range(len(dk)):
        if Lmin == None:
            nb = L
        else:
            nb = max(int(L * minzk / np.real(zk[i])), Lmin)

        # if we are combining two real frequency modes into a single mode accounting for
        # both forward and backward paths
        if combine_real and np.imag(zk[i]) < tol:
            ds.append(nb)  # set the mode dimension
            nuk.append(zk[i])  # set the mode frequency
            ck.append(ck[i])  # set the mode coupling constant
            real_mode.append(True)  # flag that this is a real valued mode

            # set up the information that will be used for additional mode combination.
            mode_dims.append(nb)
            mode_inds.append([counter])
            counter = counter + 1

        # otherwise we add on separate modes for forward and backward paths
    else:
        # handle the forward path object
        ds.append(nb)
        nuk.append(zk[i])
        ck.append(np.sqrt(dk[i]))
        real_mode.append(False)

        # handle the backward path object
        ds.append(nb)
        nuk.append(np.conj(zk[i]))
        ck.append(np.sqrt(np.conj(dk[i])))
        real_mode.append(False)

        # now set up the information that will be used for attempting additional mode combination
        mode_dims.append(nb * nb)
        mode_inds.append([counter, counter + 1])
        counter = counter + 2

    # now that we have the preliminary information we can attempt to perform mode combination on this object.
