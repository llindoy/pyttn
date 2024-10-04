import numpy as np
import scipy

#This will be left as future updates.
#Here we have not made use of any of the knowledge of the form of the A matrix
def ESPRIT(Ct, K):
    #extract the frequencies in the signal using ESPRIT
    expnu = ESPRIT_frequencies(Ct, K)

    #construct the time dependence of each of the frequency signals
    ENU, Kn = np.meshgrid(expnu, np.arange(Ct.shape[0]))
    A = np.power(ENU, Kn)

    #extract the coupling constants by performing a laest squares fit
    weights = np.linalg.lstsq(A, Ct, rcond=None)[0]

    #now return the weights, exponents and fit correlation function
    return weights, -np.log(expnu), A@weights

#Here we have not made use of any of the knowledge of the form of the Y matrix.
#This will seriously limit the number of points in Ct that can be efficiently 
#fit using this approach.
def ESPRIT_frequencies(Ct, K):
    ndata = Ct.shape[0]
    T = (ndata+1)//2
    #Form a hankel matrix from the vector of Cts.  This gives us 
    #T signal points measured on an array of T detectors
    Y = scipy.linalg.hankel(Ct[:T], Ct[T-1:-1])

    #extract the signal subspace from the detector measurements
    U, _, _ = np.linalg.svd(Y)
    Us = U[:, :K]

    #divide into two virtual sub arrays
    S1 = Us[:T-1, :]
    S2 = Us[1:, :]

    #and use these sub arrays to extract the frequencies
    P = np.linalg.lstsq(S1, S2, rcond=None)[0]
    ls, _ = np.linalg.eig(P)
    return ls[np.argsort(np.abs(ls))[::-1]]
