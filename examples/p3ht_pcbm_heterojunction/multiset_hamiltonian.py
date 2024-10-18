import numpy as np
import sys

sys.path.append("../../")
from pyttn import *

#convert from eV to hartree
eV = 0.0367493049512081
meV = eV/1000


def ot_mode_index(Nf, n, l):
    if l < 6:
        return n*6 + l
    else:
        return Nf*6 + 2*n + l-6

def hamiltonian():
    Nfragments=13
    Nf = 8
    Not = 8

    eps_le = 100*meV
    eps_cs = np.array([0.0, 33.6, 47.4, 56.0, 61.8, 65.7, 68.4, 70.0, 70.9, 71.2, 71.1, 70.5, 69.5])*meV

    J = 100*meV
    t=-120*meV
    L = -200*meV

    wr = 10*meV
    wf = (np.array([200.025, 184.269, 177.853, 141.11, 93.952, 79.933, 55.892, 33.264])*meV)[::-1]
    wot = np.array([401.283, 397.773, 182.714, 178.531, 134.550, 111.848, 42.621, 18.316])*meV

    gr_cs1cs1 = 30/np.sqrt(2.0)*meV
    gr_lecs1 = -10/np.sqrt(2.0)*meV

    gf = (np.array([45.246, 65.701, -40.280, -17.511, 28.026, -13.629, -23.732, 9.86])*meV)[::-1]
    g_otcs = np.array([7.017, -0.077, -67.849, 57.668, -40.145, 11.68, -10.784, -12.309])*meV
    g_otle = np.array([4.035, 2.921, -129.712, 46.885, -32.908, 36.591, -20.211, -7.77])*meV

    N = Nfragments*Not + Nf + 1 

    #now build the Hamiltonian
    H = multiset_SOP(26, N)

    #add on the vibrational mode terms
    for msind in range(26):
        H[msind, msind] += wr * sOP("n", 0)
        for i in range(Nf):
            H[msind, msind] += wf[i] * sOP("n", 1 + i)

        for n in range(Nfragments):
            for l in range(Not):
                H[msind, msind] += wot[l] * sOP("n", 1 + Nf + ot_mode_index(Nfragments, n, l))

    #add on the purely electronic terms
    #the onsite energies
    for i in range(Nfragments):
        H[2*i, 2*i] += eps_le
        H[2*i+1, 2*i+1] += eps_cs[i]

    #now add on the hopping terms
    for i in range(Nfragments-1):
        H[2*i, 2*(i+1)] += J
        H[2*(i+1), 2*i] += J

        H[2*i+1, 2*(i+1)+1] += t
        H[2*(i+1)+1, 2*i+1] += t

    H[0, 1] += L
    H[1, 0] += L

    #now add on the vibronic couplings
    #first we do the vibronic couplings to the R mode
    H[1, 1] += gr_cs1cs1*sOP("adag", 0)
    H[1, 1] += gr_cs1cs1*sOP("a", 0)

    H[0, 1] += gr_lecs1*sOP("adag", 0)
    H[0, 1] += gr_lecs1*sOP("a", 0)

    H[1, 0] += gr_lecs1*sOP("adag", 0)
    H[1, 0] += gr_lecs1*sOP("a", 0)

    #now we add on the vibronic couplings to the fullerene super-particle modes
    for n in range(Nfragments):
        for l in range(Nf):
            H[2*n+1, 2*n+1] += gf[l]*sOP("adag", 1+l)
            H[2*n+1, 2*n+1] += gf[l]*sOP("a", 1+l)

    #now we add on the oligothiophene modes
    for n in range(Nfragments):
        for l in range(Not):
            print(n, l, ot_mode_index(Nfragments, n, l))
            H[2*n+1, 2*n+1] += g_otcs[l] * sOP("adag", 1+Nf+ ot_mode_index(Nfragments, n, l))
            H[2*n+1, 2*n+1] += g_otcs[l] * sOP("a", 1+Nf+ ot_mode_index(Nfragments, n, l))
            H[2*n, 2*n] += g_otle[l] * sOP("adag", 1+Nf+ ot_mode_index(Nfragments, n, l))
            H[2*n, 2*n] += g_otle[l] * sOP("a", 1+Nf+ ot_mode_index(Nfragments, n, l))

    return H



