import numpy as np
import copy

from pyttn import sOP, coeff

def __generate_binds(binds, bskip, Nb):
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i+bskip for i in range(Nb)]
    return binds

def add_heom_bath_generator(H, Sp, dks, zks, Sm=None, binds = None, bskip=2):
    Nb = 0
    for dk in dks:
        Nb = Nb + len(dk)
        if not (len(dk)==1 or len(dk)==2) :
            raise Exception("Cannot add HEOM  bath unless each unexpected mode size")
    binds =  __generate_binds(binds, bskip, Nb)

    #set up the system bath operator
    c = 0
    for dk, zk in zip(dks, zks):
        #if we are dealing with a single terms.  This corresponds to the case that we have a single real value frequency
        if len(dk) == 1:
            #add on the bath terms
            H += -1.0j*zk[0]*sOP("n", binds[c])

            #add on the bath annihilation terms

            #add on the bath creation terms

            c = c+1
        #otherwise we need to add on modes corresponding to forward and backward paths
        elif len(dk) == 2:
            #add on the bath terms
            H += -1.0j*zk[0]*sOP("n", binds[c])
            H += -1.0j*zk[1]*sOP("n", binds[c+1])

            #add on the bath annihilation terms
            H += complex(dk[0])*(Sp[0] - Sp[1])*sOP("a", binds[c])
            H += complex(dk[1])*(Sp[0] - Sp[1])*sOP("a", binds[c+1])
                
            #add on the bath creation terms
            #if the Sm operator is correctly defined use it
            if isinstance(Sm, list) and len(Sm) == 2:
                H +=  dk[0]*Sm[0]*sOP("adag", binds[c])
                H += -dk[1]*Sm[1]*sOP("adag", binds[c+1])

            #otherwise just use the Sp operators
            else:
                H +=  dk[0]*Sp[0]*sOP("adag", binds[c])
                H += -dk[1]*Sp[1]*sOP("adag", binds[c+1])
            c = c+2

    return H

def add_pseudomode_bath_generator(H, Sp, dks, zks, Sm=None, binds = None, bskip=2):
    Nb = 0
    for dk in dks:
        Nb = Nb + len(dk)
        if not (len(dk)==2):
            raise Exception("Cannot add pseudomode bath unless each mode corresponds to forward and backward paths")
        
    binds =  __generate_binds(binds, bskip, Nb)

    c = 0
    for dk, zk in zip(dks, zks):
        gk = np.real(zk[0])
        Ek = np.imag(zk[0])
        Mk = -np.imag(dk[0])

        i1 = binds[c]
        i2 = binds[c+1]

        #add on the bath only terms
        H += complex(Ek)*(sOP("n", i1)-sOP("n", i2))                                         #the energy terms
        H += 2.0j*complex(gk)*(sOP("a", i1)*sOP("a", i2)-0.5*(sOP("n", i1)+sOP("n", i2)))    #the dissipators

        #now add on the system bath coupling terms
        #if the Sm operator is correctly defined use it
        if isinstance(Sm, list) and len(Sm) == 2:
            H += 2.0j*complex(Mk)*(Sp[1]*sOP("a", i1)) 
            H += 2.0j*complex(np.conj(Mk))*(Sp[0]*sOP("a", i2))

            H += complex(dk[0])*Sm[0]*sOP("adag", i1) - complex(dk[1])*Sm[1]*sOP("adag", i2)                                  
            H += complex(dk[0])*Sp[0]*sOP("a", i1) - complex(dk[1])*Sp[1]*sOP("a", i2)
            
        #otherwise just use the Sp operators
        else:
            H += 2.0j*complex(Mk)*(Sp[1]*sOP("a", i1)) 
            H += 2.0j*complex(np.conj(Mk))*(Sp[0]*sOP("a", i2))

            H += complex(dk[0])*Sp[0]*sOP("adag", i1) - complex(dk[1])*Sp[1]*sOP("adag", i2)                                  
            H += complex(dk[0])*Sp[0]*sOP("a", i1) - complex(dk[1])*Sp[1]*sOP("a", i2)

        c = c+2

    return H

def add_bosonic_bath_generator(H, Sp, dks, zks, Sm=None, binds=None, bskip=2, method="heom"):
    if not isinstance(Sp, list):
        raise RuntimeError("Invalid Sp operator for heom.add_bosonic_bath_generator")
    if method == "heom":
        return add_heom_bath_generator(H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip)
    elif method == "pseudomode":
        return add_pseudomode_bath_generator(H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip)
    else:
        raise RuntimeError("Pseudomode based bath method not recognised.")
