import numpy as np
from pymps.tensornetwork.SOP import *
from itertools import product

from openfermion.ops import FermionOperator
from openfermion import *

def inds(i,s,N) :
    if s == -1 :
        return N-1-i
    else :
        return N+i

def Hed(U, eimp, density_int=False, use_open_fermion = False):
    """
    contruct H from parameters read in param_3
    """
    norb = eimp.shape[0]
    N = norb
    nimp = U.shape[0]

    if not use_open_fermion:
        h0 = SOP()
        hi = SOP()

        spins = [-1,1]
        for i, j,k,l in product(range(nimp), range(nimp),range(nimp), range(nimp)):
            u = U[i,j,k,l]
            if density_int :
                for s1,s2 in product(spins,spins) :
                    if s1!= s2 :
                        if (i!=k) or (j!=l) :
                            continue
                    hi += u/2.0*fOp("cdag", inds(i,s1,N))*fOp("cdag", inds(j,s2,N))*fOp("c", inds(l,s2,N))*fOp("c", inds(k,s1, N))

            else :
                for s1,s2 in product(spins,spins) :
                    hi += u/2.0*fOp("cdag", inds(i,s1,N))*fOp("cdag", inds(j,s2,N))*fOp("c", inds(l,s2,N))*fOp("c", inds(k,s1, N))

        for i, j in product(range(norb), range(norb)):
            for s1 in spins :
                h0 += eimp[i,j]*fOp("cdag", inds(i, s1, N))*fOp("c", inds(j, s1, N))
        return h0, hi, 2*N
    else:
        hi, h0 = 0, 0
        spins = [-1,1]
        for i, j,k,l in product(range(nimp), range(nimp),range(nimp), range(nimp)):
            u = U[i,j,k,l]
            if density_int :
                for s1,s2 in product(spins,spins) :
                    if s1!= s2 :
                        if (i!=k) or (j!=l) :
                            continue
                    hi +=FermionOperator(f"{inds(i,s1,N)}^  {inds(j,s2,N)}^  {inds(l,s2,N)} {inds(k,s1,N)} ", u/2. )

            else :
                for s1,s2 in product(spins,spins) :
                    hi +=FermionOperator(f"{inds(i,s1,N)}^  {inds(j,s2,N)}^  {inds(l,s2,N)} {inds(k,s1,N)} ", u/2. )
 

        for i, j in product(range(norb), range(norb)):
            for s1 in spins :
                h0 += FermionOperator(f"{inds(i,s1,N)}^ {inds(j,s1,N)} ", eimp[i, j])

        return h0, hi, 2*N

def Hsiam(U, eimp, use_open_fermion=False):
    """
    contruct H from parameters read in param_3
    """
    norb = eimp.shape[0]
    N = norb
    nimp = U.shape[0]

    if not use_open_fermion:
        h0 = SOP()
        hi = SOP()

        spins = [-1,1]
        hi += U[0,0,0,0]*fOp("n", inds(0, 1, N))*fOp("n", inds(0, -1, N))
        for i, j in product(range(norb), range(norb)):
            for s1 in spins :
                h0 += eimp[i,j]*fOp("cdag", inds(i, s1, N))*fOp("c", inds(j, s1, N))
        return h0, hi, 2*N

    else:
        hi, h0 = 0, 0
        spins = [-1,1]
        hi += FermionOperator(f"{inds(0,1,N)}^  {inds(0, 1, N)}  {inds(0, -1,N)}^ {inds(0, -1,N)} ", U[0,0,0,0] )

        for i, j in product(range(norb), range(norb)):
            for s1 in spins :
                h0 += FermionOperator(f"{inds(i,s1,N)}^ {inds(j,s1,N)} ", eimp[i, j])

        return h0, hi, 2*N

def hamiltonian(e, U, type='mpo'):
    if type == 'csr':
        return hamiltonian_csr(e, U)
    N = 0
    Es = 0
    h0, hi, N = Hsiam(U, e, use_open_fermion=False)
    h0.prune_zeros()
    hi.prune_zeros()
    H = h0+hi

    Hjw = H.jordan_wigner()
    si = Sites(N = N, site=FermionSite())

    if type == 'mpo':
        Hmpo = Hjw.asMPO(si, tol = 1e-12)
        return Hmpo, N, Es
    elif type == 'sop':
        Hop = Hjw.linear_operator(si)
        return Hop, N, Es


def Hdqd(U, Uc, eimp):
    """
    contruct H from parameters read in param_3
    """
    norb = eimp.shape[0]
    N = norb

    h0 = SOP()
    hi = SOP()

    spins = [-1,1]
    hi += U*fOp("n", inds(0, 1, N))*fOp("n", inds(0, -1, N))
    hi += U*fOp("n", inds(1, 1, N))*fOp("n", inds(1, -1, N))
    hi += Uc*fOp("n", inds(0, 1, N))*fOp("n", inds(1, 1, N))
    hi += Uc*fOp("n", inds(0, 1, N))*fOp("n", inds(1, -1, N))
    hi += Uc*fOp("n", inds(0, -1, N))*fOp("n", inds(1, 1, N))
    hi += Uc*fOp("n", inds(0, -1, N))*fOp("n", inds(1, -1, N))
    for i, j in product(range(norb), range(norb)):
        for s1 in spins :
            h0 += eimp[i,j]*fOp("cdag", inds(i, s1, N))*fOp("c", inds(j, s1, N))
    return h0, hi, 2*N


def hamiltonian_mpo_dqd(e, U, Uc):
    Hmpo = None
    N = 0
    Es = 0
    h0, hi, N = Hdqd(U, Uc, e)
    h0.prune_zeros()
    hi.prune_zeros()
    H = h0+hi

    H.prune_zeros()
    Hjw = H.jordan_wigner()
    si = Sites(N = N, site=FermionSite())
    Hmpo = Hjw.asMPO(si, tol = 1e-12)
    print(Hmpo.maximum_bond_dimension())
    return Hmpo, N, Es

def number_operators(N, bath_operators = False):
    Nu = SOP()
    Nd = SOP()
    Ntot = SOP()
    Nub = SOP()
    Ndb = SOP()
    Nu += fOp("n", inds(0, 1, N))
    Nd += fOp("n", inds(0, -1, N))

    for i in range(N*2):
        Ntot += fOp("n", i)
    for i in range(N):
        Nub += fOp("n", inds(i, 1, N))
        Ndb += fOp("n", inds(i, -1, N))

    Nujw = Nu.jordan_wigner()
    Ndjw = Nd.jordan_wigner()

    Nubjw = Nub.jordan_wigner()
    Ndbjw = Ndb.jordan_wigner()
    Ntotjw = Ntot.jordan_wigner()
    si = Sites(N = 2*N, site=FermionSite())

    if(bath_operators):
        return Nujw.asMPO(si, tol = 1e-12), Ndjw.asMPO(si, tol = 1e-12), Nubjw.asMPO(si, tol=1e-12), Ndbjw.asMPO(si, tol=1e-12), Ntotjw.asMPO(si, tol = 1e-12)
    else:
        return Nujw.asMPO(si, tol = 1e-12), Ndjw.asMPO(si, tol = 1e-12), Ntotjw.asMPO(si, tol = 1e-12)


def constraint(N):
    Nub = SOP()
    Ndb = SOP()

    si = Sites(N = 2*N, site=FermionSite())
    for i in range(N):
        Nub += fOp("n", inds(i, 1, N))
        Nub += fOp("n", inds(i, -1, N))
        Nub -= 2*fOp("n", inds(i, -1, N))*fOp("n", inds(i, 1, N))

    Cs = Nub
    Csjw = Cs.jordan_wigner()
    return Csjw.asMPO(si, tol = 1e-12)



def impurity_creation_and_annihilation_operators(spin, N, type='mpo'):
    if type == 'csr':
        return impurity_creation_and_annihilation_operators_csr(spin, N)
    cd = SOP()
    c = SOP()
    cd += fOp("cdag", inds(0, spin, N))
    c += fOp("c", inds(0, spin, N))

    cdjw = cd.jordan_wigner()
    cjw = c.jordan_wigner()
    si = Sites(N = 2*N, site=FermionSite())
    if type == 'mpo':
        cdmpo = cdjw.asMPO(si, tol = 1e-12)
        cmpo = cjw.asMPO(si, tol = 1e-12)
        return cdmpo, cmpo

    elif type == 'sop':
        cdmpo = cdjw.linear_operator(si)
        cmpo = cjw.linear_operator(si)
        return cdmpo, cmpo


def hamiltonian_csr(e, U):
    N = 0
    Es = 0
    h0, hi, N = Hsiam(U, e, use_open_fermion=True)
    H = get_sparse_operator(jordan_wigner(h0+hi), 2*e.shape[0])

    return H, N, Es


def impurity_creation_and_annihilation_operators_csr(spin, N):
    cd = jordan_wigner(FermionOperator(f"{inds(0,spin,N)}^", 1.0))
    c  = jordan_wigner(FermionOperator(f"{inds(0,spin,N)}", 1.0))

    return  get_sparse_operator(cd, 2*N), get_sparse_operator(c, 2*N)
