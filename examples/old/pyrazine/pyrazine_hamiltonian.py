import numpy as np
import sys

sys.path.append("../../")
from pyttn import *
from pyttn._pyttn import operator_dictionary_complex


#convert from eV to hartree
eV = 0.0367493049512081

def modes():
    return ['el', 'v10a', 'v6a', 'v1', 'v9a', 'v8a', 'v2', 'v6b', 'v8b', 'v4', 'v5', 'v3', 'v16a', 'v12', 'v13', 'v19b', 'v18b', 'v18a', 'v14', 'v19a', 'v17a', 'v20b', 'v16b' , 'v11', 'v7b']

def mode_index(mode):
    return modes().index(mode)

def vibrational_mode_index(mode):
    return modes().index(mode)-1

def Ag():
    return ['v6a', 'v1', 'v9a', 'v8a', 'v2']

def B1g():
    return ['v10a']

def B2g():
    return ['v4' , 'v5']

def B3g():
    return ['v6b', 'v3', 'v8b', 'v7b']

def Au():
    return ['v16a', 'v17a']

def B1u():
    return ['v12', 'v18a', 'v19a', 'v13']

def B2u():
    return ['v18b', 'v14', 'v19b', 'v20b']

def B3u():
    return ['v16b', 'v11']


def G1():
    return Ag()

def G2(): 
    return [ Ag(), Au(), B1g(), B2g(), B3g(), B1u(), B2u(), B3u()]

def G3():
    return B1g()

def G4():
    G4i = [B1g(), B2g(), Au(), B3u()]
    G4j = [Ag(), B3g(), B1u(), B2u()]
    return G4i, G4j

def w():
    return {
        'v6a' : 0.0739*eV, 
        'v1'  : 0.1258*eV, 
        'v9a' : 0.1525*eV, 
        'v8a' : 0.1961*eV, 
        'v2'  : 0.3788*eV, 
        'v10a': 0.1139*eV,
        'v4'  : 0.0937*eV, 
        'v5'  : 0.1219*eV,
        'v6b' : 0.0873*eV, 
        'v3'  : 0.1669*eV, 
        'v8b' : 0.1891*eV, 
        'v7b' : 0.3769*eV,
        'v16a': 0.0423*eV, 
        'v17a': 0.1189*eV,
        'v12' : 0.1266*eV, 
        'v18a': 0.1408*eV, 
        'v19a': 0.1840*eV, 
        'v13' : 0.3734*eV,
        'v18b': 0.1318*eV, 
        'v14' : 0.1425*eV, 
        'v19b': 0.1756*eV, 
        'v20b': 0.3798*eV,
        'v16b': 0.0521*eV, 
        'v11' : 0.0973*eV
    }


def ai():
    return np.array([-0.0981,-0.0503, 0.1452,-0.0445, 0.0247])*eV

def bi():
    return np.array([ 0.1355,-0.1710, 0.0375, 0.0168, 0.0162])*eV

def aij():
    ret = [
        np.array(
            [[       0, 0.00108,-0.00204,-0.00135,-0.00285], 
             [       0,       0, 0.00474, 0.00154,-0.00163],
             [       0,       0,       0, 0.00872,-0.00474],
             [       0,       0,       0,       0,-0.00143],
             [       0,       0,       0,       0,       0]]
        )*eV,
        np.array(
            [[ 0.01145, 0.00100],
             [       0,-0.02040]]
        )*eV,
        np.array(
            [[-0.01159]]
        )*eV,
        np.array(
            [[-0.02252,-0.00049],
             [       0,-0.01825]]
        )*eV,
        np.array(
            [[-0.00741, 0.01321,-0.00717, 0.00515],
             [       0, 0.05183,-0.03942, 0.00170],
             [       0,       0,-0.05733,-0.00204],
             [       0,       0,       0,-0.00333]]
        )*eV,

        np.array(
            [[-0.04819, 0.00525,-0.00485,-0.00326],
             [       0,-0.00792, 0.00852, 0.00888],
             [       0,       0,-0.02429,-0.00443],
             [       0,       0,       0,-0.00492]]

        )*eV,
        np.array(
            [[-0.00277, 0.00016,-0.00250, 0.00357],
             [       0, 0.03924,-0.00197,-0.00355],
             [       0,       0, 0.00992, 0.00623],
             [       0,       0,       0,-0.00110]]

        )*eV,
        np.array(
            [[-0.02176,-0.00624],
             [       0, 0.00315]]
        )*eV
    ]
    for i in range(len(ret)):
        ret[i] = ret[i] + ret[i].T
        np.fill_diagonal(ret[i], ret[i].diagonal()/2)
    return ret

def bij():
    ret = [
        np.array(
            [[       0,-0.00298,-0.00189,-0.00203,-0.00128], 
             [       0,       0, 0.00155, 0.00311,-0.00600],
             [       0,       0,       0, 0.01194,-0.00334],
             [       0,       0,       0,       0,-0.00713],
             [       0,       0,       0,       0,       0]]
        )*eV,
        np.array(
            [[-0.01459,-0.00091],
             [       0,-0.00618]]
        )*eV,
        np.array(
            [[-0.01159]]
        )*eV,

        np.array(
            [[-0.03445, 0.00911],
             [       0,-0.00265]]
        )*eV,
        np.array(
            [[-0.00385,-0.00661, 0.00429,-0.00246],
             [       0, 0.04842,-0.03034,-0.00185],
             [       0,       0,-0.06332,-0.00388],
             [       0,       0,       0,-0.00040]]
        )*eV,

        np.array(
            [[-0.00840, 0.00536,-0.00097, 0.00034],
             [       0, 0.00429, 0.00209,-0.00049],
             [       0,       0,-0.00734, 0.00346],
             [       0,       0,       0, 0.00062]]

        )*eV,
        np.array(
            [[-0.01179,-0.00844, 0.07000,-0.01249],
             [       0, 0.04000,-0.05000, 0.00265],
             [       0,       0, 0.01246,-0.00422],
             [       0,       0,       0, 0.00069]]

        )*eV,
        np.array(
            [[-0.02214,-0.00261],
             [       0,-0.00496]]
        )*eV
    ]
    for i in range(len(ret)):
        ret[i] = ret[i] + ret[i].T
        np.fill_diagonal(ret[i], ret[i].diagonal()/2)
    return ret


def ci():
    return np.array([0.2080])*eV


def cij():
    return [
        np.array(
            [[-0.01000,-0.00551, 0.00127, 0.00799,-0.00512]]
        )*eV,
        np.array(
            [[-0.01372,-0.00466, 0.00329,-0.00031],
             [ 0.00598,-0.00914, 0.00961, 0.00500]]
        )*eV,
        np.array(
            [[-0.01056, 0.00559, 0.00401,-0.00226],
             [-0.01200,-0.00213, 0.00328,-0.00396]]
        )*eV,
        np.array(
            [[ 0.00118,-0.00009,-0.00285,-0.00095],
             [ 0.01281,-0.01780, 0.00134,-0.00481]]
        )*eV
    ]


def hamiltonian(mode_dims):
    N = len(modes())
    H = SOP(N)
    omegas = w()
    vs = modes()[1:]
    
    delta = 0.8460/2.0*eV
    #set up the system information object
    sysinf = system_modes(N)
    sysinf[0] = generic_mode(2)
    for i in range(N-1):
        sysinf[i+1] = boson_mode(mode_dims[i])

    #set up the system operator dictionary
    opdict = operator_dictionary_complex(N)
    for i in range(2):
        v = np.zeros( (2,2), dtype=np.complex128)
        v[i, i] = 1.0;
        op =  site_operator(v, optype="matrix", mode=0)
        opdict.insert(0, "|%d><%d|"%(i, i), op)

    v = np.zeros( (2,2), dtype=np.complex128)
    v[0, 0] = 1.0;
    v[1, 1] = -1.0
    op =  site_operator(v, optype="matrix", mode=0)
    opdict.insert(0, "sz", op)

    v = np.zeros( (2,2), dtype=np.complex128)
    v[0, 1] = 1.0;
    v[1, 0] = 1.0
    op =  site_operator(v, optype="matrix", mode=0)
    opdict.insert(0, "sx", op)

    #energy spacing of the S1 and S2 state
    H += -delta*sOP("sz", 0)

    #add on the vibrational mode terms
    for v in vs:
        H += omegas[v]*sOP("n", mode_index(v))

    #the linear on-diagonal couplings
    for i, g in enumerate(G1()):
        H += ai()[i]*sOP("|0><0|", 0)*sOP("q", mode_index(g))
        H += bi()[i]*sOP("|1><1|", 0)*sOP("q", mode_index(g))

    #the quadratic on-diagonal couplings
    for g, Aij, Bij in zip(G2(), aij(), bij()):
        for i, vi in enumerate(g):
            for j, vj in enumerate(g):
                print(vi, vj, mode_index(vi), mode_index(vj))
                H += Aij[i, j]*sOP("|0><0|", 0)*sOP("q", mode_index(vi))*sOP("q", mode_index(vj))
                H += Bij[i, j]*sOP("|1><1|", 0)*sOP("q", mode_index(vi))*sOP("q", mode_index(vj))

    #the linear off-diagonal couplings
    for i, g in enumerate(G3()):
        H += ci()[i]*sOP("sx", 0)*sOP("q", mode_index(g))

    #the quadratic on-diagonal couplings
    G4i, G4j = G4()
    for gi, gj, Cij in zip(G4i, G4j, cij()):
        for i, vi in enumerate(gi):
            for j, vj in enumerate(gj):
                H += Cij[i, j]*sOP("sx", 0)*sOP("q", mode_index(vi))*sOP("q", mode_index(vj))
    print(H)
    return H, sysinf, opdict

def multiset_hamiltonian(mode_dims):
    N = len(modes())
    Nb = len(mode_dims)
    H = multiset_SOP(2, Nb)
    omegas = w()
    vs = modes()[1:]

    delta = 0.8460/2.0*eV
    #set up the system information object
    sysinf = system_modes(Nb)
    for i in range(Nb):
        sysinf[i] = boson_mode(mode_dims[i])

    #now set up the Hamiltonian
    H[0,0] -= delta
    H[0,1] += 0
    H[1,0] += 0
    H[1,1] += delta
    for v in vs:
        H[0,0] += omegas[v]*sOP("n", vibrational_mode_index(v))
        H[1,1] += omegas[v]*sOP("n", vibrational_mode_index(v))

    #the linear on-diagonal couplings
    for i, g in enumerate(G1()):
        H[0,0] += ai()[i]*sOP("q", vibrational_mode_index(g))
        H[1,1] += bi()[i]*sOP("q", vibrational_mode_index(g))

    #the quadratic on-diagonal couplings
    for g, Aij, Bij in zip(G2(), aij(), bij()):
        for i, vi in enumerate(g):
            for j, vj in enumerate(g):
                print(vi, vj, vibrational_mode_index(vi), vibrational_mode_index(vj))
                H[0,0] += Aij[i, j]*sOP("q", vibrational_mode_index(vi))*sOP("q", vibrational_mode_index(vj))
                H[1,1] += Bij[i, j]*sOP("q", vibrational_mode_index(vi))*sOP("q", vibrational_mode_index(vj))

    #the linear off-diagonal couplings
    for i, g in enumerate(G3()):
        H[0,1] += ci()[i]*sOP("q", vibrational_mode_index(g))
        H[1,0] += ci()[i]*sOP("q", vibrational_mode_index(g))

    #the quadratic on-diagonal couplings
    G4i, G4j = G4()
    for gi, gj, Cij in zip(G4i, G4j, cij()):
        for i, vi in enumerate(gi):
            for j, vj in enumerate(gj):
                H[0,1] += Cij[i, j]*sOP("q", vibrational_mode_index(vi))*sOP("q", vibrational_mode_index(vj))
                H[1,0] += Cij[i, j]*sOP("q", vibrational_mode_index(vi))*sOP("q", vibrational_mode_index(vj))
    print(H)
    return H, sysinf
