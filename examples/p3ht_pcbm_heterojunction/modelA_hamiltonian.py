import numpy as np
import sys

sys.path.append("../../")
from pyttn import *
from pyttn._pyttn import operator_dictionary_complex


#convert from eV to hartree
eV = 0.0367493049512081
meV = eV/1000

def build_operator_dictionary(N):
    Nfragments = 1
    #set up the system operator dictionary
    opdict = operator_dictionary(N)

    #add on the onsite terms
    for i in range(Nfragments):
        v = np.zeros( (2*Nfragments, 2*Nfragments), dtype=np.complex128)
        v[2*i, 2*i] = 1.0;
        op =  site_operator(v, optype="matrix", mode=0)
        opdict.insert(0, "|LE%d><LE%d|"%(i, i), op)

        v2 = np.zeros( (2*Nfragments, 2*Nfragments), dtype=np.complex128)
        v2[2*i+1, 2*i+1] = 1.0;
        op2 =  site_operator(v2, optype="matrix", mode=0)
        opdict.insert(0, "|CS%d><CS%d|"%(i, i), op2)

    #add on the hopping terms
    for i in range(Nfragments-1):
        v = np.zeros( (2*Nfragments, 2*Nfragments), dtype=np.complex128)
        v[2*i, 2*(i+1)] = 1.0;
        v[2*(i+1), 2*i] = 1.0;
        op =  site_operator(v, optype="matrix", mode=0)
        opdict.insert(0, "LE_hop%d"%(i), op)

        v3 = np.zeros( (2*Nfragments, 2*Nfragments), dtype=np.complex128)
        v3[2*i+1, 2*(i+1)+1] = 1.0;
        v3[2*(i+1)+1, 2*i+1] = 1.0;
        op3 =  site_operator(v3, optype="matrix", mode=0)
        opdict.insert(0, "CS_hop%d"%(i), op3)

    v = np.zeros( (2*Nfragments,2*Nfragments), dtype=np.complex128)
    v[0, 1] = 1.0;
    v[1, 0] = 1.0;
    op =  site_operator(v, optype="matrix", mode=0)
    opdict.insert(0, "LE_CS_hop", op)

    return opdict


def ot_mode_index(Nf, n, l):
    if l < 6:
        return n*6 + l
    else:
        return Nf*6 + 2*n + l-6
    

def hamiltonian():
    eps_cs = -79.0*meV
    w = np.array([ 3.643, 7.286, 10.929, 14.573, 18.216, 21.859, 25.502, 29.145, 32.788, 36.431, 40.075, 43.718, 47.361, 51.004, 54.647, 58.29, 61.933, 65.577, 69.22, 72.863, 76.506, 80.149, 83.792, 87.435, 91.079, 94.722, 98.365, 102.008, 105.651, 109.294, 112.937, 116.581, 120.224, 123.867, 127.51, 131.153, 134.796, 138.439, 142.083, 145.726, 149.369, 153.012, 156.655, 160.298, 163.941, 167.585, 171.228, 174.871, 178.514, 182.157, 185.8, 189.443, 193.087, 196.73, 200.373, 204.016, 207.659, 211.302, 214.945, 218.589, 222.232, 225.875, 229.518, 233.161, 236.804, 240.447, 244.091, 247.734, 251.377, 255.02, 258.663, 262.306, 265.949, 269.593, 273.236, 276.879, 280.522, 284.165, 287.808, 291.451, 295.095, 298.738, 302.381, 306.024, 309.667, 313.31, 316.953, 320.597, 324.24, 327.883, 331.526, 335.169, 338.812, 342.455, 346.099, 349.742, 353.385, 357.028, 360.671])*meV

    g=np.array([ 2.511, 2.359, 2.347, 2.586, 3.19, 4.203, 5.224, 5.741, 5.572, 5.547, 6.578, 8.456, 9.935, 10.056, 9.147, 8.002, 7.379, 8.038, 10.582, 14.242, 17.279, 18.38, 17.698, 15.808, 13.623, 12.158, 11.779, 12.196, 13.061, 13.549, 12.606, 10.303, 8.069, 7.192, 7.63, 8.721, 9.858, 10.601, 10.599, 10.123, 10.344, 12.335, 15.285, 16.939, 16.095, 14.735, 15.279, 19.071, 26.827, 38.225, 47.272, 47.873, 43.415, 39.088, 34.622, 28.686, 22.148, 16.585, 12.443, 9.701, 8.142, 7.254, 6.554, 5.91, 5.362, 4.932, 4.586, 4.287, 4.02, 3.785, 3.578, 3.395, 3.23, 3.081, 2.945, 2.822, 2.709, 2.605, 2.509, 2.42, 2.338, 2.262, 2.19, 2.123, 2.061, 2.003, 1.948, 1.897, 1.848, 1.803, 1.761, 1.721, 1.685, 1.65, 1.619, 1.593, 1.573, 1.552, 1.53])*meV


    t=130*meV

    wr = 10*meV
    gr_cs1cs1 = 30/np.sqrt(2.0)*meV
    gr_lecs1 = -10/np.sqrt(2.0)*meV

    N = 1 + 1 + len(w)

    #construct the operator dictionary 
    opdict = build_operator_dictionary(N)

    #now build the Hamiltonian
    H = SOP(N)

    #add on the vibrational mode terms
    H += wr * sOP("n", 1)
    for i in range(len(w)):
        H += w[i] * sOP("n", 1 + 1 + i)

    #add on the purely electronic terms
    #the onsite energies
    H += eps_cs*sOP("|CS0><CS0|", 0)

    #now add on the hopping terms
    H += t*sOP("LE_CS_hop", 0)

    #now add on the vibronic couplings
    #first we do the vibronic couplings to the R mode
    H += np.sqrt(2)*gr_cs1cs1*sOP("|CS0><CS0|", 0)*sOP("q", 1)
    H += np.sqrt(2)*gr_lecs1*sOP("LE_CS_hop", 0)*sOP("q", 1)

    #now we add on the vibronic couplings to the fullerene super-particle modes
    for n in range(len(w)):
        H += np.sqrt(2)*g[n]*sOP("|CS0><CS0|", 0) * sOP("q", 1+1+n)


    return H, opdict

















