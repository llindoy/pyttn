import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../../")
from pyttn import ntree, ntreeBuilder
from numba import jit


def get_nskip(Nl, d=3):
    ret = 0
    for i in range(Nl):
        if i == 0:
            ret += 1
        else:
            ret += d*(d-1)**(i-1)
    return ret 


def get_index(ind, d=3):
    if len(ind) == 0:
        return 0
    else:
        c = get_nskip(len(ind))
        ns = [2**i for i in reversed(range(len(ind)))]
        ls = [d-1 for i in range(len(ind))]
        ls[0] = d

        for i in range(len(ind)):
            c += ind[i]*ns[i]
        return c


def get_spin_connectivity(Nl, d=3):
    topo = ntree('(1)')

    inds = []

    #build the cayley tree.  
    if Nl > 0:

        #add the first layer
        x = get_index([])
        for i in range(d):
            y = get_index([i])
            inds.append([x,y])
            topo().insert(1)

    
        for layer in range(1, Nl):
            #get all leaves
            leaf_indices=topo.leaf_indices()

            for li in leaf_indices:
                x = get_index(li)
                for i in range(d-1):
                    topo.at(li).insert(1)
                    y = get_index(li + [i])
                    inds.append([x,y])
    return inds, topo.size()


def build_topology(Nl, ds, chi, chiS,  chiB, nbose, b_mode_dims, degree, d=3):
    topo = ntree('(1)')

    #build the cayley tree.  
    if Nl > 0:

        #add the first layer
        for i in range(d):
            topo().insert(chi)

    
        for layer in range(1, Nl):
            #get all leaves
            leaf_indices=topo.leaf_indices()

            for li in leaf_indices:
                for i in range(d-1):
                    topo.at(li).insert(chi)

    #now get each site using a depth first search traversal and determine the indices associated with the site
    indices = []
    for i in topo.dfs():
        indices.append(i.index())

    #now iterating backwards through each sites.  Attach the spin and bath degrees of freedom as the first child of this site. 
    #by performing this process in reverse order we ensure we do not affect the ordering of progressive sites
    for ni in reversed(indices):
        topo.at(ni).insert_front(chiS)
        topo.at(ni)[0].insert(ds)
        if degree == 1:
            ntreeBuilder.mps_subtree(topo.at(ni)[0], b_mode_dims, chiB, max(min(chiB, nbose), 1))
        else:
            ntreeBuilder.mlmctdh_subtree(topo.at(ni)[0], b_mode_dims, degree, chiB, max(min(chiB, nbose), 1))

    ntreeBuilder.sanitise(topo)

    return topo
