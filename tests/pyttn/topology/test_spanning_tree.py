from pyttn import generate_spanning_tree, convert_nx_to_tree, convert_nx_to_subtree, ntree
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"


def test_spannig_tree_msb_1():
    # set up a matrix corresponding to the coupling matrix of modes in a set 
    # of star topology spin boson models with linear coupling between spin degrees of 
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom

    N = 4
    M = 3
    #coupling between spin and bosonic modes
    v = N-np.arange(N)

    #bosonic frequencies
    w = np.diag(np.ones(N))
    gamma = np.zeros((N+1, N+1))

    #set up the coupling matrix for each spin
    gamma[0, 1:] = v
    gamma[1:, 0] = v
    gamma[1:, 1:] = w

    #and construct a coupling matrix for all of the spins
    G2 = np.zeros((M*(N+1),M*(N+1)))
    for i in range(M):
        G2[i*(N+1):(i+1)*(N+1), i*(N+1):(i+1)*(N+1)]=gamma

    #and add on the spin-spin coupling terms
    for i in range(M-1):
        G2[i*(N+1), (i+1)*(N+1)] = 0.01
        G2[(i+1)*(N+1), i*(N+1)] = 0.01

    #now build the networkx spanning tree associated with this tree 
    #here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, max_nchild=1, root_index=0)

    #now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)
