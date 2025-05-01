from pyttn import generate_spanning_tree, convert_nx_to_tree
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

def coupling_matrix_star(N, M):
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
    return G2

def coupling_matrix_chain(N, M):
    #coupling between spin and bosonic modes
    v = N-np.arange(N)

    #bosonic frequencies
    w = np.diag(np.ones(N))
    gamma = np.zeros((N+1, N+1))

    #set up the coupling matrix for each spin
    gamma[1:, 1:] = w
    for j in range(N):
        gamma[j, j+1] = v[j]
        gamma[j+1, j] = v[j]

    #and construct a coupling matrix for all of the spins
    G2 = np.zeros((M*(N+1),M*(N+1)))
    for i in range(M):
        G2[i*(N+1):(i+1)*(N+1), i*(N+1):(i+1)*(N+1)]=gamma

    #and add on the spin-spin coupling terms
    for i in range(M-1):
        G2[i*(N+1), (i+1)*(N+1)] = 0.01
        G2[(i+1)*(N+1), i*(N+1)] = 0.01
    return G2

def test_spanning_tree_msb_star_1():
    # set up a matrix corresponding to the coupling matrix of modes in a set 
    # of star topology spin boson models with linear coupling between spin degrees of 
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom.  In this case we don't restrict the number of children a 
    # node can have
    N = 4
    M = 3

    G2 = coupling_matrix_star(N, M)

    #now build the networkx spanning tree associated with this tree 
    #here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, root_index=0)

    assert spanning_tree.number_of_nodes() == 18
    assert spanning_tree.number_of_edges() == 17

    #now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)

    assert leaf_ordering == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0]
    assert tree.root().value == 15
    assert tree.at([0]).value == 1
    assert tree.at([1]).value == 2
    assert tree.at([2]).value == 3
    assert tree.at([3]).value == 4
    assert tree.at([4]).value == 16
    assert tree.at([5]).value == 0

    assert tree.at([4, 0]).value == 6
    assert tree.at([4, 1]).value == 7
    assert tree.at([4, 2]).value == 8
    assert tree.at([4, 3]).value == 9
    assert tree.at([4, 4]).value == 17
    assert tree.at([4, 5]).value == 5

    assert tree.at([4, 4, 0]).value == 11
    assert tree.at([4, 4, 1]).value == 12
    assert tree.at([4, 4, 2]).value == 13
    assert tree.at([4, 4, 3]).value == 14
    assert tree.at([4, 4, 4]).value == 10

    
def test_spanning_tree_msb_star_2():
    # set up a matrix corresponding to the coupling matrix of modes in a set 
    # of star topology spin boson models with linear coupling between spin degrees of 
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom.  In this case we don't restrict the number of children a 
    # node can have
    N = 4
    M = 3

    G2 = coupling_matrix_star(N, M)

    #now build the networkx spanning tree associated with this tree 
    #here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, max_internal_children=1, max_leaf_children=1, root_index=0)

    assert spanning_tree.number_of_nodes() == 27
    assert spanning_tree.number_of_edges() == 26

    #now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)

    assert leaf_ordering == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0]
    assert tree.root().value == 24
    assert tree.at([0]).value == 15
    assert tree.at([0, 0]).value == 1
    assert tree.at([0, 1]).value == 16
    assert tree.at([0, 1, 0]).value == 2
    assert tree.at([0, 1, 1]).value == 17
    assert tree.at([0, 1, 1, 0]).value == 3
    assert tree.at([0, 1, 1, 1]).value == 4

    assert tree.at([1]).value == 25
    assert tree.at([1, 0]).value == 18
    assert tree.at([1, 0, 0]).value == 6
    assert tree.at([1, 0, 1]).value == 19
    assert tree.at([1, 0, 1, 0]).value == 7
    assert tree.at([1, 0, 1, 1]).value == 20
    assert tree.at([1, 0, 1, 1, 0]).value == 8
    assert tree.at([1, 0, 1, 1, 1]).value == 9

    assert tree.at([1, 1]).value == 26
    assert tree.at([1, 1, 0]).value == 21
    assert tree.at([1, 1, 0, 0]).value == 11
    assert tree.at([1, 1, 0, 1]).value == 22
    assert tree.at([1, 1, 0, 1, 0]).value == 12
    assert tree.at([1, 1, 0, 1, 1]).value == 23
    assert tree.at([1, 1, 0, 1, 1, 0]).value == 13
    assert tree.at([1, 1, 0, 1, 1, 1]).value == 14

    assert tree.at([1, 1, 1]).value == 10
    assert tree.at([1, 2]).value == 5
    assert tree.at([2]).value == 0

def test_spanning_tree_msb_star_3():
    # set up a matrix corresponding to the coupling matrix of modes in a set 
    # of star topology spin boson models with linear coupling between spin degrees of 
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom.  In this case we don't restrict the number of children a 
    # node can have
    N = 4
    M = 3

    G2 = coupling_matrix_star(N, M)

    #now build the networkx spanning tree associated with this tree 
    #here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, max_internal_children=2, max_leaf_children=2, root_index=0)

    assert spanning_tree.number_of_nodes() == 27
    assert spanning_tree.number_of_edges() == 26

    #now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)

    assert leaf_ordering == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0]
    assert tree.root().value == 24
    assert tree.at([0]).value == 15
    assert tree.at([0, 0]).value == 16
    assert tree.at([0, 0, 0]).value == 1
    assert tree.at([0, 0, 1]).value == 2
    assert tree.at([0, 1]).value == 17
    assert tree.at([0, 1, 0]).value == 3
    assert tree.at([0, 1, 1]).value == 4

    assert tree.at([1]).value == 25
    assert tree.at([1, 0]).value == 18

    assert tree.at([1, 0, 0]).value == 19
    assert tree.at([1, 0, 0, 0]).value == 6
    assert tree.at([1, 0, 0, 1]).value == 7
    assert tree.at([1, 0, 1]).value == 20
    assert tree.at([1, 0, 1, 0]).value == 8
    assert tree.at([1, 0, 1, 1]).value == 9


    assert tree.at([1, 1]).value == 26
    
    assert tree.at([1, 1, 0]).value == 21
    assert tree.at([1, 1, 0, 0]).value == 22
    assert tree.at([1, 1, 0, 0, 0]).value == 11
    assert tree.at([1, 1, 0, 0, 1]).value == 12
    assert tree.at([1, 1, 0, 1]).value == 23
    assert tree.at([1, 1, 0, 1, 0]).value == 13
    assert tree.at([1, 1, 0, 1, 1]).value == 14

    assert tree.at([1, 1, 1]).value == 10
    assert tree.at([1, 2]).value == 5
    assert tree.at([2]).value == 0
        

def test_spanning_tree_msb_star_4():
    # set up a matrix corresponding to the coupling matrix of modes in a set 
    # of star topology spin boson models with linear coupling between spin degrees of 
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom.  In this case we don't restrict the number of children a 
    # node can have
    N = 4
    M = 3

    G2 = coupling_matrix_star(N, M)

    #now build the networkx spanning tree associated with this tree 
    #here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, max_internal_children=2, root_index=0)

    assert spanning_tree.number_of_nodes() == 26
    assert spanning_tree.number_of_edges() == 25

    #now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)

    assert leaf_ordering == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0]
    assert tree.root().value == 23

    assert tree.at([0]).value == 15
    assert tree.at([0,0]).value == 16
    assert tree.at([0,0,0]).value == 1
    assert tree.at([0,0,1]).value == 2
    assert tree.at([0,1]).value == 3
    assert tree.at([1]).value == 17
    assert tree.at([1,0]).value == 4
    assert tree.at([1,1]).value == 24
    assert tree.at([1,1,0]).value == 18
    assert tree.at([1,1,0,0]).value == 19
    assert tree.at([1,1,0,0,0]).value == 6
    assert tree.at([1,1,0,0,1]).value == 7
    assert tree.at([1,1,0,1]).value == 8
    assert tree.at([1,1,1]).value == 20
    assert tree.at([1,1,1,0]).value == 9
    assert tree.at([1,1,1,1]).value == 25
    assert tree.at([1,1,1,1,0]).value == 21
    assert tree.at([1,1,1,1,0,0]).value == 11
    assert tree.at([1,1,1,1,0,1]).value == 12
    assert tree.at([1,1,1,1,1]).value == 22
    assert tree.at([1,1,1,1,1,0]).value == 13
    assert tree.at([1,1,1,1,1,1]).value == 14
    assert tree.at([1,1,1,1,2]).value == 10
    assert tree.at([1,1,2]).value == 5
    assert tree.at([2]).value == 0


def test_spanning_tree_msb_star_5():
    # set up a matrix corresponding to the coupling matrix of modes in a set 
    # of star topology spin boson models with linear coupling between spin degrees of 
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom.  In this case we don't restrict the number of children a 
    # node can have
    N = 4
    M = 3

    G2 = coupling_matrix_star(N, M)

    #now build the networkx spanning tree associated with this tree 
    #here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, max_leaf_children=2, root_index=0)

    assert spanning_tree.number_of_nodes() == 27
    assert spanning_tree.number_of_edges() == 26

    #now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)    

    assert leaf_ordering == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0]
    assert tree.root().value == 24
    assert tree.at([0]).value == 15
    assert tree.at([0, 0]).value == 16
    assert tree.at([0, 0, 0]).value == 1
    assert tree.at([0, 0, 1]).value == 2
    assert tree.at([0, 1]).value == 17
    assert tree.at([0, 1, 0]).value == 3
    assert tree.at([0, 1, 1]).value == 4

    assert tree.at([1]).value == 25
    assert tree.at([1, 0]).value == 18

    assert tree.at([1, 0, 0]).value == 19
    assert tree.at([1, 0, 0, 0]).value == 6
    assert tree.at([1, 0, 0, 1]).value == 7
    assert tree.at([1, 0, 1]).value == 20
    assert tree.at([1, 0, 1, 0]).value == 8
    assert tree.at([1, 0, 1, 1]).value == 9

    assert tree.at([1, 1]).value == 26
    
    assert tree.at([1, 1, 0]).value == 21
    assert tree.at([1, 1, 0, 0]).value == 22
    assert tree.at([1, 1, 0, 0, 0]).value == 11
    assert tree.at([1, 1, 0, 0, 1]).value == 12
    assert tree.at([1, 1, 0, 1]).value == 23
    assert tree.at([1, 1, 0, 1, 0]).value == 13
    assert tree.at([1, 1, 0, 1, 1]).value == 14

    assert tree.at([1, 1, 1]).value == 10
    assert tree.at([1, 2]).value == 5
    assert tree.at([2]).value == 0
    
def test_spanning_tree_msb_chain():
    # set up a matrix corresponding to the coupling matrix of modes in a set 
    # of star topology spin boson models with linear coupling between spin degrees of 
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom.  In this case we don't restrict the number of children a 
    # node can have
    N = 4
    M = 3

    G2 = coupling_matrix_chain(N, M)

    #now build the networkx spanning tree associated with this tree 
    #here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, root_index=0)

    assert spanning_tree.number_of_nodes() == 27
    assert spanning_tree.number_of_edges() == 26

    #now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)

    assert leaf_ordering == [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert tree.root().value == 15

    assert tree.at([0]).value == 19
    assert tree.at([0,0]).value == 23
    assert tree.at([0,0,0]).value == 24
    assert tree.at([0,0,0,0]).value == 25
    assert tree.at([0,0,0,0,0]).value == 26
    assert tree.at([0,0,0,0,0,0]).value == 14
    assert tree.at([0,0,0,0,0,1]).value == 13
    assert tree.at([0,0,0,0,1]).value == 12
    assert tree.at([0,0,0,1]).value == 11
    assert tree.at([0,0,1]).value == 10
    assert tree.at([0,1]).value == 20
    assert tree.at([0,1,0]).value == 21
    assert tree.at([0,1,0,0]).value == 22
    assert tree.at([0,1,0,0,0]).value == 9
    assert tree.at([0,1,0,0,1]).value == 8
    assert tree.at([0,1,0,1]).value == 7
    assert tree.at([0,1,1]).value == 6
    assert tree.at([0,2]).value == 5
    assert tree.at([1]).value == 16
    assert tree.at([1,0]).value == 17
    assert tree.at([1,0,0]).value == 18
    assert tree.at([1,0,0,0]).value == 4
    assert tree.at([1,0,0,1]).value == 3
    assert tree.at([1,0,1]).value == 2
    assert tree.at([1,1]).value == 1
    assert tree.at([2]).value == 0
