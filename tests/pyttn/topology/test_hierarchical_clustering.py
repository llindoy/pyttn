from pyttn import generate_hierarchical_clustering_tree, convert_nx_to_tree, convert_nx_to_subtree, ntree
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"


def test_hierarchical_clustering_tree():
    # set up for the distance matrix corresponding to the dendrogram
    #
    #        --------------
    #       |              |
    #      ---         ----------
    #     |   |       |          |
    #     1  ---      3        -----
    #       |   |             |     |
    #       5   6            ---   ---
    #                       |   | |   |
    #                       2   7 0   4
    X = [2, 8, 0, 4, 1, 9, 9, 0]
    M = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            M[i, j] = np.abs(X[i]-X[j])

    M = np.max(M) - M

    # build the networkx tree from this
    nxtree, root_ind = generate_hierarchical_clustering_tree(M, distance_metric='sub')

    assert root_ind == 14
    assert nxtree.number_of_nodes() == 15
    assert nxtree.number_of_edges() == 14

    # build the ntree object from this tree
    #        ------14------
    #       |              |
    #      -11-         ---13---
    #     |    |       |        |
    #     1   -8-      3      --12--
    #        |   |           |      |
    #        5   6          -9-    -10-
    #                      |   |  |    |
    #                      2   7  0    4
    tree, leaf_ordering = convert_nx_to_tree(nxtree, root_ind=root_ind)

    # check that the leaf ordering is correct
    assert leaf_ordering == [1, 5, 6, 3, 2, 7, 0, 4]

    # check that the nxtree also has this ordering
    for i, linds in enumerate(tree.leaf_indices()):
        assert tree.at(linds).value == leaf_ordering[i]

    assert tree.root().value == 14
    assert tree.at([0]).value == 11
    assert tree.at([0, 0]).value == 1
    assert tree.at([0, 1]).value == 8
    assert tree.at([0, 1, 0]).value == 5
    assert tree.at([0, 1, 1]).value == 6
    assert tree.at([1]).value == 13
    assert tree.at([1, 0]).value == 3
    assert tree.at([1, 1]).value == 12
    assert tree.at([1, 1, 0]).value == 9
    assert tree.at([1, 1, 0, 0]).value == 2
    assert tree.at([1, 1, 0, 1]).value == 7
    assert tree.at([1, 1, 1, 0]).value == 0
    assert tree.at([1, 1, 1, 1]).value == 4


def test_hierarchical_clustering_subtree():
    # set up for the distance matrix corresponding to the dendrogram
    #
    #        --------------
    #       |              |
    #      ---         ----------
    #     |   |       |          |
    #     1  ---      3        -----
    #       |   |             |     |
    #       5   6            ---   ---
    #                       |   | |   |
    #                       2   7 0   4
    X = [2, 8, 0, 4, 1, 9, 9, 0]
    M = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            M[i, j] = np.abs(X[i]-X[j])

    M = np.max(M) - M

    # build the networkx tree from this
    nxtree, root_ind = generate_hierarchical_clustering_tree(M, distance_metric='sub')

    assert root_ind == 14
    assert nxtree.number_of_nodes() == 15
    assert nxtree.number_of_edges() == 14

    # build the ntree object from this tree
    #   ----------111
    #  |           |
    # 121    ------14------
    #       |              |
    #      -11-         ---13---
    #     |    |       |        |
    #     1   -8-      3      --12--
    #        |   |           |      |
    #        5   6          -9-    -10-
    #                      |   |  |    |
    #                      2   7  0    4
    tree = ntree('(111(121))')
    leaf_ordering = convert_nx_to_subtree(nxtree, tree(), root_ind=root_ind)

    # check that the leaf ordering is correct
    assert leaf_ordering == [1, 5, 6, 3, 2, 7, 0, 4]

    tree_leaves = [121, 1, 5, 6, 3, 2, 7, 0, 4]
    # check that the nxtree also has this ordering
    for i, linds in enumerate(tree.leaf_indices()):
        assert tree.at(linds).value == tree_leaves[i]

    assert tree.root().value == 111
    assert tree.at([0]).value == 121
    assert tree.at([1]).value == 14
    assert tree.at([1, 0]).value == 11
    assert tree.at([1, 0, 0]).value == 1
    assert tree.at([1, 0, 1]).value == 8
    assert tree.at([1, 0, 1, 0]).value == 5
    assert tree.at([1, 0, 1, 1]).value == 6
    assert tree.at([1, 1]).value == 13
    assert tree.at([1, 1, 0]).value == 3
    assert tree.at([1, 1, 1]).value == 12
    assert tree.at([1, 1, 1, 0]).value == 9
    assert tree.at([1, 1, 1, 0, 0]).value == 2
    assert tree.at([1, 1, 1, 0, 1]).value == 7
    assert tree.at([1, 1, 1, 1, 0]).value == 0
    assert tree.at([1, 1, 1, 1, 1]).value == 4
