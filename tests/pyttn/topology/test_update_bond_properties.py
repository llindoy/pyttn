import pytest
from pyttn import (
    generate_spanning_tree,
    generate_hierarchical_clustering_tree,
    convert_nx_to_tree,
    set_dims,
    set_bond_dimensions,
    set_topology_properties,
    NodeSumSetter,
    NodeIncrementSetter,
)
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.fixture
def hierarchical_tree():
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
            M[i, j] = np.abs(X[i] - X[j])

    # build the networkx tree from this
    nxtree, root_ind = generate_hierarchical_clustering_tree(M)

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
    return tree


@pytest.fixture
def spanning_tree():
    # set up a matrix corresponding to the coupling matrix of modes in a set
    # of star topology spin boson models with linear coupling between spin degrees of
    # freedom.  Here we include 4 bath sites per spin and treat 3 spin sites for a total
    # of 15 degrees of freedom.  In this case we don't restrict the number of children a
    # node can have
    N = 4
    M = 3

    v = N - np.arange(N)

    # bosonic frequencies
    w = np.diag(np.ones(N))
    gamma = np.zeros((N + 1, N + 1))

    # set up the coupling matrix for each spin
    gamma[0, 1:] = v
    gamma[1:, 0] = v
    gamma[1:, 1:] = w

    # and construct a coupling matrix for all of the spins
    G2 = np.zeros((M * (N + 1), M * (N + 1)))
    for i in range(M):
        G2[i * (N + 1) : (i + 1) * (N + 1), i * (N + 1) : (i + 1) * (N + 1)] = gamma

    # and add on the spin-spin coupling terms
    for i in range(M - 1):
        G2[i * (N + 1), (i + 1) * (N + 1)] = 0.01
        G2[(i + 1) * (N + 1), i * (N + 1)] = 0.01

    # now build the networkx spanning tree associated with this tree
    # here we build with max_nchild = 1 to ensure an mps representation of the tree
    spanning_tree, spanning_root_ind = generate_spanning_tree(G2, root_index=0)

    assert spanning_tree.number_of_nodes() == 18
    assert spanning_tree.number_of_edges() == 17

    # now convert this to an ntree object and construct the leaf_ordering
    tree, leaf_ordering = convert_nx_to_tree(spanning_tree, root_ind=spanning_root_ind)
    return tree


@pytest.mark.parametrize(
    "tree, dims, expected",
    [
        ("spanning_tree", 1, [1 for i in range(15)]),
        (
            "spanning_tree",
            [2 * i for i in range(15)],
            [2, 4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 20, 10, 0],
        ),
        ("hierarchical_tree", 1, [1 for i in range(8)]),
        ("hierarchical_tree", [2 * i for i in range(8)], [2, 10, 12, 6, 4, 14, 0, 8]),
    ],
)
def test_set_dims(request, tree, dims, expected):
    tree = request.getfixturevalue(tree)
    set_dims(tree, dims)

    for i, li in enumerate(tree.leaf_indices()):
        assert tree.at(li).value == expected[i]


def func(l):
    return l + 5


@pytest.mark.parametrize(
    "tree, bdims, ldims, bond_dims",
    [
        (
            "spanning_tree",
            1,
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0],
            [([], 1), ([4], 1), ([4, 4], 1)],
        ),
        (
            "spanning_tree",
            func,
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0],
            [([], 1), ([4], 6), ([4, 4], 7)],
        ),
        (
            "spanning_tree",
            NodeSumSetter(),
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0],
            [([], 1), ([4], 95), ([4, 4], 60)],
        ),
        (
            "spanning_tree",
            NodeIncrementSetter(5, combination="min"),
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0],
            [([], 1), ([4], 10), ([4, 4], 15)],
        ),
        (
            "spanning_tree",
            NodeIncrementSetter(5, combination="max"),
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0],
            [([], 1), ([4], 24), ([4, 4], 19)],
        ),
        (
            "spanning_tree",
            NodeIncrementSetter(5, combination="mean"),
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 10, 5, 0],
            [([], 1), ([4], 13), ([4, 4], 17)],
        ),
        (
            "hierarchical_tree",
            1,
            [1, 5, 6, 3, 2, 7, 0, 4],
            [
                ([], 1),
                ([0], 1),
                ([0, 1], 1),
                ([1], 1),
                ([1, 1], 1),
                ([1, 1, 0], 1),
                ([1, 1, 1], 1),
            ],
        ),
        (
            "hierarchical_tree",
            func,
            [1, 5, 6, 3, 2, 7, 0, 4],
            [
                ([], 1),
                ([0], 6),
                ([0, 1], 7),
                ([1], 6),
                ([1, 1], 7),
                ([1, 1, 0], 8),
                ([1, 1, 1], 8),
            ],
        ),
        (
            "hierarchical_tree",
            NodeSumSetter(),
            [1, 5, 6, 3, 2, 7, 0, 4],
            [
                ([], 1),
                ([0], 12),
                ([0, 1], 11),
                ([1], 16),
                ([1, 1], 13),
                ([1, 1, 0], 9),
                ([1, 1, 1], 4),
            ],
        ),
        (
            "hierarchical_tree",
            NodeIncrementSetter(5, combination="min"),
            [1, 5, 6, 3, 2, 7, 0, 4],
            [
                ([], 1),
                ([0], 6),
                ([0, 1], 10),
                ([1], 8),
                ([1, 1], 10),
                ([1, 1, 0], 7),
                ([1, 1, 1], 5),
            ],
        ),
        (
            "hierarchical_tree",
            NodeIncrementSetter(5, combination="max"),
            [1, 5, 6, 3, 2, 7, 0, 4],
            [
                ([], 1),
                ([0], 16),
                ([0, 1], 11),
                ([1], 22),
                ([1, 1], 17),
                ([1, 1, 0], 12),
                ([1, 1, 1], 9),
            ],
        ),
        (
            "hierarchical_tree",
            NodeIncrementSetter(5, combination="mean"),
            [1, 5, 6, 3, 2, 7, 0, 4],
            [
                ([], 1),
                ([0], 10),
                ([0, 1], 10),
                ([1], 13),
                ([1, 1], 13),
                ([1, 1, 0], 9),
                ([1, 1, 1], 7),
            ],
        ),
    ],
)
def test_set_bond_dims(request, tree, bdims, ldims, bond_dims):
    tree = request.getfixturevalue(tree)

    set_bond_dimensions(tree, bdims)
    for i, li in enumerate(tree.leaf_indices()):
        assert tree.at(li).value == ldims[i]

    for bind in bond_dims:
        assert tree.at(bind[0]).value == bind[1]


@pytest.mark.parametrize(
    "tree, bdims, dims, ldims, bond_dims",
    [
        (
            "spanning_tree",
            1,
            1,
            [1 for i in range(15)],
            [([], 1), ([4], 1), ([4, 4], 1)],
        ),
        (
            "spanning_tree",
            1,
            [2 * i for i in range(15)],
            [2, 4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 20, 10, 0],
            [([], 1), ([4], 1), ([4, 4], 1)],
        ),
        (
            "spanning_tree",
            func,
            1,
            [1 for i in range(15)],
            [([], 1), ([4], 6), ([4, 4], 7)],
        ),
        (
            "spanning_tree",
            func,
            [2 * i for i in range(15)],
            [2, 4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 20, 10, 0],
            [([], 1), ([4], 6), ([4, 4], 7)],
        ),
        (
            "spanning_tree",
            NodeSumSetter(),
            1,
            [1 for i in range(15)],
            [([], 1), ([4], 10), ([4, 4], 5)],
        ),
        (
            "spanning_tree",
            NodeSumSetter(),
            [2 * i for i in range(15)],
            [2, 4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 20, 10, 0],
            [([], 1), ([4], 190), ([4, 4], 120)],
        ),
        (
            "hierarchical_tree",
            1,
            1,
            [1 for i in range(8)],
            [
                ([], 1),
                ([0], 1),
                ([0, 1], 1),
                ([1], 1),
                ([1, 1], 1),
                ([1, 1, 0], 1),
                ([1, 1, 1], 1),
            ],
        ),
        (
            "hierarchical_tree",
            1,
            [2 * i for i in range(8)],
            [2, 10, 12, 6, 4, 14, 0, 8],
            [
                ([], 1),
                ([0], 1),
                ([0, 1], 1),
                ([1], 1),
                ([1, 1], 1),
                ([1, 1, 0], 1),
                ([1, 1, 1], 1),
            ],
        ),
        (
            "hierarchical_tree",
            func,
            1,
            [1 for i in range(8)],
            [
                ([], 1),
                ([0], 6),
                ([0, 1], 7),
                ([1], 6),
                ([1, 1], 7),
                ([1, 1, 0], 8),
                ([1, 1, 1], 8),
            ],
        ),
        (
            "hierarchical_tree",
            func,
            [2 * i for i in range(8)],
            [2, 10, 12, 6, 4, 14, 0, 8],
            [
                ([], 1),
                ([0], 6),
                ([0, 1], 7),
                ([1], 6),
                ([1, 1], 7),
                ([1, 1, 0], 8),
                ([1, 1, 1], 8),
            ],
        ),
        (
            "hierarchical_tree",
            NodeSumSetter(),
            1,
            [1 for i in range(8)],
            [
                ([], 1),
                ([0], 3),
                ([0, 1], 2),
                ([1], 5),
                ([1, 1], 4),
                ([1, 1, 0], 2),
                ([1, 1, 1], 2),
            ],
        ),
        (
            "hierarchical_tree",
            NodeSumSetter(),
            [2 * i for i in range(8)],
            [2, 10, 12, 6, 4, 14, 0, 8],
            [
                ([], 1),
                ([0], 24),
                ([0, 1], 22),
                ([1], 32),
                ([1, 1], 26),
                ([1, 1, 0], 18),
                ([1, 1, 1], 8),
            ],
        ),
    ],
)
def test_set_bond_properties(request, tree, bdims, dims, ldims, bond_dims):
    tree = request.getfixturevalue(tree)

    set_topology_properties(tree, bdims, dims)
    for i, li in enumerate(tree.leaf_indices()):
        assert tree.at(li).value == ldims[i]

    for bind in bond_dims:
        assert tree.at(bind[0]).value == bind[1]
